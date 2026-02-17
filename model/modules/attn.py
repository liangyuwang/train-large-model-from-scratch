import torch
import torch.nn as nn
import torch.nn.functional as F

from training.config import ModelConfig
from distributed import (
    parallel_state,
    ulysses_all_to_all,
)


def rope_impl(q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor, rope_theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Shared RoPE (Rotary Positional Embedding) implementation for Qwen3 family models.
    Supports both single and batched position_ids
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # Create frequency tensor
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=q.device) / head_dim))
    
    # Handle position_ids - support both single batch and multi-batch scenarios
    if position_ids.dim() > 1 and position_ids.shape[0] > 1:
        # Multi-batch case: handle each batch separately
        freqs = []
        for i in range(batch_size):
            if i < position_ids.shape[0]:
                batch_freqs = torch.outer(position_ids[i].float(), inv_freq)
            else:
                # Use first batch if not enough position_ids
                batch_freqs = torch.outer(position_ids[0].float(), inv_freq)
            freqs.append(batch_freqs)
        freqs = torch.stack(freqs, dim=0)  # [batch_size, seq_len, head_dim//2]
        
        # Create cos and sin - repeat to match full head_dim
        cos = torch.cos(freqs).unsqueeze(1).repeat(1, 1, 1, 2)  # [batch_size, 1, seq_len, head_dim]
        sin = torch.sin(freqs).unsqueeze(1).repeat(1, 1, 1, 2)  # [batch_size, 1, seq_len, head_dim]
    else:
        # Single batch case - take the first sequence if batched
        if position_ids.dim() > 1:
            t = position_ids[0].float()  # [seq_len] - use first batch
        else:
            t = position_ids.float()     # [seq_len]
        
        # Create position encodings
        freqs = torch.outer(t, inv_freq)    # [seq_len, head_dim//2]
        
        # Duplicate to create full cos/sin tensors
        cos = freqs.cos()  # [seq_len, head_dim//2]
        sin = freqs.sin()  # [seq_len, head_dim//2]
        
        # Expand to full head_dim by repeating each element
        cos = torch.stack([cos, cos], dim=-1).flatten(-2)  # [seq_len, head_dim]
        sin = torch.stack([sin, sin], dim=-1).flatten(-2)  # [seq_len, head_dim]
        
        # Reshape to match q and k dimensions: [1, 1, seq_len, head_dim]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    
    # Apply rotation
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


def gqa_impl(k: torch.Tensor, v: torch.Tensor, num_key_value_heads: int, num_attention_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    k,v: (B, H_kv, T, D)
    return: (B, H_q, T, D)
    """
    if num_key_value_heads == num_attention_heads:
        return k, v
    elif num_key_value_heads == 1:
        k = k.expand(-1, num_attention_heads, -1, -1)
        v = v.expand(-1, num_attention_heads, -1, -1)
        return k, v
    elif num_attention_heads % num_key_value_heads == 0:
        repeat_factor = num_attention_heads // num_key_value_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)
        return k, v
    else:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")


class Attention(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.device = torch.cuda.current_device()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.n_embd = config.hidden_size
        self.dropout = config.dropout
        self.pos = None
        # key, query, value projections for all heads, but in a batch        
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False, device=self.device)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False, device=self.device)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False, device=self.device)
        # output projection
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False, device=self.device)
        self._init_weights(config.seed)

    def _init_weights(self, base_seed: int):
        with torch.random.fork_rng(devices=[self.q_proj.weight.device]):
            torch.manual_seed(base_seed)
            torch.nn.init.normal_(self.q_proj.weight, mean=0.0, std=self.config.init_std)
        with torch.random.fork_rng(devices=[self.k_proj.weight.device]):
            torch.manual_seed(base_seed)
            torch.nn.init.normal_(self.k_proj.weight, mean=0.0, std=self.config.init_std)
        with torch.random.fork_rng(devices=[self.v_proj.weight.device]):
            torch.manual_seed(base_seed)
            torch.nn.init.normal_(self.v_proj.weight, mean=0.0, std=self.config.init_std)
        with torch.random.fork_rng(devices=[self.c_proj.weight.device]):
            torch.manual_seed(base_seed)
            torch.nn.init.normal_(self.c_proj.weight, mean=0.0, std=self.config.init_std)

    def forward(self, x: torch.Tensor):
        B, T_local, C = x.size()
        sp_group = parallel_state.get_sep_group()
        sp_size = parallel_state.get_sep_world_size()
        sp_rank = parallel_state.get_sep_rank()
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x) # (B, T, n_embd)
        q = q.view(B, T_local, self.num_attention_heads, self.head_dim).transpose(-2, -3) # (B, nh, T, hs)
        k = k.view(B, T_local, self.num_key_value_heads, self.head_dim).transpose(-2, -3) # (B, nh, T, hs)
        v = v.view(B, T_local, self.num_key_value_heads, self.head_dim).transpose(-2, -3) # (B, nh, T, hs)
        k, v = gqa_impl(k, v, self.num_key_value_heads, self.num_attention_heads)
        start_pos = sp_rank * T_local
        end_pos = start_pos + T_local
        if self.pos is None or self.pos.size(1) != T_local or self.pos[0, 0] != start_pos:
            self.pos = torch.arange(start_pos, end_pos, device=x.device).unsqueeze(0)
        q, k = rope_impl(q, k, self.pos)
        
        H = self.num_attention_heads
        if sp_size > 1:
            # sp all to all
            assert H % sp_size == 0
            H_local = H // sp_size
            qkv = torch.stack([q, k, v], dim=0)
            qkv = qkv.reshape(3, B, sp_size, H_local, T_local, self.head_dim)
            qkv = qkv.transpose(0, 2).contiguous()
            qkv = ulysses_all_to_all(qkv, sp_group)
            qkv = qkv.transpose(0, 2).contiguous()
            T_full = sp_size * T_local
            qkv = qkv.view(3, B, H_local, T_full, self.head_dim)
            q_local, k_local, v_local = qkv[0], qkv[1], qkv[2]
        else:
            q_local, k_local, v_local = q, k, v

        # local attention computation
        dropout_p = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q_local, k_local, v_local, 
            attn_mask=None, dropout_p=dropout_p, is_causal=True
        )

        if sp_size > 1:
            # sp all to all back
            y = y.view(B, H_local, sp_size, T_local, self.head_dim)
            y = y.transpose(0, 2).contiguous()
            y = ulysses_all_to_all(y, sp_group)
            y = y.transpose(0, 1).contiguous()
        y = y.view(B, H, T_local, self.head_dim)

        y = y.transpose(-2, -3).reshape(B, T_local, C)
        # output projection
        y = self.c_proj(y)
        return y