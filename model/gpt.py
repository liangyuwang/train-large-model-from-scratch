import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import GPTConfig
from .modules.attn import Attention
from .modules.mlp import MLP, MoE
from .modules.norm import LayerNorm

class Block(nn.Module):

    def __init__(self, config: GPTConfig, use_moe: bool = True, top_k: int = None):
        super().__init__()
        self.use_moe = use_moe
        self.ln_1 = LayerNorm(config)
        self.attn = Attention(config)
        self.ln_2 = LayerNorm(config)
        self.mlp = MoE(config, top_k) if use_moe else MLP(config)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        mlp_out = self.mlp(self.ln_2(x))
        x = x + mlp_out[0] if self.use_moe else x + mlp_out
        return x

# GPT-like Model

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.pos = None
        self.use_moe = config.use_moe
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([Block(config, self.use_moe) for layer in range(config.num_layer)])
        self.lnf = LayerNorm(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tied_lm_head:
            self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        x = self.wte(idx) # token embeddings of shape (B, T, n_embd)
        for block in self.blocks:
            x = block(x)
        x = self.lnf(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    def get_flops_per_fwd_bwd(self, batch_size, seq_len):
        """
        Approximate model FLOPs for forward and backward pass.
        """
        # Q, K, V projection (3D->D) + Out projection (D->D)
        qkv_out_flops = 8 * batch_size * seq_len * self.config.hidden_size * self.config.hidden_size  # 6BLD^2 + 2BLD^2
        # QK^T + softmaxV
        attn_matmul_flops = 4 * batch_size * seq_len * seq_len * self.config.hidden_size
        attn_flops = qkv_out_flops + attn_matmul_flops
        # FFN: SwiGLU = gate_proj(D->D_int) + up_proj(D->D_int) + elementwise + down_proj(D_int->D)
        intermediate_size = self.config.moe_intermediate_size if self.use_moe else self.config.intermediate_size
        ffn_flops = 2 * batch_size * seq_len * self.config.hidden_size * intermediate_size \
                + 2 * batch_size * seq_len * self.config.hidden_size * intermediate_size \
                + batch_size * seq_len * intermediate_size \
                + 2 * batch_size * seq_len * intermediate_size * self.config.hidden_size
        expert_gate_flops = 2 * batch_size * seq_len * self.config.hidden_size * self.config.num_experts  # gate_proj(D->E)
        if self.use_moe:
            per_layer_flops = attn_flops + expert_gate_flops + self.config.num_experts_per_tok * ffn_flops    # FLOPs per layer
        else:
            per_layer_flops = attn_flops + ffn_flops    # total FLOPs
        total_flops = 3 * self.config.num_layer * per_layer_flops  # fwd + bwd ≈ 3 × fwd FLOPs
        return total_flops