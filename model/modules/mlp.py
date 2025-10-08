import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import GPTConfig

"""
Features:
    0.1. Async MoE forward
    1. DeepGemm for FP8 Grouped GEMM
"""

class MLP(nn.Module):
    """Dense MLP or Single expert in MoE"""
    def __init__(self, config: GPTConfig, use_moe: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size if use_moe else config.intermediate_size
        self.use_moe = use_moe

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MoE(nn.Module):
    def __init__(self, config: GPTConfig, top_k: int = None, fused: bool = False):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = top_k if top_k is not None else config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.fused = fused

        self.moe_gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([MLP(config, use_moe=True) for _ in range(self.num_experts)])

    def forward(self, x: torch.Tensor):
        if self.fused:
            return self.forward_fused(x)
        
        """ copied from https://github.com/huggingface/transformers/blob/v4.56.1/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py """
        B, N, d = x.shape
        x = x.view(-1, d)
        # router_logits: (batch * N, n_experts)
        router_logits = self.moe_gate(x)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(x.dtype)
        final_x = torch.zeros((B * N, d), dtype=x.dtype, device=x.device)

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = x[None, top_x].reshape(-1, d)
            current_x = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_x.index_add_(0, top_x, current_x.to(x.dtype))
        final_x = final_x.reshape(B, N, d)
        return final_x, router_logits

    def forward_fused(self, x: torch.Tensor):
        """
        Efficient forward pass using grouped_gemm:
          y = down_proj( SiLU(gate_proj(x)) * up_proj(x) )

        Steps:
        1. Router selects top-k experts per token.
        2. Expand tokens for top-k and pack them by expert into contiguous blocks (aligned to group_size_m).
        3. Compute gate_proj and up_proj for all packed tokens using grouped_gemm.
        4. Apply activation and elementwise product.
        5. Compute down_proj using grouped_gemm.
        6. Multiply each contribution by its routing weight and scatter-add back to the original token positions.
        """
        from ..ops.grouped_gemm import grouped_gemm
        B, N, d = x.shape
        assert d == self.hidden_size
        device = x.device
        dtype = x.dtype

        x_flat = x.reshape(-1, d).contiguous()          # (M, d)
        M = x_flat.shape[0]

        # (1) Router: top-k expert selection
        router_logits = self.moe_gate(x_flat)           # (M, E)
        routing_weights_full = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights_k, selected_experts = torch.topk(routing_weights_full, self.top_k, dim=-1)  # (M, k)
        routing_weights_k = routing_weights_k.to(dtype)

        # Expand tokens for top-k
        k = self.top_k
        inputs_expanded = x_flat.repeat_interleave(k, dim=0)                      
        selected_experts_expanded = selected_experts.reshape(-1).to(torch.int32)  
        routing_expanded = routing_weights_k.reshape(-1)                          
        orig_token_ids = torch.arange(M, device=device, dtype=torch.long).repeat_interleave(k)  

        # (2) Pack tokens per expert (with group_size_m padding)
        group_size_m = 128  
        inp_packed, rw_packed, orig_packed, exp_idx_packed = _pack_tokens_by_expert(
            inputs_expanded, selected_experts_expanded, routing_expanded, orig_token_ids,
            num_experts=self.num_experts, group_size_m=group_size_m
        )
        M_p = inp_packed.shape[0]

        # (3) Collect expert weights
        inter = self.experts[0].intermediate_size
        W_gate = torch.empty((self.num_experts, inter, d), device=device, dtype=dtype)
        W_up   = torch.empty((self.num_experts, inter, d), device=device, dtype=dtype)
        W_down = torch.empty((self.num_experts, d, inter), device=device, dtype=dtype)
        for e, expert in enumerate(self.experts):
            W_gate[e].copy_(expert.gate_proj.weight.detach())
            W_up[e].copy_(expert.up_proj.weight.detach())
            W_down[e].copy_(expert.down_proj.weight.detach())

        W_gate = W_gate.contiguous()
        W_up   = W_up.contiguous()
        W_down = W_down.contiguous()

        # (4) gate_proj and up_proj using grouped_gemm
        g = grouped_gemm(inp_packed, W_gate, exp_idx_packed, group_size_m=group_size_m)  
        u = grouped_gemm(inp_packed, W_up,   exp_idx_packed, group_size_m=group_size_m)  
        z = self.experts[0].act_fn(g) * u                                                                

        # (5) down_proj using grouped_gemm
        y_packed = grouped_gemm(z, W_down, exp_idx_packed, group_size_m=group_size_m)    
        y_packed = y_packed * rw_packed.unsqueeze(1)                                     

        # (6) Scatter-add back to original positions (skip padded tokens where orig_id == -1)
        out_flat = torch.zeros((M, d), device=device, dtype=dtype)
        valid_mask = (orig_packed != -1)
        if valid_mask.any():
            out_flat.index_add_(0, orig_packed[valid_mask], y_packed[valid_mask])

        out = out_flat.view(B, N, d)
        return out, router_logits


def _pack_tokens_by_expert(
    inputs_expanded: torch.Tensor,          # (M*k, d)
    selected_experts_expanded: torch.Tensor,# (M*k,)
    routing_weights_expanded: torch.Tensor, # (M*k,)
    orig_token_ids: torch.Tensor,           # (M*k,)
    num_experts: int,
    group_size_m: int,
):
    """
    Pack tokens by expert. Each expert gets its own contiguous block, 
    padded up to a multiple of group_size_m so that each block belongs to a single expert.
    Returns packed inputs, routing weights, original indices, and expert_indices.
    """
    d = inputs_expanded.shape[1]
    device = inputs_expanded.device
    dtype = inputs_expanded.dtype

    packed_inputs = []
    packed_routing = []
    packed_orig_ids = []
    packed_expert_indices = []

    for e in range(num_experts):
        mask = (selected_experts_expanded == e)
        cnt = int(mask.sum().item())
        if cnt == 0:
            continue
        x_e = inputs_expanded[mask]              # (L_e, d)
        w_e = routing_weights_expanded[mask]     # (L_e,)
        id_e = orig_token_ids[mask]              # (L_e,)

        # Pad this expert's tokens up to a multiple of group_size_m
        pad_e = (-cnt) % group_size_m
        if pad_e > 0:
            x_pad = torch.zeros((pad_e, d), device=device, dtype=dtype)
            w_pad = torch.zeros((pad_e,), device=device, dtype=dtype)
            id_pad = torch.full((pad_e,), -1, device=device, dtype=torch.long)
            x_e = torch.cat([x_e, x_pad], dim=0)
            w_e = torch.cat([w_e, w_pad], dim=0)
            id_e = torch.cat([id_e, id_pad], dim=0)

        packed_inputs.append(x_e)
        packed_routing.append(w_e)
        packed_orig_ids.append(id_e)
        packed_expert_indices.append(torch.full((x_e.shape[0],), e, device=device, dtype=torch.int32))

    if len(packed_inputs) == 0:
        # No tokens routed to any expert (should not happen in practice)
        return (
            torch.empty(0, d, device=device, dtype=dtype),
            torch.empty(0, device=device, dtype=dtype),
            torch.empty(0, device=device, dtype=torch.long),
            torch.empty(0, device=device, dtype=torch.int32),
        )

    inputs_packed = torch.cat(packed_inputs, dim=0).contiguous()              
    routing_packed = torch.cat(packed_routing, dim=0).contiguous()            
    orig_ids_packed = torch.cat(packed_orig_ids, dim=0).contiguous()          
    expert_indices_packed = torch.cat(packed_expert_indices, dim=0).contiguous()  

    return inputs_packed, routing_packed, orig_ids_packed, expert_indices_packed