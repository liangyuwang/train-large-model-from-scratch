import torch

GPU_PEAK_FLOPS = {
    "T4":   {"fp32": 8.1e12, "fp16": 65e12, "bf16": 0},
    "V100": {"fp32": 15.7e12, "fp16": 125e12, "bf16": 0},
    "A100": {"fp32": 19.5e12, "fp16": 312e12, "bf16": 312e12},
    "A40":  {"fp32": 37.4e12, "fp16": 149e12, "bf16": 149e12},
    "A30":  {"fp32": 10.3e12, "fp16": 165e12, "bf16": 165e12},
    "RTX 6000 Ada": {"fp32": 91e12, "fp16": 181e12, "bf16": 181e12},
    "L4":   {"fp32": 30e12, "fp16": 120e12, "bf16": 120e12},
    "L40":  {"fp32": 91e12, "fp16": 181e12, "bf16": 181e12},
    "L40S": {"fp32": 91e12, "fp16": 181e12, "bf16": 181e12},
    "3090": {"fp32": 35.6e12, "fp16": 142e12, "bf16": 142e12},
    "4090": {"fp32": 82.6e12, "fp16": 330e12, "bf16": 330e12},
    "H100": {"fp32": 60e12, "fp16": 989e12, "bf16": 989e12, "fp8": 1979e12},
    "H800": {"fp32": 34e12, "fp16": 734e12, "bf16": 734e12, "fp8": 1468e12},
    "H200": {"fp32": 67e12, "fp16": 989e12, "bf16": 989e12, "fp8": 1979e12},
    "H20":  {"fp32": 21e12, "fp16": 494e12, "bf16": 494e12, "fp8": 988e12},
}


def get_gpu_peak_flops(dtype="bf16", per_device=True):
    """Detect GPU type and return theoretical peak FLOPs/s (for all GPUs)."""
    gpu_name = torch.cuda.get_device_name(0)
    dtype = dtype.lower()
    peak = None
    for k, v in GPU_PEAK_FLOPS.items():
        if k in gpu_name:
            peak = v.get(dtype, None)
            break
    if peak is None or peak == 0:
        print(f"Warning: unknown or unsupported FLOPs for GPU {gpu_name} with dtype {dtype}")
        peak = 0
    num_gpus = torch.cuda.device_count()
    return peak if per_device else peak * num_gpus

def compute_mfu_from_profiler(prof, dtype="bf16", warmup_steps=0):
    """
    Compute MFU using torch.profiler result.
    Args:
        prof: torch.profiler.profile result
        dtype: computation precision ("fp16", "bf16", "fp32")
    Returns:
        mfu (float), actual_flops_per_sec (float), peak_flops (float)
    """
    events = prof.key_averages()
    if warmup_steps > 0:
        events = [evt for evt in events if evt.count > warmup_steps]

    flops_total = sum(getattr(evt, "flops", 0) or 0 for evt in events)
    time_total = sum(getattr(evt, "device_time_total", 0) or 0 for evt in events) / 1e6  # seconds

    actual_flops_per_sec = flops_total / time_total if time_total > 0 else 0.0
    peak_flops = get_gpu_peak_flops(dtype=dtype, per_device=True)
    mfu = actual_flops_per_sec / peak_flops if peak_flops != 0 else 0
    return mfu, actual_flops_per_sec, peak_flops

def compute_mfu_from_time(batch_size, seq_len, hidden_dim, intermediate_dim, 
                          topk, num_experts, num_layers, 
                          time, ga=1, dtype="bf16"):
    """
    Approximate Transformer Block FLOPs and MFU.
    """
    # Q, K, V projection (3D->D) + Out projection (D->D)
    qkv_out_flops = 8 * batch_size * seq_len * hidden_dim * hidden_dim  # 6BLD^2 + 2BLD^2
    # QK^T + softmaxV
    attn_matmul_flops = 4 * batch_size * seq_len * seq_len * hidden_dim
    attn_flops = qkv_out_flops + attn_matmul_flops
    # FFN: SwiGLU = gate_proj(D->D_int) + up_proj(D->D_int) + elementwise + down_proj(D_int->D)
    ffn_flops = 2 * batch_size * seq_len * hidden_dim * intermediate_dim \
            + 2 * batch_size * seq_len * hidden_dim * intermediate_dim \
            + batch_size * seq_len * intermediate_dim \
            + 2 * batch_size * seq_len * intermediate_dim * hidden_dim
    expert_gate_flops = 2 * batch_size * seq_len * hidden_dim * num_experts  # gate_proj(D->E)
    per_layer_flops = attn_flops + expert_gate_flops + topk * ffn_flops    # FLOPs per layer
    total_flops = 3 * num_layers * per_layer_flops  # fwd + bwd ≈ 3 × fwd FLOPs
    total_flops *= ga   # gradient accumulation
    actual_flops_per_sec = total_flops / time if time > 0 else 0.0
    peak_flops = get_gpu_peak_flops(dtype=dtype, per_device=True)
    mfu = actual_flops_per_sec / peak_flops if peak_flops != 0 else 0
    return mfu, actual_flops_per_sec, peak_flops