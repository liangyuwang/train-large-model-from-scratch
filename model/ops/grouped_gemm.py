# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# credit - Persistent rewrite of forward kernel with L2 caching optimization from @AdnanHoque, IBM Research.

"""
Copied from https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments/kernels/triton_contiguous_group_gemm
"""

import logging
from packaging import version

import torch
import triton
import triton.language as tl


# Configuration for autotuning

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

GROUP_SIZE_M = 128

# ===== Supporting utils, CUDA and TMA =====

class CudaUtils:
    @staticmethod
    def is_cuda() -> bool:
        """Check if Triton is running on CUDA backend."""
        return driver.active.get_current_target().backend == "cuda"

    @staticmethod
    def verify_tma() -> bool:
        """Check if TMA is supported on the current device."""
        return (
            CudaUtils.is_cuda()
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 9
        )

    @staticmethod
    def get_num_sms() -> int:
        """Get the number of streaming multiprocessors on the current device."""
        if not CudaUtils.is_cuda():
            raise RuntimeError("Triton is not running on CUDA backend")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        return torch.cuda.get_device_properties("cuda").multi_processor_count


class TmaDescriptorHelper:
    """Helper class for managing TMA descriptors in Triton kernels."""

    class KernelParamWrapper:
        """Wrapper to implement the TmaDescKernelParam interface."""

        def __init__(self, desc: torch.Tensor):
            self.desc = desc

        def tma_desc_cpu_ptr(self) -> int:
            """Return the CPU pointer to the TMA descriptor."""
            return self.desc.data_ptr()

    def __init__(self, tma_size: int = 128):

        if not CudaUtils.verify_tma():
            raise RuntimeError(
                "TMA not supported on this device (requires Hopper or newer)"
            )
        if "nv_tma_desc_type" not in dir(tl):
            raise RuntimeError(
                "TMA grid constant descriptors not supported in your Triton version"
            )

        self.tma_size = tma_size
        self.fill_1d_tma_descriptor_inner = driver.active.utils.fill_1d_tma_descriptor
        self.fill_2d_tma_descriptor_inner = driver.active.utils.fill_2d_tma_descriptor
        self.descriptors: Dict[str, torch.Tensor] = {}

    def init_tma_descriptor(self, name: str) -> None:
        """Initialize a TMA descriptor with the given name.

        Call this method outside of the lambda function for grid size.
        """
        self.descriptors[name] = torch.empty(
            self.tma_size, device="cpu", dtype=torch.int8
        )

    def fill_1d_tma_descriptor(
        self, name: str, ptr: int, dim: int, block_dim: int, element_size: int
    ) -> None:
        """Fill a 1D TMA descriptor.

        Call this method inside the lambda function for grid size.
        """
        if name not in self.descriptors:
            raise ValueError(f"TMA descriptor '{name}' not initialized")

        desc_x = self.descriptors[name]
        if desc_x.data_ptr() % 64 != 0:
            raise ValueError("TMA descriptor must be 64-byte aligned")
        self.fill_1d_tma_descriptor_inner(
            ptr, dim, block_dim, element_size, desc_x.data_ptr()
        )

    def fill_2d_tma_descriptor(
        self,
        name: str,
        ptr: int,
        dim1: int,
        dim0: int,
        block_dim1: int,
        block_dim0: int,
        element_size: int,
    ) -> None:
        """Fill a 2D TMA descriptor.

        Call this method inside the lambda function for grid size.
        """
        if name not in self.descriptors:
            raise ValueError(f"TMA descriptor '{name}' not initialized")

        desc_x = self.descriptors[name]
        if desc_x.data_ptr() % 64 != 0:
            raise ValueError("TMA descriptor must be 64-byte aligned")
        self.fill_2d_tma_descriptor_inner(
            ptr, dim1, dim0, block_dim1, block_dim0, element_size, desc_x.data_ptr()
        )

    def get_tma_descriptor_kernel_param(self, name: str) -> KernelParamWrapper:
        """Get the TMA descriptor kernel parameter for the given name."""
        if name not in self.descriptors or self.descriptors[name] is None:
            raise ValueError(f"TMA descriptor '{name}' not initialized")
        return self.KernelParamWrapper(self.descriptors[name])


# ================== End of supporting functions ==================

'''
def early_config_prune(configs, args, **kwargs):
    """Filter out configurations that would exceed shared memory capacity."""
    sms = kwargs.get("NUM_SMS", 108)  # Default value if not provided
    k = kwargs.get("K", 0)
    configs = [
        config for config in configs if config.kwargs.get("BLOCK_SIZE_K", 0) <= k
    ]
    return configs
'''

# Define standard configurations for Hopper GPUs
HOPPER_CONFIGS = [
    # Configurations for small matrices
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
        num_stages=2,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32},
        num_stages=3,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
        num_stages=3,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
        num_stages=4,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
        num_stages=4,
        num_warps=4,
    ),
    # Configurations for medium to large matrices
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
        num_stages=3,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
        num_stages=3,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
        num_stages=3,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
        num_stages=4,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
        num_stages=4,
        num_warps=8,
    ),
]


# Define standard configurations - simplified for robustness
STANDARD_CONFIGS = [
    # Configurations for small matrices
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
        num_stages=2,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
        num_stages=2,
        num_warps=4,
    ),
    # Medium sizes
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
        num_stages=2,
        num_warps=8,
    ),
    # Larger sizes with more warps, stages
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
        num_stages=3,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
        num_stages=4,
        num_warps=8,
    ),
]


def early_config_prune(configs, args, **kwargs):
    """Filter out configurations that would exceed shared memory capacity."""
    k = kwargs.get("K", 0)
    valid_configs = [
        config for config in configs if config.kwargs.get("BLOCK_SIZE_K", 0) <= k
    ]
    # If all configs were filtered out, return at least one config
    if not valid_configs and configs:
        # Find the config with the smallest BLOCK_SIZE_K
        return [min(configs, key=lambda c: c.kwargs.get("BLOCK_SIZE_K", float("inf")))]

    return valid_configs

# ============ Triton kernel for contiguous grouped GEMM ============

# L2 Caching optimization

@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, super_group_m):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * super_group_m
    group_size_m = min(num_pid_m - first_pid_m, super_group_m)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.autotune(
    configs=STANDARD_CONFIGS,
    key=["M_TOTAL", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def _kernel_cg_persistent_forward(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Pointer to indices array
    indices_ptr,
    # Matrix dimensions
    M_TOTAL: tl.constexpr,  # Total M dimension (sum of all groups)
    N: tl.constexpr,  # N dimension
    K: tl.constexpr,  # K dimension
    # Number of experts
    NUM_EXPERTS: tl.constexpr,
    # Tiling parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    # NUM_CONSUMER_GROUPS: tl.constexpr,
    # Group size (for aligned loads)
    GROUP_SIZE_M: tl.constexpr = 128,
    SUPER_GROUP_M: tl.constexpr = 32,  # 32 works best
):
    """
    Contiguous Grouped GEMM kernel forward.
    IMPORTANT: Assumes GROUP_SIZE_M is a multiple of BLOCK_SIZE_M or vice versa,
    and all inputs are pre-aligned to these block boundaries.
    """

    c_type = c_ptr.dtype.element_ty

    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M_TOTAL, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    tile_id_c = start_pid - NUM_SMS
    num_pid_in_group = SUPER_GROUP_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):

        tile_m_idx, tile_n_idx = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, SUPER_GROUP_M
        )

        # starting indices for this tile
        m_start = tile_m_idx * BLOCK_SIZE_M
        n_start = tile_n_idx * BLOCK_SIZE_N

        # Only process if in bounds
        if m_start < M_TOTAL:

            offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
            offs_n = n_start + tl.arange(0, BLOCK_SIZE_N)

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for ki in range(k_tiles):

                # Offsets for K dim
                offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

                # Create masks for bounds checking
                mask_m = offs_m < M_TOTAL
                mask_n = offs_n < N
                mask_k = offs_k < K

                # masks for A and B
                mask_a = mask_m[:, None] & mask_k[None, :]
                mask_b = mask_n[:, None] & mask_k[None, :]

                # Determine the expert group index and load expert ID
                group_idx = m_start // GROUP_SIZE_M
                expert_idx = tl.load(indices_ptr + group_idx * GROUP_SIZE_M)

                # Load inputs (A) with bounds checking
                a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
                a = tl.load(a_ptrs, mask=mask_a, other=0.0)

                # Load expert weights (B) for the expert assigned to this block
                b_ptrs = (
                    b_ptr + expert_idx * N * K + offs_n[:, None] * K + offs_k[None, :]
                )
                b = tl.load(b_ptrs, mask=mask_b, other=0.0)

                # Accumulate matrix multiplication for this K tile
                accumulator += tl.dot(
                    a, b.T
                )  # out_dtype=tl.float32) # * a_scale # * b_scale

            tile_id_c += NUM_SMS
            tile_m_idx, tile_n_idx = _compute_pid(
                tile_id_c, num_pid_in_group, num_pid_m, SUPER_GROUP_M
            )

            offs_m = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_n = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

            # Create masks for bounds checking
            mask_m = offs_m < M_TOTAL
            mask_n = offs_n < N
            mask_c = mask_m[:, None] & mask_n[None, :]

            c = accumulator.to(tl.float32)

            # Store output (C) with bounds checking
            c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
            tl.store(c_ptrs, c.to(c_type), mask=mask_c)


@triton.autotune(
    configs=STANDARD_CONFIGS,
    key=["M_TOTAL", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def _kernel_cg_forward_aligned(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Pointer to indices array
    indices_ptr,
    # Matrix dimensions
    M_TOTAL: tl.constexpr,  # Total M dimension (sum of all groups)
    N: tl.constexpr,  # N dimension
    K: tl.constexpr,  # K dimension
    # Number of experts
    NUM_EXPERTS: tl.constexpr,
    # Tiling parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # Group size (for aligned loads)
    GROUP_SIZE_M: tl.constexpr = 128,
):
    """
    Contiguous Grouped GEMM kernel forward.
    IMPORTANT: Assumes GROUP_SIZE_M is a multiple of BLOCK_SIZE_M
    and all inputs are pre-aligned/padded to these block boundaries.
    """

    pid = tl.program_id(0)

    c_type = c_ptr.dtype.element_ty

    # number of tiles per matrix dimension
    num_m_tiles = tl.cdiv(M_TOTAL, BLOCK_SIZE_M)
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)

    # 2D tile index from linear
    tile_m = pid // num_n_tiles
    tile_n = pid % num_n_tiles

    # starting indices for this tile
    m_start = tile_m * BLOCK_SIZE_M
    n_start = tile_n * BLOCK_SIZE_N

    # Only process if in bounds
    if m_start < M_TOTAL:

        # Create offset arrays for input, output coordinates
        offs_m = tl.arange(0, BLOCK_SIZE_M) + m_start
        offs_n = tl.arange(0, BLOCK_SIZE_N) + n_start

        # Create masks for bounds checking
        mask_m = offs_m < M_TOTAL
        mask_n = offs_n < N

        # Determine the expert group index and load expert ID
        group_idx = m_start // GROUP_SIZE_M
        expert_idx = tl.load(indices_ptr + group_idx * GROUP_SIZE_M)

        # Initialize accumulator
        acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

        # matrix multiplication in tiles along K dimension
        for k in range(0, K, BLOCK_SIZE_K):
            # offsets and mask for K dimension
            offs_k = tl.arange(0, BLOCK_SIZE_K) + k
            mask_k = offs_k < K

            # masks for A and B
            mask_a = mask_m[:, None] & mask_k[None, :]
            mask_b = mask_n[:, None] & mask_k[None, :]

            # Load inputs (A) with bounds checking
            a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
            a = tl.load(a_ptrs, mask=mask_a, other=0.0)

            # Load expert weights (B) for the expert assigned to this block
            b_ptrs = b_ptr + expert_idx * N * K + offs_n[:, None] * K + offs_k[None, :]
            b = tl.load(b_ptrs, mask=mask_b, other=0.0)

            # Accumulate matrix multiplication for this K tile
            acc += tl.dot(a, b.T)

        # Store results with bounds checking

        c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
        mask_c = mask_m[:, None] & mask_n[None, :]
        tl.store(c_ptrs, acc.to(c_type), mask=mask_c)


# =============== Forward Wrapper for CGGEMM =================


def cg_grouped_gemm_forward(
    inputs: torch.Tensor,  # [M_total, K]
    expert_weights: torch.Tensor,  # [num_experts, N, K]
    expert_indices: torch.Tensor,  # [M_total]
    group_size_m: int = 128,
) -> torch.Tensor:
    """
    contiguous grouped GEMM forward pass for MoE.
    All tokens mapped to the same expert must be in contiguous blocks of size group_size_m.

    Args:
        inputs: Input tensor of shape [M_total, K]
        expert_weights: Expert weight tensor of shape [num_experts, N, K]
        expert_indices: Indices tensor of shape [M_total] mapping each token to its expert
        group_size_m: Size of contiguous token blocks for each expert (default: 128)
        x_scale: Input tensor scales of shape [M_total, 1]
        w_scale: Expert weight tensor scales of shape [num_experts, N]
    Returns:
        Output tensor of shape [M_total, N]
    """
    # Validate inputs
    assert inputs.is_contiguous(), "Input tensor must be contiguous"
    assert expert_weights.is_contiguous(), "Expert weights tensor must be contiguous"
    assert expert_indices.is_contiguous(), "Expert indices tensor must be contiguous"

    # Check if inputs are properly aligned
    M_total, K = inputs.shape
    assert (
        M_total % group_size_m == 0
    ), f"M_total ({M_total}) must be a multiple of group_size_m ({group_size_m})"

    # Convert expert_indices to int32 if needed
    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    # Get dimensions
    num_experts, N, K_weights = expert_weights.shape

    # Validate dimensions
    assert K == K_weights, f"Input K ({K}) must match weight K ({K_weights})"
    assert (
        expert_indices.shape[0] == M_total
    ), "Expert indices length must match M_total"

    # Create output tensor
    output = torch.empty((M_total, N), device=inputs.device, dtype=inputs.dtype)

    # Calculate grid size for the kernel
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    grid = (NUM_SMS, 1, 1)
    # Launch kernel
    _kernel_cg_persistent_forward[grid](
        inputs,
        expert_weights,
        output,
        expert_indices,
        M_TOTAL=M_total,
        N=N,
        K=K,
        NUM_EXPERTS=num_experts,
        GROUP_SIZE_M=group_size_m,
        NUM_SMS=NUM_SMS,
    )

    return output


def cg_grouped_gemm_forward_dynamic(
    inputs: torch.Tensor,  # [M_total, K]
    expert_weights: torch.Tensor,  # [num_experts, N, K]
    expert_indices: torch.Tensor,  # [M_total]
    group_size_m: int = 128,
) -> torch.Tensor:
    """
    contiguous grouped GEMM forward pass for MoE.
    All tokens mapped to the same expert must be in contiguous blocks of size group_size_m.

    Args:
        inputs: Input tensor of shape [M_total, K]
        expert_weights: Expert weight tensor of shape [num_experts, N, K]
        expert_indices: Indices tensor of shape [M_total] mapping each token to its expert
        group_size_m: Size of contiguous token blocks for each expert (default: 128)

    Returns:
        Output tensor of shape [M_total, N]
    """
    # Validate inputs
    assert inputs.is_contiguous(), "Input tensor must be contiguous"
    assert expert_weights.is_contiguous(), "Expert weights tensor must be contiguous"
    assert expert_indices.is_contiguous(), "Expert indices tensor must be contiguous"

    # Check if inputs are properly aligned
    M_total, K = inputs.shape
    assert (
        M_total % group_size_m == 0
    ), f"M_total ({M_total}) must be a multiple of group_size_m ({group_size_m})"
    # assert (
    #    expert_indices.shape[0] == M_total // group_size_m
    # ), "Expert indices length must match number of groups"

    # Convert expert_indices to int32 if needed
    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    # Get dimensions
    num_experts, N, K_weights = expert_weights.shape

    # Validate dimensions
    assert K == K_weights, f"Input K ({K}) must match weight K ({K_weights})"
    assert (
        expert_indices.shape[0] == M_total
    ), "Expert indices length must match M_total"
    # assert (
    #    expert_indices.shape[0] == M_total // group_size_m
    # ), "Expert indices length must match number of groups"

    # Create output tensor
    output = torch.empty((M_total, N), device=inputs.device, dtype=inputs.dtype)

    # Calculate grid size for the kernel
    grid = lambda meta: (
        triton.cdiv(M_total, meta["BLOCK_SIZE_M"])
        * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
    )

    # Launch kernel
    _kernel_cg_forward_aligned[grid](
        inputs,
        expert_weights,
        output,
        expert_indices,
        M_TOTAL=M_total,
        N=N,
        K=K,
        NUM_EXPERTS=num_experts,
        GROUP_SIZE_M=group_size_m,
    )

    return output


# ============ Triton kernel for contiguous grouped GEMM backward inputs ============


@triton.autotune(
    configs=STANDARD_CONFIGS,
    key=["M_TOTAL", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
)
@triton.jit
def _kernel_cg_backward_dx(
    # Pointers to matrices
    grad_output_ptr,  # [M_TOTAL, N]
    b_ptr,  # expert weights [num_experts, N, K]
    grad_input_ptr,  # [M_TOTAL, K]
    # Pointer to indices array
    indices_ptr,  # [M_TOTAL]
    # Matrix dimensions
    M_TOTAL: tl.constexpr,  # Total M dimension (sum of all groups)
    N: tl.constexpr,  # N dimension
    K: tl.constexpr,  # K dimension
    # Number of experts
    NUM_EXPERTS: tl.constexpr,
    # Tiling parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # Group size (for aligned loads)
    GROUP_SIZE_M: tl.constexpr = 128,
):
    """
    Computes the gradient with respect to the inputs (backward pass).
    Performs: grad_input = grad_output @ expert_weights
    """
    pid = tl.program_id(0)

    # number of tiles per matrix dimension
    num_m_tiles = tl.cdiv(M_TOTAL, BLOCK_SIZE_M)
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    # 2D tile index from linear
    tile_m = pid // num_k_tiles
    tile_k = pid % num_k_tiles

    # starting indices for this tile
    m_start = tile_m * BLOCK_SIZE_M
    k_start = tile_k * BLOCK_SIZE_K

    # Only process if in bounds
    if m_start < M_TOTAL:

        # Create offset arrays for input, output coordinates
        offs_m = tl.arange(0, BLOCK_SIZE_M) + m_start
        offs_k = tl.arange(0, BLOCK_SIZE_K) + k_start

        # Create masks for bounds checking
        mask_m = offs_m < M_TOTAL
        mask_k = offs_k < K

        # Determine the expert group index and load expert ID
        group_idx = m_start // GROUP_SIZE_M
        expert_idx = tl.load(indices_ptr + group_idx * GROUP_SIZE_M)

        # Initialize accumulator for the gradient
        grad_input = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=tl.float32)

        # Compute gradient with respect to inputs in tiles along N dimension
        for n in range(0, N, BLOCK_SIZE_N):
            # offsets and mask for N dimension
            offs_n = tl.arange(0, BLOCK_SIZE_N) + n
            mask_n = offs_n < N

            # Masks for grad_output and weights
            mask_go = mask_m[:, None] & mask_n[None, :]
            mask_w = mask_n[:, None] & mask_k[None, :]

            # Load grad_output with bounds checking
            go_ptrs = grad_output_ptr + offs_m[:, None] * N + offs_n[None, :]
            go = tl.load(go_ptrs, mask=mask_go, other=0.0)

            # Load expert weights for the expert assigned to this block
            # For backward pass, we need W, not W^T, so dimensions are [N, K]
            w_ptrs = b_ptr + expert_idx * N * K + offs_n[:, None] * K + offs_k[None, :]
            w = tl.load(w_ptrs, mask=mask_w, other=0.0)

            # Compute partial gradient for this N tile: grad_input += grad_output @ weights
            # Note: We're doing matmul without explicit transpose in triton
            grad_input += tl.dot(go, w)

        # Store results with bounds checking
        grad_input_ptrs = grad_input_ptr + offs_m[:, None] * K + offs_k[None, :]
        mask_gi = mask_m[:, None] & mask_k[None, :]
        tl.store(grad_input_ptrs, grad_input, mask=mask_gi)


# ============ Triton kernel for contiguous grouped GEMM backward weights ============


# =============== Functions for backward pass =================
# ==== simpler approach =============
@triton.jit
def _kernel_cg_backward_dw(
    # Pointers to matrices
    grad_output_ptr,  # [M_TOTAL, N]
    inputs_ptr,  # [M_TOTAL, K]
    grad_weights_ptr,  # [num_experts, N, K]
    indices_ptr,  # [M_total]
    # Matrix dimensions
    M_TOTAL: tl.constexpr,  # Total M dimension
    N: tl.constexpr,  # N dimension
    K: tl.constexpr,  # K dimension
    # Number of experts
    NUM_EXPERTS: tl.constexpr,
    # Group parameters
    GROUP_SIZE_M: tl.constexpr,
    # Tiling parameters
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    """
    Significantly simplified kernel for weight gradient computation.
    This kernel processes one N-K tile across all groups that use the same expert.
    """
    pid = tl.program_id(0)

    # Determine expert and position within the matrix
    expert_id = pid // ((N * K) // (BLOCK_SIZE_N * BLOCK_SIZE_K))
    position_id = pid % ((N * K) // (BLOCK_SIZE_N * BLOCK_SIZE_K))

    # Only process if expert is valid
    if expert_id < NUM_EXPERTS:
        # Calculate positions in N and K dimensions
        n_tiles = K // BLOCK_SIZE_K
        tile_n = position_id // n_tiles
        tile_k = position_id % n_tiles

        n_start = tile_n * BLOCK_SIZE_N
        k_start = tile_k * BLOCK_SIZE_K

        # Only process if in bounds
        if n_start < N and k_start < K:
            # Create offset arrays
            offs_n = tl.arange(0, BLOCK_SIZE_N) + n_start
            offs_k = tl.arange(0, BLOCK_SIZE_K) + k_start

            # Create masks for bounds checking
            mask_n = offs_n < N
            mask_k = offs_k < K

            # Initialize accumulator for the gradient
            grad_weights = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_K], dtype=tl.float32)

            # Go through all groups to find those using this expert
            for group_idx in range(0, M_TOTAL // GROUP_SIZE_M):
                group_start = group_idx * GROUP_SIZE_M

                # Get expert ID for this group
                group_expert = tl.load(indices_ptr + group_start)

                # Only process if this group uses the current expert
                if group_expert == expert_id:
                    # Process the group in blocks
                    for m_offset in range(0, GROUP_SIZE_M, BLOCK_SIZE_M):
                        # Global offsets for group's data
                        m_start = group_start + m_offset
                        offs_m = tl.arange(0, BLOCK_SIZE_M) + m_start

                        # Create mask for M dimension
                        mask_m = offs_m < min(group_start + GROUP_SIZE_M, M_TOTAL)

                        # Load grad_output [M, N]
                        go_ptrs = (
                            grad_output_ptr + offs_m[:, None] * N + offs_n[None, :]
                        )
                        mask_go = mask_m[:, None] & mask_n[None, :]
                        go = tl.load(go_ptrs, mask=mask_go, other=0.0)

                        # Load inputs [M, K]
                        in_ptrs = inputs_ptr + offs_m[:, None] * K + offs_k[None, :]
                        mask_in = mask_m[:, None] & mask_k[None, :]
                        inp = tl.load(in_ptrs, mask=mask_in, other=0.0)

                        # Compute gradient contribution
                        go_t = tl.trans(go)  # Transpose: [N, M]
                        grad_weights += tl.dot(go_t, inp)

            # Store results to the appropriate part of the expert's weight gradients
            grad_w_ptrs = (
                grad_weights_ptr
                + expert_id * N * K
                + offs_n[:, None] * K
                + offs_k[None, :]
            )
            mask_gw = mask_n[:, None] & mask_k[None, :]
            tl.store(grad_w_ptrs, grad_weights, mask=mask_gw)


def cg_grouped_gemm_backward_weights(
    grad_output: torch.Tensor,  # [M_total, N]
    inputs: torch.Tensor,  # [M_total, K]
    expert_indices: torch.Tensor,  # [M_total]
    num_experts: int,
    group_size_m: int = 128,
) -> torch.Tensor:
    """
    Simple version of backward pass for weights using a single kernel launch.

    Args:
        grad_output: Gradient from output, shape [M_total, N]
        inputs: Input tensor, shape [M_total, K]
        expert_indices: Indices tensor mapping each token to its expert, shape [M_total]
        num_experts: Number of experts
        group_size_m: Size of contiguous token blocks for each expert (default: 128)

    Returns:
        grad_weights: Gradient with respect to expert weights, shape [num_experts, N, K]
    """
    # Validate inputs
    assert grad_output.is_contiguous(), "Grad output tensor must be contiguous"
    assert inputs.is_contiguous(), "Inputs tensor must be contiguous"
    assert expert_indices.is_contiguous(), "Expert indices tensor must be contiguous"

    # Get dimensions
    M_total, N = grad_output.shape
    _, K = inputs.shape

    # Ensure expert_indices has the right dtype
    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    # Create output tensor for gradients
    grad_weights = torch.zeros(
        (num_experts, N, K), device=grad_output.device, dtype=grad_output.dtype
    )

    # Define block sizes based on dimensions
    # These are chosen to balance parallelism and shared memory usage
    block_size_n = min(128, N)
    block_size_k = min(32, K)
    block_size_m = min(32, group_size_m)

    # Calculate grid size for the kernel
    # Each thread block handles one expert's N-K tile
    n_tiles = triton.cdiv(N, block_size_n)
    k_tiles = triton.cdiv(K, block_size_k)
    grid = (num_experts * n_tiles * k_tiles,)

    # Launch kernel
    _kernel_cg_backward_dw[grid](
        grad_output,
        inputs,
        grad_weights,
        expert_indices,
        M_TOTAL=M_total,
        N=N,
        K=K,
        NUM_EXPERTS=num_experts,
        GROUP_SIZE_M=group_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        BLOCK_SIZE_M=block_size_m,
    )

    return grad_weights


def cg_grouped_gemm_backward_inputs(
    grad_output: torch.Tensor,  # [M_total, N]
    expert_weights: torch.Tensor,  # [num_experts, N, K]
    expert_indices: torch.Tensor,  # [M_total]
    group_size_m: int = 128,
) -> torch.Tensor:
    """
    Backward pass for contiguous grouped GEMM with respect to inputs.

    Args:
        grad_output: Gradient from output, shape [M_total, N]
        expert_weights: Expert weight tensor, shape [num_experts, N, K]
        expert_indices: Indices tensor mapping each token to its expert, shape [M_total]
        group_size_m: Size of contiguous token blocks for each expert (default: 128)

    Returns:
        grad_inputs: Gradient with respect to inputs, shape [M_total, K]
    """
    # Validate inputs
    assert grad_output.is_contiguous(), "Grad output tensor must be contiguous"
    assert expert_weights.is_contiguous(), "Expert weights tensor must be contiguous"
    assert expert_indices.is_contiguous(), "Expert indices tensor must be contiguous"

    # Get dimensions
    M_total, N = grad_output.shape
    num_experts, _, K = expert_weights.shape

    # Check if dimensions match
    assert (
        M_total % group_size_m == 0
    ), f"M_total ({M_total}) must be a multiple of group_size_m ({group_size_m})"

    # Create output tensor for gradients
    grad_inputs = torch.zeros(
        (M_total, K), device=grad_output.device, dtype=grad_output.dtype
    )

    # Calculate grid size for the kernel
    grid = lambda meta: (
        triton.cdiv(M_total, meta["BLOCK_SIZE_M"])
        * triton.cdiv(K, meta["BLOCK_SIZE_K"]),
    )

    # Launch kernel
    _kernel_cg_backward_dx[grid](
        grad_output,
        expert_weights,
        grad_inputs,
        expert_indices,
        M_TOTAL=M_total,
        N=N,
        K=K,
        NUM_EXPERTS=num_experts,
        GROUP_SIZE_M=group_size_m,
    )

    return grad_inputs


# =============== Update the autograd function =================


class ContiguousGroupedGEMM(torch.autograd.Function):
    """
    Autograd function for contiguous grouped GEMM with complete backward pass.
    """

    @staticmethod
    def forward(ctx, inputs, expert_weights, expert_indices, group_size_m=128):
        """Forward pass for contiguous grouped GEMM."""
        # Save for backward
        ctx.save_for_backward(inputs, expert_weights, expert_indices)
        ctx.group_size_m = group_size_m

        return cg_grouped_gemm_forward(
            inputs=inputs,
            expert_weights=expert_weights,
            expert_indices=expert_indices,
            group_size_m=group_size_m,
        )

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for contiguous grouped GEMM."""
        inputs, expert_weights, expert_indices = ctx.saved_tensors
        group_size_m = ctx.group_size_m

        # Get number of experts
        num_experts = expert_weights.shape[0]

        # Make sure grad_output is contiguous
        grad_output = grad_output.contiguous()

        # Compute gradients
        grad_inputs = cg_grouped_gemm_backward_inputs(
            grad_output=grad_output,
            expert_weights=expert_weights,
            expert_indices=expert_indices,
            group_size_m=group_size_m,
        )

        grad_weights = cg_grouped_gemm_backward_weights(
            grad_output=grad_output,
            inputs=inputs,
            expert_indices=expert_indices,
            num_experts=num_experts,
            group_size_m=group_size_m,
        )

        # No gradient for expert_indices (it's just an index tensor)
        grad_indices = None

        # No gradient for group_size_m (it's just a parameter)
        grad_group_size_m = None

        return grad_inputs, grad_weights, grad_indices, grad_group_size_m


def cg_grouped_gemm(
    inputs: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    group_size_m: int = 128,
) -> torch.Tensor:
    """
    Interface for contiguous grouped GEMM with full backward pass support.

    Args:
        inputs: Input tensor of shape [M_total, K]
        expert_weights: Expert weight tensor of shape [num_experts, N, K]
        expert_indices: Indices tensor of shape [M_total] mapping each token to its expert
        group_size_m: Size of contiguous token blocks for each expert (default: 128)

    Returns:
        Output tensor of shape [M_total, N]
    """
    if expert_indices.dtype != torch.int32:
        expert_indices = expert_indices.to(torch.int32)

    return ContiguousGroupedGEMM.apply(
        inputs, expert_weights, expert_indices, group_size_m
    )


# =============== API function =================

def grouped_gemm(
    inputs: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    group_size_m: int = 128,
) -> torch.Tensor:
    """
    API function for contiguous grouped GEMM with full backward pass support.

    Args:
        inputs: Input tensor of shape [M_total, K]
        expert_weights: Expert weight tensor of shape [num_experts, N, K]
        expert_indices: Indices tensor of shape [M_total] mapping each token to its expert
        group_size_m: Size of contiguous token blocks for each expert (default: 128)

    Returns:
        Output tensor of shape [M_total, N]
    """
    # check contiguity
    assert inputs.is_contiguous(), "Input tensor must be contiguous"
    assert expert_weights.is_contiguous(), "Expert weights tensor must be contiguous"
    assert expert_indices.is_contiguous(), "Expert indices tensor must be contiguous"
    if version.parse(torch.__version__) >= version.parse("2.8.0"):
        # ===== Adapt for torch._grouped_mm =====
        M, K = inputs.shape
        num_experts, N, K_ = expert_weights.shape
        assert K == K_, "Input dim mismatch"

        # (1) Sort inputs by expert_indices
        sorted_idx = torch.argsort(expert_indices)
        inputs_sorted = inputs[sorted_idx]

        # (2) Compute offsets (prefix sum of the number of tokens for each expert)
        counts = torch.bincount(expert_indices, minlength=num_experts)
        offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)

        # (3) weight: [E, K, N]
        weights_t = expert_weights.transpose(-1, -2).contiguous()

        # (4) Call torch._grouped_mm
        outputs_sorted = torch._grouped_mm(inputs_sorted, weights_t, offsets)

        # (5) back
        inv_idx = torch.empty_like(sorted_idx)
        inv_idx[sorted_idx] = torch.arange(M, device=inputs.device)
        outputs = outputs_sorted[inv_idx]

        return outputs
    else:
        # ===== Fallback: use custom kernel =====
        return cg_grouped_gemm(
            inputs, expert_weights, expert_indices, group_size_m=group_size_m
        )