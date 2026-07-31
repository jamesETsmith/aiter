# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MLA decode launcher.

Routes to the appropriate kernel implementation based on num_qo_heads
and dtype, then prepares tensors and dispatches via ``@flyc.jit``.
"""

import torch

from aiter.jit.utils.chip_info import get_cu_num, get_gfx, get_lds_size_per_cu


def _is_fp8(dtype: torch.dtype) -> bool:
    return dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


def flydsl_mla_decode_stage1_fwd(
    query: torch.Tensor,  # [num_seqs, num_heads, head_size]
    kv_buffer: torch.Tensor,  # [num_page, page_size, num_kv_heads, head_size]
    qo_indptr: torch.Tensor,  # [batch_size+1]
    kv_indptr: torch.Tensor,  # [batch_size+1]
    kv_page_indices: torch.Tensor,  # [num_page_used]
    kv_last_page_lens: torch.Tensor,  # [batch_size]
    work_indptr: torch.Tensor,  # metadata
    work_info_set: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    split_output: torch.Tensor,  # [num_partial_slots, 1, num_heads, v_head_dim]
    split_lse: torch.Tensor,  # [num_partial_slots, 1, num_heads, 1]
    final_output: torch.Tensor,  # [num_seqs, num_heads, v_head_dim]
    q_scale: torch.Tensor,
    kv_scale: torch.Tensor,
) -> None:
    """Launch the standalone gfx950 FlyDSL MLA decode stage-one kernel."""
    if get_gfx() != "gfx950":
        raise ValueError(f"flydsl_mla_decode_stage1_fwd requires gfx950, got {get_gfx()}")
    tensors = {
        "query": query,
        "kv_buffer": kv_buffer,
        "kv_page_indices": kv_page_indices,
        "kv_last_page_lens": kv_last_page_lens,
        "work_indptr": work_indptr,
        "work_info_set": work_info_set,
        "split_output": split_output,
        "split_lse": split_lse,
        "final_output": final_output,
        "q_scale": q_scale,
        "kv_scale": kv_scale,
    }
    for name, tensor in tensors.items():
        if not tensor.is_cuda or tensor.device != query.device:
            raise ValueError(f"{name} must be on {query.device}, got {tensor.device}")
    if query.dim() != 3 or query.size(2) != 576 or not query.is_contiguous():
        raise ValueError(f"query must be contiguous [total_q, heads, 576], got {query.shape}")
    if kv_buffer.dim() != 4 or kv_buffer.size(2) != 1 or kv_buffer.size(3) != 576:
        raise ValueError(f"kv_buffer must be [pages, page_size, 1, 576], got {kv_buffer.shape}")
    if kv_buffer.size(1) not in (1, 64):
        raise ValueError(f"page_size must be 1 or 64, got {kv_buffer.size(1)}")
    if q_scale.numel() != 1 or kv_scale.numel() != 1:
        raise ValueError("q_scale and kv_scale must be scalar tensors")
    if q_scale.dtype != torch.float32 or kv_scale.dtype != torch.float32:
        raise ValueError("q_scale and kv_scale must have dtype torch.float32")
    num_heads = query.size(1)
    q_dtype = query.dtype
    kv_dtype = kv_buffer.dtype

    # ----- dispatch to the right kernel based on (num_heads, dtype) -----
    if num_heads in (16, 32, 64, 128) and _is_fp8(q_dtype) and _is_fp8(kv_dtype):
        from .kernels.mla_fwd_decode_m16x8_fp8_fp8 import (
            OCCUPANCY,
            QK_HEAD_DIM,
            V_HEAD_DIM,
            launch_mla_fwd_decode_m16x8_fp8_fp8,
        )

        num_seqs = query.size(0)
        num_pages = kv_buffer.size(0)
        num_partial = split_output.size(0)
        if max_seqlen_q < 1 or num_seqs % max_seqlen_q != 0:
            raise ValueError(
                f"total_q={num_seqs} must be divisible by max_seqlen_q={max_seqlen_q}"
            )

        query_flat = query.reshape(num_seqs * num_heads, QK_HEAD_DIM)
        page_size = kv_buffer.size(1)
        kv_flat = kv_buffer.reshape(num_pages * page_size, QK_HEAD_DIM)
        final_flat = final_output.reshape(num_seqs * num_heads, V_HEAD_DIM)
        split_o_flat = split_output.reshape(num_partial * num_heads, V_HEAD_DIM)
        split_lse_flat = split_lse.reshape(num_partial * num_heads)

        work_indptr_flat = work_indptr.contiguous()
        work_info_flat = work_info_set.contiguous().view(-1)
        kv_idx_flat = kv_page_indices.contiguous()

        num_cus = get_cu_num()
        lds_size = get_lds_size_per_cu() // OCCUPANCY

        launch_mla_fwd_decode_m16x8_fp8_fp8(
            query_flat,
            kv_flat,
            kv_idx_flat,
            kv_last_page_lens.contiguous(),
            work_indptr_flat,
            work_info_flat,
            final_flat,
            split_o_flat,
            split_lse_flat,
            q_scale.contiguous(),
            kv_scale.contiguous(),
            softmax_scale,
            num_heads,
            page_size,
            num_cus,
            lds_size,
            stream=torch.cuda.current_stream(),
        )
    else:
        raise NotImplementedError(
            f"flydsl_mla_decode_stage1_fwd: unsupported num_heads={num_heads}, "
            f"q_dtype={q_dtype}, kv_dtype={kv_dtype}"
        )
