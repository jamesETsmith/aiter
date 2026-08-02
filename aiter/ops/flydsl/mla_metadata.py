# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch


def _is_fp8(dtype: torch.dtype) -> bool:
    return dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


def get_flydsl_mla_metadata_config(
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    num_heads: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
) -> dict[str, int]:
    """Select persistent MLA metadata parameters for the FlyDSL stage-one kernel."""
    if (
        batch_size == 1
        and max_seqlen_q == 8
        and max_seqlen_kv >= 65536
        and num_heads == 16
        and _is_fp8(q_dtype)
        and _is_fp8(kv_dtype)
    ):
        return {"num_kv_splits": 64, "kv_granularity": 32}
    return {"num_kv_splits": 32, "kv_granularity": 16}
