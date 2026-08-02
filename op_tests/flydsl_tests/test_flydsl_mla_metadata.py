# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.flydsl.mla_metadata import get_flydsl_mla_metadata_config


@pytest.mark.parametrize(
    "batch_size,max_seqlen_q,max_seqlen_kv,num_heads,q_dtype,kv_dtype,expected",
    [
        (1, 8, 65536, 16, torch.float8_e4m3fn, torch.float8_e4m3fn, (64, 32)),
        (1, 8, 131072, 16, torch.float8_e4m3fnuz, torch.float8_e4m3fnuz, (64, 32)),
        (2, 8, 65536, 16, torch.float8_e4m3fn, torch.float8_e4m3fn, (32, 16)),
        (1, 4, 65536, 16, torch.float8_e4m3fn, torch.float8_e4m3fn, (32, 16)),
        (1, 8, 65535, 16, torch.float8_e4m3fn, torch.float8_e4m3fn, (32, 16)),
        (1, 8, 65536, 32, torch.float8_e4m3fn, torch.float8_e4m3fn, (32, 16)),
        (1, 8, 65536, 16, torch.bfloat16, torch.float8_e4m3fn, (32, 16)),
    ],
)
def test_flydsl_mla_metadata_config(
    batch_size,
    max_seqlen_q,
    max_seqlen_kv,
    num_heads,
    q_dtype,
    kv_dtype,
    expected,
):
    config = get_flydsl_mla_metadata_config(
        batch_size,
        max_seqlen_q,
        max_seqlen_kv,
        num_heads,
        q_dtype,
        kv_dtype,
    )
    assert (config["num_kv_splits"], config["kv_granularity"]) == expected
