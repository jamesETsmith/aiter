# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest

from aiter.ops.flydsl.mla_fwd_decode import get_flydsl_mla_kernel_name


@pytest.mark.parametrize(
    "batch_size,max_seqlen_q,max_seqlen_kv,num_heads,page_size,expected",
    [
        (16, 1, 65536, 128, 1, "bn64_v3"),
        (32, 4, 32768, 32, 1, "bn64_v3"),
        (64, 8, 16384, 16, 1, "bn64_v3"),
        (64, 16, 65536, 16, 1, "bn64_v3"),
        (8, 8, 65536, 16, 1, "bn32"),
        (16, 8, 8192, 16, 1, "bn32"),
        (16, 17, 65536, 16, 1, "bn32"),
        (16, 8, 65536, 16, 64, "bn32"),
    ],
)
def test_flydsl_mla_kernel_dispatch(
    batch_size,
    max_seqlen_q,
    max_seqlen_kv,
    num_heads,
    page_size,
    expected,
):
    assert (
        get_flydsl_mla_kernel_name(
            batch_size,
            max_seqlen_q,
            max_seqlen_kv,
            num_heads,
            page_size,
        )
        == expected
    )
