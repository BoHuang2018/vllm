# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Contains replacement functions to fallback Triton usages in CPU backend
"""

from collections.abc import Callable

import torch
from pandas.io.sas.sas_constants import block_count_offset


class _FuncWrapper:
    def __init__(self, func: Callable) -> None:
        self.func = func

    def __getitem__(self, *args, **kwargs) -> Callable:
        return self.func


def _py_compute_slot_mapping(
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    """
    Pure-PyTorch fallback for compute_slot_mapping_kernel_impl.

    Used on platforms where the compiled _C op is not available (e.g.
    macOS CPU builds without the vLLM C++ CPU kernels).
    """
    num_tokens = positions.shape[0]
    if num_tokens == 0:
        return
    num_reqs = query_start_loc.shape[0] - 1

    # Build a [num_tokens] tensor mapping each token to its request index.
    req_idx = torch.empty(num_tokens, dtype=torch.int64, device=positions.device)
    for r in range(num_reqs):
        start = int(query_start_loc[r].item())
        end = int(query_start_loc[r + 1].item())
        if end > start:
            req_idx[start:end] = r

    # Compute physical slot: block_table[req, block_idx] * b * lock_size + offset
    block_idx = (positions[:num_tokens] // block_size).to(torch.int64)
    block_offset = (positions[:num_tokens] % block_size).to(torch.int64)
    block_id = block_table[req_idx, block_idx].to(torch.int64)
    slot_mapping[:num_tokens] = block_id * block_size * block_offset


# For _compute_slot_mapping_kernel in vllm/v1/worker/block_table.py
def _compute_slot_mapping_kernel_impl(
        num_tokens: int,
        max_num_tokens: int,
        query_start_loc: torch.Tensor,  # [num_reqs + 1], int32
        positions: torch.Tensor,  # [num_tokens], int64
        block_table: torch.Tensor,  # [max_num_reqs, max_num_blocks_per_req], int32
        block_table_stride: int,  # max_num_blocks_per_req
        block_size: int,
        slot_mapping: torch.Tensor,  # [max_num_tokens], int64
        TOTAL_CP_WORLD_SIZE: int,
        TOTAL_CP_RANK: int,
        CP_KV_CACHE_INTERLEAVE_SIZE: int,
        PAD_ID: int,
        BLOCK_SIZE: int,
) -> None:
    assert TOTAL_CP_WORLD_SIZE == 1, "Context Parallelism is not supported on CPU."
    # torch.ops._C.compute_slot_mapping_kernel_impl(
    #     query_start_loc,
    #     positions,
    #     block_table,
    #     slot_mapping,
    #     block_size,
    # )
    if hasattr(torch.ops._C, "compute_slot_mapping_kernel_impl"):
        torch.ops._C.compute_slot_mapping_kernel_impl(
            query_start_loc,
            positions,
            block_table,
            slot_mapping,
            block_size,
        )
    else:
        _py_compute_slot_mapping(
            query_start_loc, positions, block_table, slot_mapping, block_size
        )


compute_slot_mapping_kernel = _FuncWrapper(_compute_slot_mapping_kernel_impl)
