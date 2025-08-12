import functools
import json
import os
from typing import Any, Callable, Optional


import torch
import triton
import triton.language as tl


from vllm import _custom_ops as ops


def vllm_topk_softmax(topk_weights: torch.Tensor, topk_indices: torch.Tensor,
                      token_expert_indices: torch.Tensor,
                      gating_output: torch.Tensor,
                      renormalize: bool,
                      ) -> tuple[torch.Tensor, ...]:
    ops.topk_softmax(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
    )
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_indices


def ori_fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert hidden_states.shape[0] == gating_output.shape[0], (
        "Number of tokens mismatch")

    M, _ = hidden_states.shape

    topk_weights = torch.empty(M,
                               topk,
                               dtype=torch.float32,
                               device=hidden_states.device)
    topk_ids = torch.empty(
        M,
        topk,
        dtype=torch.int32 if indices_type is None else indices_type,
        device=hidden_states.device)
    token_expert_indices = torch.empty(M,
                                       topk,
                                       dtype=torch.int32,
                                       device=hidden_states.device)

    gating_output_float = gating_output.float()  # TODO(woosuk): Optimize this.

    topk_func = vllm_topk_softmax
    topk_weights, topk_ids = topk_func(topk_weights, topk_ids,
                                       token_expert_indices,
                                       gating_output_float, renormalize)

    return topk_weights, topk_ids



def my_fused_topk(
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        indices_type: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert hidden_states.shape[0] == gating_output.shape[0], (
            "Number of tokens mismatch")

        M, _ = hidden_states.shape

        topk+=1

        topk_weights = torch.empty(M,
                                topk,
                                dtype=torch.float32,
                                device=hidden_states.device)
        topk_ids = torch.empty(
            M,
            topk,
            dtype=torch.int32 if indices_type is None else indices_type,
            device=hidden_states.device)
        token_expert_indices = torch.empty(M,
                                        topk,
                                        dtype=torch.int32,
                                        device=hidden_states.device)

        gating_output_float = gating_output.float()  # TODO(woosuk): Optimize this.

        topk_func = vllm_topk_softmax
        topk_weights, topk_ids = topk_func(topk_weights, topk_ids,
                                        token_expert_indices,
                                        gating_output_float, renormalize)
        
        topk_weights=torch.cat([topk_weights[...,:-2],topk_weights[...,-1:]],dim=-1)
        topk_ids=torch.cat([topk_ids[...,:-2],topk_ids[...,-1:]],dim=-1)

        return topk_weights, topk_ids
