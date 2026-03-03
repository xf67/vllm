# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    import torch.fx as fx


def _is_moe_module_call(node: torch.fx.Node, graph_module: "fx.GraphModule") -> bool:
    """
    Check if this node is a call to an MoE module (e.g. OlmoeMoE).
    Used to split piecewise CUDA graph between attention and MoE so that
    attention graph can be shared across different top-k.
    """
    if node.op != "call_module":
        return False
    try:
        mod = graph_module.get_submodule(node.target)
    except (AttributeError, KeyError):
        return False
    return getattr(mod, "_vllm_moe_module", False)


def should_split(
    node: torch.fx.Node,
    splitting_ops: list[str],
    graph_module: "fx.GraphModule | None" = None,
) -> tuple[bool, bool]:
    """
    Check if a node should be split for dynamo graph partition.
    Returns (should_split, is_moe_split).
    - When splitting on aten ops: (True, False) — segment is marked as splitting_graph.
    - When splitting on MoE module call: (True, True) — segment is compiled but is_moe_segment.
    """
    # MoE boundary: split so that attention and MoE are in different subgraphs
    if graph_module is not None and _is_moe_module_call(node, graph_module):
        return (True, True)

    if node.op != "call_function":
        return (False, False)

    target = node.target

    if isinstance(target, torch._ops.OpOverloadPacket):
        return (target._qualified_op_name in splitting_ops, False)

    if isinstance(target, torch._ops.OpOverload):
        packet_name = target.name()
        op_overload_name = f"{packet_name}.{target._overloadname}"
        if op_overload_name in splitting_ops or packet_name in splitting_ops:
            return (True, False)
    return (False, False)


@contextlib.contextmanager
def inductor_partition_rule_context(splitting_ops: list[str]):
    """Context manager to temporarily register Inductor partition rules.

    Registers custom partition rules for specified operators, forcing the
    Inductor scheduler to partition the graph at these operators. The rules
    are automatically restored to their previous state on exit.

    Args:
        splitting_ops: List of operator names to partition on.
    """
    if not splitting_ops:
        logger.debug("No partition ops provided; skipping rule registration.")
        yield
        return

    # Save current state before registering

    saved_splitting_ops: list[str] = list(
        torch._inductor.config.custom_should_partition_ops
    )
    torch._inductor.config.custom_should_partition_ops = splitting_ops

    logger.debug(
        "Registered inductor partition rules for %d operators", len(splitting_ops)
    )

    try:
        yield
    finally:
        # Clear and restore previous state
        torch._inductor.config.custom_should_partition_ops = saved_splitting_ops
        logger.debug("Restored previous partition rules state.")
