"""Convert PyTorch FX graphs to/from the lightweight Graph representation."""

from __future__ import annotations

import operator
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph import Graph

# ── Canonical op-type mappings ────────────────────────────────────

_FUNC_CANONICAL: dict[Any, str] = {
    operator.add: "add",
    torch.add: "add",
    operator.mul: "mul",
    torch.mul: "mul",
    operator.matmul: "matmul",
    torch.matmul: "matmul",
    torch.bmm: "bmm",
    F.relu: "relu",
    torch.relu: "relu",
    F.gelu: "gelu",
    F.silu: "silu",
    F.sigmoid: "sigmoid",
    torch.sigmoid: "sigmoid",
    F.tanh: "tanh",
    torch.tanh: "tanh",
    F.softmax: "softmax",
    F.linear: "linear",
    F.conv2d: "conv2d",
    F.batch_norm: "batch_norm",
    F.layer_norm: "layer_norm",
    F.dropout: "dropout",
    operator.getitem: "getitem",
}

_METHOD_CANONICAL: dict[str, str] = {
    "relu": "relu",
    "sigmoid": "sigmoid",
    "tanh": "tanh",
    "add": "add",
    "add_": "add",
    "mul": "mul",
    "mul_": "mul",
    "matmul": "matmul",
    "view": "view",
    "reshape": "reshape",
    "permute": "permute",
    "transpose": "transpose",
    "contiguous": "contiguous",
    "flatten": "flatten",
    "unsqueeze": "unsqueeze",
    "squeeze": "squeeze",
    "mean": "mean",
    "sum": "sum",
}

_MODULE_CANONICAL: dict[type, str] = {
    nn.Linear: "linear",
    nn.Conv2d: "conv2d",
    nn.BatchNorm2d: "batch_norm",
    nn.LayerNorm: "layer_norm",
    nn.ReLU: "relu",
    nn.GELU: "gelu",
    nn.SiLU: "silu",
    nn.Sigmoid: "sigmoid",
    nn.Tanh: "tanh",
    nn.Dropout: "dropout",
    nn.Softmax: "softmax",
}

# Node ops that are not real compute — skip during conversion.
_SKIP_OPS = {"placeholder", "get_attr", "output"}


# ── Public API ────────────────────────────────────────────────────


def fx_to_graph(traced: torch.fx.GraphModule) -> Graph:
    """Convert a traced ``GraphModule`` into a lightweight :class:`Graph`.

    Placeholder and get_attr nodes are omitted; only "real" compute nodes
    are included.  Edges between compute nodes are preserved.
    """
    g = Graph()
    fx_graph = traced.graph

    # First pass: add compute nodes.
    for node in fx_graph.nodes:
        if node.op in _SKIP_OPS:
            continue
        op_type = _resolve_op_type(node, traced)
        g.add_node(node.name, op_type)

    # Second pass: add edges (only between nodes present in g).
    for node in fx_graph.nodes:
        if node.name not in g.nodes:
            continue
        for arg in _iter_node_args(node):
            if isinstance(arg, torch.fx.Node) and arg.name in g.nodes:
                g.add_edge(arg.name, node.name)

    # Set outputs from the FX output node.
    for node in fx_graph.nodes:
        if node.op == "output":
            out_args = node.args[0]
            if isinstance(out_args, torch.fx.Node) and out_args.name in g.nodes:
                g.set_outputs([out_args.name])
            elif isinstance(out_args, (tuple, list)):
                outs = [a.name for a in out_args
                        if isinstance(a, torch.fx.Node) and a.name in g.nodes]
                if outs:
                    g.set_outputs(outs)
            break

    return g


def trace_to_graph(module: nn.Module, example_args: tuple | None = None) -> Graph:
    """Symbolically trace *module* and convert to :class:`Graph`."""
    traced = torch.fx.symbolic_trace(module)
    return fx_to_graph(traced)


# ── Internals ─────────────────────────────────────────────────────


def _resolve_op_type(node: torch.fx.Node, traced: torch.fx.GraphModule) -> str:
    if node.op == "call_function":
        return _FUNC_CANONICAL.get(node.target, _fallback_name(node.target))
    if node.op == "call_method":
        return _METHOD_CANONICAL.get(node.target, str(node.target))
    if node.op == "call_module":
        mod = traced.get_submodule(node.target)
        return _MODULE_CANONICAL.get(type(mod), type(mod).__name__.lower())
    return str(node.op)


def _fallback_name(target) -> str:
    if hasattr(target, "__name__"):
        return target.__name__
    return str(target)


def _iter_node_args(node: torch.fx.Node):
    """Yield all Node-valued args and kwargs."""
    for a in node.args:
        if isinstance(a, torch.fx.Node):
            yield a
        elif isinstance(a, (tuple, list)):
            for item in a:
                if isinstance(item, torch.fx.Node):
                    yield item
    for v in node.kwargs.values():
        if isinstance(v, torch.fx.Node):
            yield v
