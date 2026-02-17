"""Integration tests: PyTorch FX → Graph → tiler."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from fx_adapter import trace_to_graph, fx_to_graph
from graph import Graph
from solver import tile


# ── FX adapter unit tests ─────────────────────────────────────────


class TestFxAdapter:
    def test_simple_sequential(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        g = trace_to_graph(model)
        types = set(g.nodes.values())
        assert "linear" in types
        assert "relu" in types
        assert len(g) == 3  # linear, relu, linear

    def test_op_canonicalization_add(self):
        """operator.add used via + should canonicalize to 'add'."""

        class AddModel(nn.Module):
            def forward(self, x, y):
                return x + y

        g = trace_to_graph(AddModel())
        assert "add" in g.nodes.values()

    def test_op_canonicalization_relu_functional(self):
        class M(nn.Module):
            def forward(self, x):
                return F.relu(x)

        g = trace_to_graph(M())
        assert "relu" in g.nodes.values()

    def test_op_canonicalization_relu_module(self):
        model = nn.Sequential(nn.ReLU())
        g = trace_to_graph(model)
        assert "relu" in g.nodes.values()

    def test_edges_preserved(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        g = trace_to_graph(model)
        # relu should take linear as input
        relu_nodes = [n for n, t in g.nodes.items() if t == "relu"]
        assert len(relu_nodes) == 1
        relu = relu_nodes[0]
        assert len(g.inputs[relu]) == 1
        assert g.nodes[g.inputs[relu][0]] == "linear"

    def test_skip_connection_model(self):
        class ResBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)

            def forward(self, x):
                return x + F.relu(self.linear(x))

        g = trace_to_graph(ResBlock())
        types = list(g.nodes.values())
        assert "linear" in types
        assert "relu" in types
        assert "add" in types
        # add has 1 internal input (relu); the skip from placeholder x is external.
        add_nodes = [n for n, t in g.nodes.items() if t == "add"]
        assert len(add_nodes) == 1
        assert len(g.inputs[add_nodes[0]]) >= 1

    def test_outputs_set(self):
        model = nn.Sequential(nn.Linear(4, 2))
        g = trace_to_graph(model)
        assert len(g.outputs) >= 1


# ── End-to-end: FX model → tile ──────────────────────────────────


class TestFxEndToEnd:
    def test_tile_sequential_with_itself(self):
        """tile(model, [model]) should give full coverage in 1 tile."""
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        g = trace_to_graph(model)
        result = tile(g, [g])
        assert result.fully_tiled
        assert len(result.tiles) == 1

    def test_tile_sequential_with_primitives(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        g = trace_to_graph(model)
        lib = [
            Graph.from_spec("1:linear"),
            Graph.from_spec("1:relu"),
        ]
        result = tile(g, lib)
        assert result.fully_tiled

    def test_tile_sequential_with_block(self):
        model = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(),
            nn.Linear(8, 4), nn.ReLU(),
        )
        g = trace_to_graph(model)
        block = Graph.from_spec("1:linear->2:relu")
        result = tile(g, [block])
        assert result.fully_tiled
        assert len(result.tiles) == 2

    def test_tile_resblock(self):
        """ResBlock: linear->relu->add, with skip -> add."""

        class ResBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)

            def forward(self, x):
                return x + F.relu(self.linear(x))

        g = trace_to_graph(ResBlock())
        lib = [
            Graph.from_spec("1:linear"),
            Graph.from_spec("1:relu"),
            Graph.from_spec("1:add"),
        ]
        result = tile(g, lib)
        assert result.fully_tiled

    def test_tile_two_layer_mlp_prefer_block(self):
        model = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(),
            nn.Linear(8, 4), nn.ReLU(),
        )
        g = trace_to_graph(model)
        lib = [
            Graph.from_spec("1:linear"),
            Graph.from_spec("1:relu"),
            Graph.from_spec("1:linear->2:relu"),
        ]
        result = tile(g, lib)
        assert result.fully_tiled
        # Should prefer two linear->relu blocks over four singles.
        assert len(result.tiles) == 2

    def test_resblock_with_block_pattern(self):
        """Test that linear->relu block works inside a residual model."""

        class TwoResBlocks(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(4, 4)
                self.l2 = nn.Linear(4, 4)

            def forward(self, x):
                y = x + F.relu(self.l1(x))
                z = y + F.relu(self.l2(y))
                return z

        g = trace_to_graph(TwoResBlocks())
        lib = [
            Graph.from_spec("1:linear->2:relu", outputs=["1", "2"]),
            Graph.from_spec("1:add"),
        ]
        result = tile(g, lib)
        assert result.fully_tiled

    def test_dot_visualization_e2e(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        g = trace_to_graph(model)
        result = tile(g, [Graph.from_spec("1:linear->2:relu")])
        dot = g.to_dot_with_tiling(result)
        assert "cluster_" in dot
        assert result.fully_tiled
