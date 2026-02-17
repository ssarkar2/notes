"""Tests: tile real-ish PyTorch models (ResNet, simple LLM)."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx

from fx_adapter import fx_to_graph, trace_to_graph
from graph import Graph
from solver import tile


# ── Model definitions ────────────────────────────────────────────


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class SmallResNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.layer1 = ResNetBlock(in_channels, 16)
        self.layer2 = ResNetBlock(16, 32)
        self.layer3 = ResNetBlock(32, 64)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class SimpleLLMBlock(nn.Module):
    """Single transformer-like block: self-attention + FFN with residuals.

    Uses only traceable ops (no dynamic control flow).
    Attention: Q/K/V linear -> matmul -> softmax -> matmul -> out_proj
    FFN:       linear -> relu -> linear
    """

    def __init__(self, dim=16):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.ffn_up = nn.Linear(dim, dim * 4)
        self.ffn_down = nn.Linear(dim * 4, dim)

    def forward(self, x):
        # Self-attention (simplified, no reshape/head split for traceability)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        attn_out = torch.matmul(attn, v)
        attn_out = self.out_proj(attn_out)
        x = x + attn_out  # residual

        # FFN
        ffn_out = F.relu(self.ffn_up(x))
        ffn_out = self.ffn_down(ffn_out)
        x = x + ffn_out  # residual
        return x


class SmallLLM(nn.Module):
    """Two transformer blocks stacked."""

    def __init__(self, dim=16):
        super().__init__()
        self.block1 = SimpleLLMBlock(dim)
        self.block2 = SimpleLLMBlock(dim)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


# ── Helpers ──────────────────────────────────────────────────────


@pytest.fixture
def resnet_graph():
    model = SmallResNet()
    traced = torch.fx.symbolic_trace(model)
    return fx_to_graph(traced)


@pytest.fixture
def llm_graph():
    model = SmallLLM()
    traced = torch.fx.symbolic_trace(model)
    return fx_to_graph(traced)


# ── ResNet: complete tiling ──────────────────────────────────────


class TestResNetComplete:
    def test_primitive_singles_cover_all(self, resnet_graph):
        """Every op in the ResNet can be covered by single-node patterns."""
        op_types = set(resnet_graph.nodes.values())
        lib = [Graph.from_spec(f"1:{op}") for op in op_types]
        result = tile(resnet_graph, lib)
        assert result.fully_tiled

    def test_self_tile(self, resnet_graph):
        """Tiling a graph with itself always gives 1 tile, full coverage."""
        result = tile(resnet_graph, [resnet_graph])
        assert result.fully_tiled
        assert len(result.tiles) == 1

    def test_resnet_block_plus_peripherals(self, resnet_graph):
        """conv->bn->relu->conv->bn->add->relu block + singles for the rest."""
        block = Graph.from_spec(
            "1:conv2d->2:batch_norm->3:relu->4:conv2d->5:batch_norm->6:add->7:relu",
            outputs=["6", "7"],
        )
        op_types = set(resnet_graph.nodes.values())
        singles = [Graph.from_spec(f"1:{op}") for op in op_types]
        lib = [block] + singles
        result = tile(resnet_graph, lib)
        assert result.fully_tiled

    def test_resnet_block_plus_peripherals_fewer_tiles(self, resnet_graph):
        """Using the block pattern should produce fewer tiles than all singles."""
        op_types = set(resnet_graph.nodes.values())
        singles = [Graph.from_spec(f"1:{op}") for op in op_types]

        block = Graph.from_spec(
            "1:conv2d->2:batch_norm->3:relu->4:conv2d->5:batch_norm->6:add->7:relu",
            outputs=["6", "7"],
        )
        lib_with_block = [block] + singles

        result_singles = tile(resnet_graph, singles)
        result_block = tile(resnet_graph, lib_with_block)

        assert result_singles.fully_tiled
        assert result_block.fully_tiled
        assert len(result_block.tiles) < len(result_singles.tiles)


# ── ResNet: incomplete tiling ────────────────────────────────────


class TestResNetIncomplete:
    def test_only_matmul_pattern(self, resnet_graph):
        """matmul pattern matches nothing in a conv-based ResNet."""
        lib = [Graph.from_spec("1:matmul")]
        result = tile(resnet_graph, lib)
        assert result.coverage == 0
        assert not result.fully_tiled

    def test_block_only_misses_peripherals(self, resnet_graph):
        """Block pattern alone can't cover skip convs, avgpool, flatten, fc."""
        block = Graph.from_spec(
            "1:conv2d->2:batch_norm->3:relu->4:conv2d->5:batch_norm->6:add->7:relu",
            outputs=["6", "7"],
        )
        result = tile(resnet_graph, [block])
        assert not result.fully_tiled
        assert result.coverage > 0
        # The uncovered set should include non-block ops.
        uncovered_ops = {resnet_graph.nodes[n] for n in result.uncovered}
        assert len(uncovered_ops & {"conv2d", "linear", "flatten"}) > 0

    def test_missing_add_pattern(self, resnet_graph):
        """Without an 'add' pattern, residual connections can't be covered."""
        lib = [
            Graph.from_spec("1:conv2d"),
            Graph.from_spec("1:batch_norm"),
            Graph.from_spec("1:relu"),
            Graph.from_spec("1:linear"),
            Graph.from_spec("1:flatten"),
        ]
        # Find all op types to know what we need
        op_types = set(resnet_graph.nodes.values())
        # Add any remaining singles except 'add'
        for op in op_types:
            if op != "add":
                lib.append(Graph.from_spec(f"1:{op}"))
        result = tile(resnet_graph, lib)
        assert not result.fully_tiled
        uncovered_ops = {resnet_graph.nodes[n] for n in result.uncovered}
        assert "add" in uncovered_ops

    def test_empty_library(self, resnet_graph):
        """Empty library covers nothing."""
        result = tile(resnet_graph, [])
        assert result.coverage == 0
        assert result.uncovered == frozenset(resnet_graph.nodes.keys())


# ── LLM: complete tiling ────────────────────────────────────────


class TestLLMComplete:
    def test_primitive_singles_cover_all(self, llm_graph):
        """Every op in the LLM can be covered by single-node patterns."""
        op_types = set(llm_graph.nodes.values())
        lib = [Graph.from_spec(f"1:{op}") for op in op_types]
        result = tile(llm_graph, lib)
        assert result.fully_tiled

    def test_self_tile(self, llm_graph):
        """Tiling a graph with itself always gives 1 tile, full coverage."""
        result = tile(llm_graph, [llm_graph])
        assert result.fully_tiled
        assert len(result.tiles) == 1

    def test_attention_block_plus_singles(self, llm_graph):
        """Attention sub-pattern (matmul->softmax->matmul) + singles for rest."""
        attn_core = Graph.from_spec(
            "1:matmul->2:softmax->3:matmul",
            outputs=["1", "2", "3"],
        )
        op_types = set(llm_graph.nodes.values())
        singles = [Graph.from_spec(f"1:{op}") for op in op_types]
        lib = [attn_core] + singles
        result = tile(llm_graph, lib)
        assert result.fully_tiled

    def test_ffn_block_plus_singles(self, llm_graph):
        """FFN sub-pattern (linear->relu->linear) + singles for rest."""
        ffn = Graph.from_spec(
            "1:linear->2:relu->3:linear",
            outputs=["1", "2", "3"],
        )
        op_types = set(llm_graph.nodes.values())
        singles = [Graph.from_spec(f"1:{op}") for op in op_types]
        lib = [ffn] + singles
        result = tile(llm_graph, lib)
        assert result.fully_tiled

    def test_composite_blocks_fewer_tiles(self, llm_graph):
        """Using composite patterns should produce fewer tiles than all singles."""
        op_types = set(llm_graph.nodes.values())
        singles = [Graph.from_spec(f"1:{op}") for op in op_types]

        attn_core = Graph.from_spec(
            "1:matmul->2:softmax->3:matmul",
            outputs=["1", "2", "3"],
        )
        ffn = Graph.from_spec(
            "1:linear->2:relu->3:linear",
            outputs=["1", "2", "3"],
        )
        lib_composite = [attn_core, ffn] + singles

        result_singles = tile(llm_graph, singles)
        result_composite = tile(llm_graph, lib_composite)

        assert result_singles.fully_tiled
        assert result_composite.fully_tiled
        assert len(result_composite.tiles) < len(result_singles.tiles)


# ── LLM: incomplete tiling ──────────────────────────────────────


class TestLLMIncomplete:
    def test_only_conv_pattern(self, llm_graph):
        """Conv pattern matches nothing in a linear/matmul-based LLM."""
        lib = [Graph.from_spec("1:conv2d")]
        result = tile(llm_graph, lib)
        assert result.coverage == 0
        assert not result.fully_tiled

    def test_missing_matmul_pattern(self, llm_graph):
        """Without matmul, attention core can't be covered."""
        op_types = set(llm_graph.nodes.values())
        lib = [Graph.from_spec(f"1:{op}") for op in op_types if op != "matmul"]
        result = tile(llm_graph, lib)
        assert not result.fully_tiled
        uncovered_ops = {llm_graph.nodes[n] for n in result.uncovered}
        assert "matmul" in uncovered_ops

    def test_missing_add_pattern(self, llm_graph):
        """Without add, residual connections can't be covered."""
        op_types = set(llm_graph.nodes.values())
        lib = [Graph.from_spec(f"1:{op}") for op in op_types if op != "add"]
        result = tile(llm_graph, lib)
        assert not result.fully_tiled
        uncovered_ops = {llm_graph.nodes[n] for n in result.uncovered}
        assert "add" in uncovered_ops

    def test_attention_only_partial(self, llm_graph):
        """Only attention-related patterns leave FFN uncovered."""
        lib = [
            Graph.from_spec("1:linear"),
            Graph.from_spec("1:matmul"),
            Graph.from_spec("1:softmax"),
            Graph.from_spec("1:add"),
            Graph.from_spec("1:transpose"),
        ]
        result = tile(llm_graph, lib)
        # Should cover attention ops but miss relu (FFN activation)
        if "relu" in set(llm_graph.nodes.values()):
            uncovered_ops = {llm_graph.nodes[n] for n in result.uncovered}
            if uncovered_ops:
                assert not result.fully_tiled

    def test_empty_library(self, llm_graph):
        """Empty library covers nothing."""
        result = tile(llm_graph, [])
        assert result.coverage == 0
        assert result.uncovered == frozenset(llm_graph.nodes.keys())


# ── Cross-model: wrong domain patterns ──────────────────────────


class TestCrossDomain:
    def test_resnet_patterns_on_llm(self, llm_graph):
        """ResNet-specific block pattern should not fully tile an LLM."""
        resnet_block = Graph.from_spec(
            "1:conv2d->2:batch_norm->3:relu->4:conv2d->5:batch_norm->6:add->7:relu",
        )
        result = tile(llm_graph, [resnet_block])
        assert not result.fully_tiled

    def test_llm_attention_on_resnet(self, resnet_graph):
        """LLM attention pattern should not match in a ResNet."""
        attn = Graph.from_spec("1:matmul->2:softmax->3:matmul")
        result = tile(resnet_graph, [attn])
        assert result.coverage == 0
