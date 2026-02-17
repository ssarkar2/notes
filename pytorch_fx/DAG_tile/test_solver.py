"""Tests for the DAG tiler (graph + matcher + solver)."""

import pytest

from graph import Graph
from solver import tile


# ── Graph DSL tests ───────────────────────────────────────────────


class TestGraphDSL:
    def test_simple_chain(self):
        g = Graph.from_spec("1:A->2:B->3:C")
        assert g.nodes == {"1": "A", "2": "B", "3": "C"}
        assert g.inputs == {"1": [], "2": ["1"], "3": ["2"]}

    def test_branch(self):
        g = Graph.from_spec("1:A->2:B; 1:A->3:C")
        assert g.inputs["2"] == ["1"]
        assert g.inputs["3"] == ["1"]

    def test_node_reference_without_type(self):
        g = Graph.from_spec("1:A->2:B; 1->3:C")
        assert g.nodes["1"] == "A"
        assert g.inputs["3"] == ["1"]

    def test_type_redefinition_raises(self):
        with pytest.raises(ValueError, match="cannot redefine"):
            Graph.from_spec("1:A->2:B; 1:X->3:C")

    def test_undefined_reference_raises(self):
        with pytest.raises(ValueError, match="not defined"):
            Graph.from_spec("1:A->2:B; 99->3:C")

    def test_auto_outputs_sinks(self):
        g = Graph.from_spec("1:A->2:B->3:C")
        assert g.outputs == ["3"]

    def test_auto_outputs_multi_sink(self):
        g = Graph.from_spec("1:A->2:B; 1:A->3:C")
        assert set(g.outputs) == {"2", "3"}

    def test_explicit_outputs(self):
        g = Graph.from_spec("1:A->2:B->3:C", outputs=["1", "3"])
        assert g.outputs == ["1", "3"]

    def test_consumers(self):
        g = Graph.from_spec("1:A->3:C; 2:B->3:C")
        assert set(g.consumers("1")) == {"3"}
        assert g.consumers("3") == []

    def test_empty_spec(self):
        g = Graph.from_spec("")
        assert len(g) == 0

    def test_single_node(self):
        g = Graph.from_spec("1:relu")
        assert g.nodes == {"1": "relu"}
        assert g.inputs == {"1": []}


# ── Basic tiling ──────────────────────────────────────────────────


class TestBasicTiling:
    def test_simple_chain_full_coverage(self):
        big = Graph.from_spec("1:A->2:B->3:C->4:D")
        lib = [Graph.from_spec("1:A->2:B"), Graph.from_spec("1:C->2:D")]
        result = tile(big, lib)
        assert result.fully_tiled
        assert len(result.tiles) == 2

    def test_repeated_pattern(self):
        big = Graph.from_spec("1:A->2:B->3:A->4:B")
        lib = [Graph.from_spec("1:A->2:B")]
        result = tile(big, lib)
        assert result.fully_tiled
        assert len(result.tiles) == 2

    def test_single_node_pattern(self):
        big = Graph.from_spec("1:relu->2:relu->3:relu")
        lib = [Graph.from_spec("1:relu")]
        result = tile(big, lib)
        assert result.fully_tiled
        assert len(result.tiles) == 3

    def test_no_library_matches(self):
        big = Graph.from_spec("1:A->2:B")
        lib = [Graph.from_spec("1:C->2:D")]
        result = tile(big, lib)
        assert result.coverage == 0
        assert len(result.tiles) == 0
        assert result.uncovered == frozenset({"1", "2"})

    def test_empty_library(self):
        big = Graph.from_spec("1:A->2:B")
        result = tile(big, [])
        assert result.coverage == 0

    def test_single_node_graph(self):
        big = Graph.from_spec("1:relu")
        lib = [Graph.from_spec("1:relu")]
        result = tile(big, lib)
        assert result.fully_tiled
        assert len(result.tiles) == 1


# ── Self-tiling ───────────────────────────────────────────────────


class TestSelfTiling:
    def test_chain_self_tile(self):
        big = Graph.from_spec("1:mm->2:relu->3:mm->4:relu")
        result = tile(big, [big])
        assert result.fully_tiled
        assert len(result.tiles) == 1

    def test_diamond_self_tile(self):
        big = Graph.from_spec("1:A->2:B->4:D; 1:A->3:C->4:D")
        result = tile(big, [big])
        assert result.fully_tiled
        assert len(result.tiles) == 1

    def test_single_node_self_tile(self):
        big = Graph.from_spec("1:X")
        result = tile(big, [big])
        assert result.fully_tiled
        assert len(result.tiles) == 1


# ── Partial coverage ──────────────────────────────────────────────


class TestPartialCoverage:
    def test_head_uncovered(self):
        """A->B->C with library=[B->C]: A is uncovered."""
        big = Graph.from_spec("1:A->2:B->3:C")
        lib = [Graph.from_spec("1:B->2:C")]
        result = tile(big, lib)
        assert result.coverage == 2
        assert result.uncovered == frozenset({"1"})

    def test_readme_example_partial(self):
        """A->B->C->B->C with library=[B->C]."""
        big = Graph.from_spec("1:A->2:B->3:C->4:B->5:C")
        lib = [Graph.from_spec("1:B->2:C")]
        result = tile(big, lib)
        assert result.coverage == 4  # B,C,B,C covered
        assert result.uncovered == frozenset({"1"})
        assert len(result.tiles) == 2

    def test_middle_uncovered(self):
        """A->B->C where library has only A and C (individually)."""
        big = Graph.from_spec("1:A->2:B->3:C")
        lib = [Graph.from_spec("1:A"), Graph.from_spec("1:C")]
        result = tile(big, lib)
        assert result.coverage == 2
        assert result.uncovered == frozenset({"2"})


# ── Prefer large tiles ───────────────────────────────────────────


class TestPreferLargeTiles:
    def test_readme_example_prefer_large(self):
        """mm->relu->mm->relu, library=[mm, relu, mm->relu]."""
        big = Graph.from_spec("1:mm->2:relu->3:mm->4:relu")
        lib = [
            Graph.from_spec("1:mm"),
            Graph.from_spec("1:relu"),
            Graph.from_spec("1:mm->2:relu"),
        ]
        result = tile(big, lib)
        assert result.fully_tiled
        # Should use two mm->relu tiles (2 tiles of size 2)
        # instead of four single-node tiles.
        assert len(result.tiles) == 2
        for t in result.tiles:
            assert len(t.covered_nodes) == 2

    def test_one_big_vs_many_small(self):
        """A->B->C, library=[A, B, C, A->B->C]."""
        big = Graph.from_spec("1:A->2:B->3:C")
        lib = [
            Graph.from_spec("1:A"),
            Graph.from_spec("1:B"),
            Graph.from_spec("1:C"),
            Graph.from_spec("1:A->2:B->3:C"),
        ]
        result = tile(big, lib)
        assert result.fully_tiled
        assert len(result.tiles) == 1  # one tile of size 3


# ── Output constraint ────────────────────────────────────────────


class TestOutputConstraint:
    def test_readme_model1_fails(self):
        """mm->relu tile where mm's value escapes, but pattern doesn't output mm."""
        big = Graph.from_spec("1:mm->2:relu->3:add; 1:mm->3:add")
        lib = [
            Graph.from_spec("1:mm->2:relu"),  # outputs=[relu] (auto)
            Graph.from_spec("1:add"),
        ]
        result = tile(big, lib)
        # mm->relu tile invalid (mm feeds add outside, but mm not a pattern output).
        # Only add can be covered.
        assert result.coverage == 1
        assert "3" not in result.uncovered  # add is covered

    def test_readme_model1_passes_with_explicit_outputs(self):
        """Same graph, but pattern exposes both mm and relu as outputs."""
        big = Graph.from_spec("1:mm->2:relu->3:add; 1:mm->3:add")
        lib = [
            Graph.from_spec("1:mm->2:relu", outputs=["1", "2"]),
            Graph.from_spec("1:add"),
        ]
        result = tile(big, lib)
        assert result.fully_tiled

    def test_fan_out_requires_output(self):
        """mm feeds both relu and sigmoid; tile {mm,relu} needs mm as output."""
        big = Graph.from_spec("1:mm->2:relu; 1:mm->3:sigmoid")
        # Pattern covers mm+relu but doesn't expose mm.
        lib_bad = [
            Graph.from_spec("1:mm->2:relu"),  # outputs=[relu]
            Graph.from_spec("1:sigmoid"),
        ]
        result = tile(big, lib_bad)
        # mm->relu invalid (mm escapes to sigmoid).
        # Best: cover sigmoid (1) + relu cannot be covered alone (no pattern)
        # Actually we have sigmoid pattern, so sigmoid covered.
        # mm and relu: no individual patterns. So coverage = 1.
        assert result.coverage == 1

        # Now fix: expose mm.
        lib_good = [
            Graph.from_spec("1:mm->2:relu", outputs=["1", "2"]),
            Graph.from_spec("1:sigmoid"),
        ]
        result = tile(big, lib_good)
        assert result.fully_tiled


# ── Commutativity ─────────────────────────────────────────────────


class TestCommutativity:
    def test_add_commutative(self):
        """Pattern has A,B order; big graph has B,A order for add."""
        pattern = Graph.from_spec("1:A->3:add; 2:B->3:add")
        big = Graph.from_spec("1:B->3:add; 2:A->3:add")
        result = tile(big, [pattern])
        assert result.fully_tiled

    def test_mul_commutative(self):
        pattern = Graph.from_spec("1:X->3:mul; 2:Y->3:mul")
        big = Graph.from_spec("1:Y->3:mul; 2:X->3:mul")
        result = tile(big, [pattern])
        assert result.fully_tiled

    def test_non_commutative_position_matters(self):
        """matmul is NOT commutative — input order must match."""
        pattern = Graph.from_spec("1:A->3:matmul; 2:B->3:matmul")
        big = Graph.from_spec("1:B->3:matmul; 2:A->3:matmul")
        result = tile(big, [pattern])
        # A must be at pos 0, B at pos 1. Big graph has B at 0, A at 1. No match.
        assert result.coverage == 0


# ── Diamond patterns ──────────────────────────────────────────────


class TestDiamond:
    def test_diamond_full_tile(self):
        big = Graph.from_spec("1:mm->2:relu->4:add; 1:mm->3:sigmoid->4:add")
        result = tile(big, [big])
        assert result.fully_tiled
        assert len(result.tiles) == 1

    def test_diamond_partial_with_singles(self):
        big = Graph.from_spec("1:mm->2:relu->4:add; 1:mm->3:sigmoid->4:add")
        lib = [
            Graph.from_spec("1:mm", outputs=["1"]),
            Graph.from_spec("1:relu"),
            Graph.from_spec("1:sigmoid"),
            Graph.from_spec("1:add"),
        ]
        result = tile(big, lib)
        assert result.fully_tiled

    def test_diamond_with_block_tile(self):
        """Tile mm->relu as a block; sigmoid and add individually."""
        big = Graph.from_spec("1:mm->2:relu->4:add; 1:mm->3:sigmoid->4:add")
        lib = [
            Graph.from_spec("1:mm->2:relu", outputs=["1", "2"]),
            Graph.from_spec("1:sigmoid"),
            Graph.from_spec("1:add"),
        ]
        result = tile(big, lib)
        assert result.fully_tiled


# ── Real-ish structures ──────────────────────────────────────────


class TestResNetLike:
    def test_two_identical_blocks(self):
        """conv->bn->relu repeated twice in sequence."""
        big = Graph.from_spec("1:conv->2:bn->3:relu->4:conv->5:bn->6:relu")
        block = Graph.from_spec("1:conv->2:bn->3:relu")
        result = tile(big, [block])
        assert result.fully_tiled
        assert len(result.tiles) == 2

    def test_resnet_block_with_skip(self):
        """conv->bn->relu->conv->bn->add->relu, with skip relu->add."""
        big = Graph.from_spec(
            "1:conv->2:bn->3:relu->4:conv->5:bn->6:add->7:relu;"
            "3:relu->6:add"
        )
        block = Graph.from_spec("1:conv->2:bn->3:relu", outputs=["1", "2", "3"])
        lib = [block, Graph.from_spec("1:conv"), Graph.from_spec("1:bn"),
               Graph.from_spec("1:relu"), Graph.from_spec("1:add")]
        result = tile(big, lib)
        assert result.fully_tiled
        # First block matched as conv->bn->relu; rest as singles.

    def test_resnet_two_residual_blocks(self):
        """Two residual blocks: each is conv->bn->add->relu with a skip."""
        big = Graph.from_spec(
            # Block 1
            "1:conv->2:bn->3:add->4:relu;"
            "0:input->3:add;"
            "0:input->1:conv;"
            # Block 2
            "4:relu->5:conv->6:bn->7:add->8:relu;"
            "4:relu->7:add"
        )
        lib = [
            Graph.from_spec("1:conv"),
            Graph.from_spec("1:bn"),
            Graph.from_spec("1:relu"),
            Graph.from_spec("1:add"),
            Graph.from_spec("1:input"),
        ]
        result = tile(big, lib)
        assert result.fully_tiled


class TestTransformerLike:
    def test_simple_ffn(self):
        """linear->relu->linear (feed-forward block)."""
        big = Graph.from_spec("1:linear->2:relu->3:linear")
        ffn = Graph.from_spec("1:linear->2:relu->3:linear")
        result = tile(big, [ffn])
        assert result.fully_tiled
        assert len(result.tiles) == 1

    def test_ffn_with_residual(self):
        """linear->relu->linear->add, with skip to add."""
        big = Graph.from_spec(
            "1:linear->2:relu->3:linear->4:add;"
            "0:input->4:add; 0:input->1:linear"
        )
        ffn = Graph.from_spec("1:linear->2:relu->3:linear", outputs=["1", "2", "3"])
        lib = [ffn, Graph.from_spec("1:add"), Graph.from_spec("1:input")]
        result = tile(big, lib)
        assert result.fully_tiled

    def test_attention_like(self):
        """Q/K/V projections -> matmul -> softmax -> matmul -> output proj."""
        big = Graph.from_spec(
            "1:linear->4:matmul;"
            "2:linear->4:matmul;"
            "4:matmul->5:softmax->6:matmul;"
            "3:linear->6:matmul;"
            "6:matmul->7:linear"
        )
        lib = [
            Graph.from_spec("1:linear"),
            Graph.from_spec("1:matmul"),
            Graph.from_spec("1:softmax"),
        ]
        result = tile(big, lib)
        assert result.fully_tiled
        assert len(result.tiles) == 7  # 4 linear + 2 matmul + 1 softmax


# ── DOT output ────────────────────────────────────────────────────


class TestDotOutput:
    def test_dot_basic(self):
        g = Graph.from_spec("1:A->2:B")
        dot = g.to_dot()
        assert "n1" in dot
        assert "n2" in dot
        assert "n1 -> n2" in dot

    def test_dot_with_tiling(self):
        big = Graph.from_spec("1:A->2:B->3:C")
        lib = [Graph.from_spec("1:A->2:B")]
        result = tile(big, lib)
        dot = big.to_dot_with_tiling(result)
        assert "cluster_" in dot
        assert "dashed" in dot  # uncovered node C
