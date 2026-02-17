"""
DAG Tiler — Example Usage
=========================

This file is a walkthrough of the DAG tiler. It covers:

  1. Defining graphs with the compact DSL
  2. Tiling with the backtracking solver
  3. Key behaviours: prefer-large-tiles, partial coverage, output
     constraints, commutativity
  4. End-to-end: tracing a real PyTorch model and tiling it

Run:  python example.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph import Graph
from solver import tile
from fx_adapter import trace_to_graph


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def show(result, big=None):
    """Pretty-print a TilingResult."""
    status = "FULL" if result.fully_tiled else "PARTIAL"
    print(f"  [{status}] coverage={result.coverage}/{result.total_nodes}  "
          f"tiles={len(result.tiles)}")
    for i, t in enumerate(result.tiles):
        nodes = ", ".join(f"{n}" for n in sorted(t.covered_nodes))
        if big:
            labels = ", ".join(f"{n}:{big.nodes[n]}" for n in sorted(t.covered_nodes))
            print(f"    tile {i}: {{{labels}}}")
        else:
            print(f"    tile {i}: {{{nodes}}}")
    if result.uncovered:
        if big:
            labels = ", ".join(f"{n}:{big.nodes[n]}" for n in sorted(result.uncovered))
            print(f"    uncovered: {{{labels}}}")
        else:
            print(f"    uncovered: {{{', '.join(sorted(result.uncovered))}}}")


# ─────────────────────────────────────────────────────────────────
# 1. THE DSL — defining graphs compactly
# ─────────────────────────────────────────────────────────────────

section("1. The DSL")

# Nodes are defined as  id:type.  Edges are  ->.  Chains separated by ;
# The first edge into a node sets input slot 0, the second slot 1, etc.

g = Graph.from_spec("1:mm->2:relu->3:mm->4:relu")
print("Chain:  1:mm -> 2:relu -> 3:mm -> 4:relu")
print(f"  nodes:   {g.nodes}")
print(f"  inputs:  {g.inputs}")
print(f"  outputs: {g.outputs}")  # auto-detected sinks

# Branches / fan-out: use ; to start a new chain from an existing node.
g2 = Graph.from_spec("1:mm->2:relu; 1:mm->3:sigmoid")
print("\nFan-out:  mm -> relu,  mm -> sigmoid")
print(f"  consumers of mm: {g2.consumers('1')}")

# Reference a node by id alone (no :type) after it has been defined.
g3 = Graph.from_spec("1:A->2:B; 1->3:C")  # node 1 already defined as A
print(f"\nBack-reference:  1:A->2:B; 1->3:C  =>  nodes={g3.nodes}")

# Type consistency is enforced — redefining a node with a different type
# raises ValueError:
try:
    Graph.from_spec("1:A->2:B; 1:X->3:C")
except ValueError as e:
    print(f"\nType redefinition caught: {e}")


# ─────────────────────────────────────────────────────────────────
# 2. BASIC TILING — chains and repeated patterns
# ─────────────────────────────────────────────────────────────────

section("2. Basic tiling")

# Tile a chain of 4 ops using a 2-op pattern that repeats twice.
big = Graph.from_spec("1:mm->2:relu->3:mm->4:relu")
lib = [Graph.from_spec("1:mm->2:relu")]

print("big:     mm -> relu -> mm -> relu")
print("library: [mm -> relu]")
result = tile(big, lib)
show(result, big)
# Expected: 2 tiles, full coverage.


# ─────────────────────────────────────────────────────────────────
# 3. SELF-TILING — tile(graph, [graph]) always works in 1 tile
# ─────────────────────────────────────────────────────────────────

section("3. Self-tiling")

big = Graph.from_spec("1:A->2:B->4:D; 1:A->3:C->4:D")
print("big (diamond):  A -> B -> D,  A -> C -> D")
result = tile(big, [big])
show(result, big)
# Expected: 1 tile, full coverage.


# ─────────────────────────────────────────────────────────────────
# 4. PARTIAL COVERAGE — when full tiling isn't possible
# ─────────────────────────────────────────────────────────────────

section("4. Partial coverage")

# From the README:  A->B->C->B->C  with library=[B->C].
# A cannot be covered, but B->C tiles are placed twice.
big = Graph.from_spec("1:A->2:B->3:C->4:B->5:C")
lib = [Graph.from_spec("1:B->2:C")]

print("big:     A -> B -> C -> B -> C")
print("library: [B -> C]")
result = tile(big, lib)
show(result, big)
# Expected: coverage=4/5, A uncovered.


# ─────────────────────────────────────────────────────────────────
# 5. PREFER LARGE TILES — solver minimises tile count
# ─────────────────────────────────────────────────────────────────

section("5. Prefer large tiles")

# Library has singles AND a composite.  Solver should prefer the composite.
big = Graph.from_spec("1:mm->2:relu->3:mm->4:relu")
lib = [
    Graph.from_spec("1:mm"),           # size 1
    Graph.from_spec("1:relu"),         # size 1
    Graph.from_spec("1:mm->2:relu"),   # size 2  <-- preferred
]

print("big:     mm -> relu -> mm -> relu")
print("library: [mm], [relu], [mm -> relu]")
result = tile(big, lib)
show(result, big)
# Expected: 2 tiles of size 2, not 4 tiles of size 1.
print(f"  (tile sizes: {[len(t.covered_nodes) for t in result.tiles]})")


# ─────────────────────────────────────────────────────────────────
# 6. OUTPUT CONSTRAINT — values escaping a tile must be outputs
# ─────────────────────────────────────────────────────────────────

section("6. Output constraint")

# From the README:
#   big:  mm -> relu -> add,  mm -> add   (mm fans out to relu AND add)
#   pattern: mm -> relu  (auto outputs = [relu],  mm is NOT an output)
#
# The mm->relu tile is INVALID because mm's value escapes to add,
# but mm is not a declared output of the pattern.

big = Graph.from_spec("1:mm->2:relu->3:add; 1:mm->3:add")

print("big:  mm -> relu -> add,  mm -> add  (mm fans out)")
print()

# BAD: pattern doesn't expose mm as output.
lib_bad = [
    Graph.from_spec("1:mm->2:relu"),   # outputs=[relu] (auto)
    Graph.from_spec("1:add"),
]
print("library (bad): [mm->relu (output=relu only)], [add]")
result = tile(big, lib_bad)
show(result, big)
# Expected: only add is covered (coverage 1/3).  mm->relu tile rejected.

print()

# GOOD: pattern exposes both mm and relu as outputs.
lib_good = [
    Graph.from_spec("1:mm->2:relu", outputs=["1", "2"]),
    Graph.from_spec("1:add"),
]
print("library (good): [mm->relu (outputs=mm,relu)], [add]")
result = tile(big, lib_good)
show(result, big)
# Expected: full coverage.


# ─────────────────────────────────────────────────────────────────
# 7. COMMUTATIVITY — add and mul match regardless of input order
# ─────────────────────────────────────────────────────────────────

section("7. Commutativity")

# Pattern defines add(A, B).  Big graph has add(B, A) — reversed inputs.
# Because add is commutative, the match succeeds.
pattern = Graph.from_spec("1:A->3:add; 2:B->3:add")
big     = Graph.from_spec("1:B->3:add; 2:A->3:add")

print("pattern: add(A, B)")
print("big:     add(B, A)   (inputs swapped)")
result = tile(big, [pattern])
show(result, big)
# Expected: full coverage — commutativity allows the swap.

print()

# matmul is NOT commutative — swapping inputs should fail.
pattern = Graph.from_spec("1:A->3:matmul; 2:B->3:matmul")
big     = Graph.from_spec("1:B->3:matmul; 2:A->3:matmul")

print("pattern: matmul(A, B)")
print("big:     matmul(B, A)  (inputs swapped)")
result = tile(big, [pattern])
show(result, big)
# Expected: no match — matmul is not commutative.


# ─────────────────────────────────────────────────────────────────
# 8. PYTORCH FX END-TO-END — a model that exercises everything
# ─────────────────────────────────────────────────────────────────

section("8. PyTorch FX end-to-end")

# A model with enough structure to probe multiple features:
#   - Repeated blocks              (tests pattern reuse)
#   - Skip / residual connections  (tests fan-out + output constraint)
#   - Gating via mul               (tests commutativity)
#   - Varying layer types          (tests partial coverage)
#
#               ┌──────────────────────┐
#   x ──► linear ──► relu ──► linear ──┤
#         │                            ▼
#         │                    mul (gate) ◄── sigmoid(linear(x))
#         │                      │
#         └──────────► add ◄─────┘
#                       │
#                  ► linear ──► relu ──► linear ──┐
#                       │                         ▼
#                       └──────────────► add ◄────┘
#                                         │
#                                       output


class GatedResidualBlock(nn.Module):
    """
    Residual block with a sigmoid gate:
        out = x + sigmoid(gate_proj(x)) * relu(up_proj(x))

    This produces:
      - linear (up_proj) -> relu -> linear (down_proj) chain
      - linear (gate_proj) -> sigmoid -> mul (gating, commutative!)
      - add (residual, commutative!)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.up_proj   = nn.Linear(dim, dim)
        self.down_proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(dim, dim)

    def forward(self, x):
        gate = torch.sigmoid(self.gate_proj(x))
        hidden = F.relu(self.up_proj(x))
        gated = gate * self.down_proj(hidden)   # mul — commutative
        return x + gated                         # add — commutative


class TwoBlockModel(nn.Module):
    """Stack two GatedResidualBlocks."""
    def __init__(self, dim: int = 16):
        super().__init__()
        self.block1 = GatedResidualBlock(dim)
        self.block2 = GatedResidualBlock(dim)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


# ── Trace and convert ──

model = TwoBlockModel(dim=16)
big = trace_to_graph(model)

print(f"Traced model → {len(big)} compute nodes")
print(f"Op types: {sorted(set(big.nodes.values()))}")
print(f"Node list:")
for nid in big.nodes:
    inps = big.inputs[nid]
    inp_str = ", ".join(inps) if inps else "(external)"
    print(f"  {nid:30s}  type={big.nodes[nid]:10s}  inputs=[{inp_str}]")


# ── Tile with primitives ──

print("\n--- Tiling with single-op primitives ---")
primitives = [
    Graph.from_spec("1:linear"),
    Graph.from_spec("1:relu"),
    Graph.from_spec("1:sigmoid"),
    Graph.from_spec("1:mul"),
    Graph.from_spec("1:add"),
]
result = tile(big, primitives)
show(result, big)


# ── Tile with a composite block + primitives ──

print("\n--- Tiling with [linear->relu] block + primitives ---")
# The linear->relu block needs to expose linear as output too,
# because linear's value might fan out (e.g., to the gate path).
lr_block = Graph.from_spec("1:linear->2:relu", outputs=["1", "2"])
lib_with_block = [lr_block] + primitives
result = tile(big, lib_with_block)
show(result, big)
print(f"  (tile sizes: {sorted([len(t.covered_nodes) for t in result.tiles], reverse=True)})")


# ── Tile with the full GatedResidualBlock pattern ──

print("\n--- Tiling with full GatedResidualBlock pattern ---")
# Build the pattern manually to match the block's structure:
#   linear(gate) -> sigmoid -> mul
#   linear(up)   -> relu -> linear(down) -> mul
#   mul -> add
# All intermediate values that fan out must be outputs.
block_pattern = Graph.from_spec(
    "1:linear->2:sigmoid->5:mul;"
    "3:linear->4:relu->6:linear->5:mul;"
    "5:mul->7:add",
    outputs=["1", "2", "3", "4", "5", "6", "7"],  # expose everything
)
result = tile(big, [block_pattern])
show(result, big)


# ── Generate DOT visualisation ──

print("\n--- DOT visualisation (saved to tiling_example.png) ---")
dot = big.to_dot_with_tiling(result, title="TwoBlockModel Tiling")
with open("tiling_example.dot", "w") as f:
    f.write(dot)

# Try to render to PNG if graphviz is available.
import shutil, subprocess
if shutil.which("dot"):
    subprocess.run(["dot", "-Tpng", "tiling_example.dot", "-o", "tiling_example.png"],
                   check=True)
    print("  Written: tiling_example.dot, tiling_example.png")
else:
    print("  Written: tiling_example.dot  (install graphviz to render PNG)")


# ─────────────────────────────────────────────────────────────────
# 9. DEBUG MODE — inspect the solver's decisions
# ─────────────────────────────────────────────────────────────────

section("9. Debug mode")

print("Running tile() with debug_dir='debug_output' ...")
print("This writes logs and visualizations into the debug_output/ folder.\n")

# Re-tile the TwoBlockModel with composite block + primitives, this time
# with debug logging enabled.
debug_result = tile(big, [block_pattern], debug_dir="debug_output")
show(debug_result, big)

import os
print("\nFiles created in debug_output/:")
for fname in sorted(os.listdir("debug_output")):
    fpath = os.path.join("debug_output", fname)
    if os.path.isdir(fpath):
        step_files = sorted(os.listdir(fpath))
        print(f"  {fname + '/':30s}  ({len(step_files)} files)")
        for sf in step_files:
            sf_path = os.path.join(fpath, sf)
            print(f"    {sf:28s}  ({os.path.getsize(sf_path):,} bytes)")
    else:
        size = os.path.getsize(fpath)
        print(f"  {fname:30s}  ({size:,} bytes)")


# ─────────────────────────────────────────────────────────────────
# 10. SUMMARY
# ─────────────────────────────────────────────────────────────────

section("10. Summary")

print("""\
Key takeaways:

  1. from_spec DSL:   Graph.from_spec("1:mm->2:relu; 1:mm->3:add")
     Quick way to define graphs for testing.

  2. tile(big, library) returns a TilingResult:
     - .fully_tiled    — did we cover every node?
     - .coverage       — number of covered nodes
     - .tiles          — list of Tile objects (pattern_id + covered_nodes)
     - .uncovered      — nodes that couldn't be tiled

  3. The solver maximises coverage first, then minimises tile count
     (preferring fewer, larger tiles).

  4. Output constraint:  if a value inside a tile escapes to a consumer
     outside the tile, the pattern must declare that value as an output.
     Use:  Graph.from_spec("...", outputs=["node1", "node2"])

  5. Commutativity:  add and mul match regardless of input order.
     Other ops (matmul, linear, ...) require exact position matching.

  6. FX integration:
       g = trace_to_graph(my_module)
       result = tile(g, library)
       dot = g.to_dot_with_tiling(result)

  7. Debug mode:
       result = tile(g, library, debug_dir="my_debug")
     Writes candidates.txt, search.txt, result.txt, and DOT/PNG graphs.
""")
