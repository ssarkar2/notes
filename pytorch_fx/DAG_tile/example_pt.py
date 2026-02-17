"""Example: tile a PyTorch model using compiled library modules.

Demonstrates the pt_tiler API — pass in nn.Module objects directly and get
back a stitching-oriented result.

Run:
    python example_pt.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pt_tiler import tile_model


# ── Models ────────────────────────────────────────────────────────


class GatedResidualBlock(nn.Module):
    """x + down_proj(sigmoid(gate_proj(x)) * relu(up_proj(x)))"""

    def __init__(self, dim: int):
        super().__init__()
        self.up_proj = nn.Linear(dim, dim)
        self.down_proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(dim, dim)

    def forward(self, x):
        gate = torch.sigmoid(self.gate_proj(x))
        hidden = F.relu(self.up_proj(x))
        gated = gate * self.down_proj(hidden)
        return x + gated


class TwoBlockModel(nn.Module):
    def __init__(self, dim: int = 16):
        super().__init__()
        self.block1 = GatedResidualBlock(dim)
        self.block2 = GatedResidualBlock(dim)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


# ── Tile the model ────────────────────────────────────────────────

def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


section("1. Tile with full-block library")

model = TwoBlockModel(dim=16)

# The library: imagine the caller has a compiled binary for each of these.
library = [GatedResidualBlock(16)]

result = tile_model(model, library, debug_dir="pt_debug_output")

status = "FULL" if result.fully_tiled else "PARTIAL"
print(f"[{status}] coverage={result.coverage}/{result.total_nodes}  "
      f"tiles={len(result.tiles)}")
print()

for i, t in enumerate(result.tiles):
    mod_name = type(t.library_module).__name__
    print(f"Tile {i}: library[{t.library_index}] ({mod_name})")
    print(f"  covered : {sorted(t.covered_nodes)}")
    print(f"  inputs  : {t.input_nodes}  (feed into this compiled binary)")
    print(f"  outputs : {t.output_nodes}  (capture from this compiled binary)")
    print(f"  mapping : {dict(sorted(t.node_mapping.items()))}")
    print()

if result.uncovered_nodes:
    print(f"Uncovered: {sorted(result.uncovered_nodes)}")


# ── Stitching pseudocode ─────────────────────────────────────────

section("2. Stitching plan (pseudocode)")

print("The tiles are returned in topological order, so you can execute")
print("them sequentially.  For each tile:\n")
print("  compiled_binaries = {i: compile(lib_mod) for i, lib_mod in enumerate(library)}")
print()

for i, t in enumerate(result.tiles):
    mod_name = type(t.library_module).__name__
    print(f"  # Tile {i}: {mod_name}")
    if t.input_nodes:
        print(f"  #   Wire inputs: {t.input_nodes} -> compiled_binaries[{t.library_index}]")
    if t.output_nodes:
        print(f"  #   Capture outputs: compiled_binaries[{t.library_index}] -> {t.output_nodes}")
    print(f"  result_{i} = compiled_binaries[{t.library_index}](...)")
    print()

if result.uncovered_nodes:
    print(f"  # Fallback: run uncovered nodes ({sorted(result.uncovered_nodes)}) with eager PyTorch")


# ── Mixed-granularity library ─────────────────────────────────────

section("3. Mixed-granularity library (block + linear->relu)")

# Library with two patterns at different granularities.
library_mixed = [
    GatedResidualBlock(16),                              # full block
    nn.Sequential(nn.Linear(16, 16), nn.ReLU()),         # linear->relu fragment
]

result_mixed = tile_model(model, library_mixed)

status = "FULL" if result_mixed.fully_tiled else "PARTIAL"
print(f"[{status}] coverage={result_mixed.coverage}/{result_mixed.total_nodes}  "
      f"tiles={len(result_mixed.tiles)}")
print()

for i, t in enumerate(result_mixed.tiles):
    mod_name = type(t.library_module).__name__
    print(f"Tile {i}: library[{t.library_index}] ({mod_name})")
    print(f"  covered : {sorted(t.covered_nodes)}")
    print(f"  inputs  : {t.input_nodes}")
    print(f"  outputs : {t.output_nodes}")
    print()

print("The tiler prefers the full block (fewer, larger tiles) over multiple")
print("linear->relu fragments, since it maximises coverage first, then")
print("minimises tile count.")
