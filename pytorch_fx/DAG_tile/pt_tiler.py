"""PyTorch-level tiling wrapper.

Accepts ``nn.Module`` objects directly (both the big model and the library
of small models), traces them with FX, runs the DAG tiler, and returns a
result oriented toward **stitching compiled library binaries** into the big
model's compute graph.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch.nn as nn

from fx_adapter import trace_to_graph
from graph import Graph
from solver import tile, TilingResult, Tile


# ── Result dataclasses ────────────────────────────────────────────


@dataclass
class PtTile:
    """One tile in the final tiling, expressed in PyTorch-model terms."""

    library_index: int
    """Index into the *library* list passed to :func:`tile_model`."""

    library_module: nn.Module
    """Reference to the library ``nn.Module`` (the caller's compiled binary)."""

    covered_nodes: frozenset[str]
    """Big-model FX node names covered by this tile."""

    node_mapping: dict[str, str]
    """{big_node_name: library_node_name} — maps each covered node in the
    big model to its counterpart in the library module's traced graph."""

    input_nodes: list[str]
    """Big-graph node names **outside** this tile whose values feed **into**
    it.  These are the stitching inputs: the caller must route these values
    into the compiled binary."""

    output_nodes: list[str]
    """Big-graph node names **inside** this tile whose values are consumed
    **outside** it (by another tile or by uncovered nodes).  These are the
    stitching outputs: the caller must capture these from the compiled binary
    and forward them."""


@dataclass
class PtTilingResult:
    """Result of :func:`tile_model`, oriented toward stitching."""

    tiles: list[PtTile]
    """Tiles in topological execution order — safe to execute sequentially."""

    uncovered_nodes: frozenset[str]
    """Big-model FX node names not covered by any tile."""

    fully_tiled: bool
    coverage: int
    total_nodes: int

    big_graph: Graph
    """The traced big-model graph (for visualization / inspection)."""


# ── Public API ────────────────────────────────────────────────────


def tile_model(
    model: nn.Module,
    library: list[nn.Module],
    debug_dir: str | None = None,
) -> PtTilingResult:
    """Tile *model* using compiled library modules.

    Parameters
    ----------
    model : nn.Module
        The big model to tile.
    library : list[nn.Module]
        Small models whose compiled binaries the caller wants to stitch in.
    debug_dir : str | None
        If given, passed straight through to :func:`solver.tile` — writes
        candidates, search trace, step snapshots, and DOT/PNG visualizations
        into this directory.

    Returns
    -------
    PtTilingResult
        Tiles in topological order with stitching boundary info.
    """
    # 1. Trace the big model.
    big_graph = trace_to_graph(model)

    # 2. Trace each library module.  Mark all nodes as outputs so patterns
    #    that land in the middle of the big graph are never rejected by the
    #    output constraint.
    lib_graphs: list[Graph] = []
    for lib_mod in library:
        g = trace_to_graph(lib_mod)
        g.set_outputs(list(g.nodes.keys()))
        lib_graphs.append(g)

    # 3. Run the tiler.
    raw: TilingResult = tile(big_graph, lib_graphs, debug_dir=debug_dir)

    # 4. Convert Tile → PtTile with stitching boundary info.
    pt_tiles = [_to_pt_tile(t, library, big_graph) for t in raw.tiles]

    # 5. Topologically sort tiles.
    pt_tiles = _topo_sort(pt_tiles, big_graph)

    return PtTilingResult(
        tiles=pt_tiles,
        uncovered_nodes=raw.uncovered,
        fully_tiled=raw.fully_tiled,
        coverage=raw.coverage,
        total_nodes=raw.total_nodes,
        big_graph=big_graph,
    )


# ── Internals ─────────────────────────────────────────────────────


def _to_pt_tile(t: Tile, library: list[nn.Module], big: Graph) -> PtTile:
    """Convert a raw :class:`Tile` into a :class:`PtTile`."""
    # Invert mapping: raw mapping is {pattern_node: big_node},
    # we want {big_node: pattern_node} for the caller.
    node_mapping = {big_n: pat_n for pat_n, big_n in t.mapping.items()}

    # Input nodes: big-graph predecessors of covered nodes that are NOT
    # themselves covered.
    input_set: set[str] = set()
    for n in t.covered_nodes:
        for pred in big.inputs.get(n, []):
            if pred not in t.covered_nodes:
                input_set.add(pred)

    # Output nodes: covered nodes that have at least one consumer outside
    # the tile.
    output_set: set[str] = set()
    for n in t.covered_nodes:
        for consumer in big.consumers(n):
            if consumer not in t.covered_nodes:
                output_set.add(n)
                break

    return PtTile(
        library_index=t.pattern_id,
        library_module=library[t.pattern_id],
        covered_nodes=t.covered_nodes,
        node_mapping=node_mapping,
        input_nodes=sorted(input_set),
        output_nodes=sorted(output_set),
    )


def _topo_sort(tiles: list[PtTile], big: Graph) -> list[PtTile]:
    """Sort tiles in topological order based on inter-tile data flow."""
    if len(tiles) <= 1:
        return tiles

    # Build a mapping: big_node → tile index (for covered nodes only).
    node_owner: dict[str, int] = {}
    for i, t in enumerate(tiles):
        for n in t.covered_nodes:
            node_owner[n] = i

    # Build adjacency: tile i depends on tile j if an input_node of i is
    # covered by j.
    n = len(tiles)
    in_degree = [0] * n
    successors: defaultdict[int, list[int]] = defaultdict(list)
    for i, t in enumerate(tiles):
        seen: set[int] = set()
        for inp in t.input_nodes:
            j = node_owner.get(inp)
            if j is not None and j != i and j not in seen:
                seen.add(j)
                successors[j].append(i)
                in_degree[i] += 1

    # Kahn's algorithm.
    queue = [i for i in range(n) if in_degree[i] == 0]
    order: list[int] = []
    while queue:
        # Stable sort: among ready tiles, pick the one with the smallest
        # original index.
        queue.sort()
        cur = queue.pop(0)
        order.append(cur)
        for nxt in successors[cur]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    return [tiles[i] for i in order]
