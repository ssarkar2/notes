"""Backtracking DAG tiler with memoisation.

Objective (lexicographic):
  1. Maximise the number of covered nodes.
  2. Among equal-coverage solutions, minimise the number of tiles
     (i.e. prefer fewer, larger tiles).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field

from graph import Graph
from matcher import find_all_matches


@dataclass
class Tile:
    pattern_id: int
    covered_nodes: frozenset[str]
    mapping: dict[str, str] = field(repr=False)


@dataclass
class TilingResult:
    tiles: list[Tile]
    uncovered: frozenset[str]
    coverage: int
    total_nodes: int

    @property
    def fully_tiled(self) -> bool:
        return self.coverage == self.total_nodes


# ── Debug logger ─────────────────────────────────────────────────


class _DebugLogger:
    """Logs tiling decisions to files in a debug directory."""

    def __init__(self, debug_dir: str, big: Graph):
        self.dir = debug_dir
        self.big = big
        os.makedirs(debug_dir, exist_ok=True)
        os.makedirs(os.path.join(debug_dir, "steps"), exist_ok=True)
        os.makedirs(os.path.join(debug_dir, "library"), exist_ok=True)
        os.makedirs(os.path.join(debug_dir, "candidates"), exist_ok=True)
        self._candidate_idx = 0

        self._candidates_f = open(os.path.join(debug_dir, "candidates.txt"), "w")
        self._search_f = open(os.path.join(debug_dir, "search.txt"), "w")
        self._result_f = open(os.path.join(debug_dir, "result.txt"), "w")
        self._depth = 0
        self._step = 0
        self._tile_stack: list[Tile] = []

        # Write the input graph DOT.
        self._write_dot(big.to_dot("Input Graph"), "input_graph")

    def close(self):
        self._candidates_f.close()
        self._search_f.close()
        self._result_f.close()

    # ── Phase 1: candidate enumeration ──

    def log_candidate(self, tile: Tile, big: Graph, pattern: Graph):
        covered = ", ".join(
            f"{n}:{big.nodes[n]}" for n in sorted(tile.covered_nodes)
        )
        pat_ops = " -> ".join(
            f"{n}:{pattern.nodes[n]}" for n in pattern.nodes
        )
        self._candidates_f.write(
            f"candidate {self._candidate_idx}: "
            f"pattern {tile.pattern_id} [{pat_ops}]  =>  {{{covered}}}\n"
        )

        # Visualize this candidate placement on the big graph.
        title = (f"Candidate {self._candidate_idx}: "
                 f"pat {tile.pattern_id} ({pat_ops})")
        self._snapshot_tiles(
            [tile],
            os.path.join("candidates", f"cand_{self._candidate_idx:04d}"),
            title,
        )
        self._candidate_idx += 1

    def log_candidates_done(self, total: int):
        self._candidates_f.write(f"\nTotal candidate placements: {total}\n")
        self._candidates_f.flush()

    def log_library(self, library: list[Graph]):
        for i, pat in enumerate(library):
            ops = ", ".join(f"{n}:{pat.nodes[n]}" for n in pat.nodes)
            title = f"Pattern {i}: {ops}"
            self._write_dot(pat.to_dot(title), os.path.join("library", f"pat_{i:02d}"))

    # ── Phase 2: backtracking search ──

    def log_pick_node(self, node: str, num_tiles: int, uncovered_size: int):
        indent = "  " * self._depth
        op = self.big.nodes[node]
        self._search_f.write(
            f"{indent}pick node {node}:{op}  "
            f"({num_tiles} covering tiles, {uncovered_size} uncovered remain)\n"
        )

    def log_try_tile(self, tile: Tile):
        indent = "  " * self._depth
        covered = ", ".join(
            f"{n}:{self.big.nodes[n]}" for n in sorted(tile.covered_nodes)
        )
        self._step += 1
        self._search_f.write(
            f"{indent}  try tile pat={tile.pattern_id}  {{{covered}}}  [step {self._step}]\n"
        )
        self._snapshot(f"step_{self._step:04d}_try", f"Step {self._step}: try pat={tile.pattern_id}")

    def log_new_best(self, coverage: int, num_tiles: int, best_tiles: list[Tile]):
        indent = "  " * self._depth
        self._step += 1
        self._search_f.write(
            f"{indent}  -> new best: coverage={coverage}, tiles={num_tiles}  [step {self._step}]\n"
        )
        self._snapshot_tiles(
            best_tiles,
            os.path.join("steps", f"step_{self._step:04d}_best"),
            f"Step {self._step}: best cov={coverage} tiles={num_tiles}",
        )

    def log_skip_node(self, node: str):
        indent = "  " * self._depth
        op = self.big.nodes[node]
        self._search_f.write(f"{indent}  skip {node}:{op}\n")

    def log_memo_hit(self, uncovered_size: int):
        indent = "  " * self._depth
        self._search_f.write(
            f"{indent}  memo hit (uncovered size={uncovered_size})\n"
        )

    def log_prune(self):
        indent = "  " * self._depth
        self._search_f.write(f"{indent}  pruned (full sub-coverage)\n")

    def push_tile(self, tile: Tile):
        self._tile_stack.append(tile)

    def pop_tile(self):
        self._tile_stack.pop()

    def enter(self):
        self._depth += 1

    def leave(self):
        self._depth -= 1
        self._search_f.flush()

    # ── Final result ──

    def log_result(self, result: TilingResult):
        f = self._result_f
        status = "FULL" if result.fully_tiled else "PARTIAL"
        f.write(f"[{status}] coverage={result.coverage}/{result.total_nodes}  "
                f"tiles={len(result.tiles)}\n\n")

        for i, t in enumerate(result.tiles):
            labels = ", ".join(
                f"{n}:{self.big.nodes[n]}" for n in sorted(t.covered_nodes)
            )
            f.write(f"  tile {i}: pattern={t.pattern_id}  {{{labels}}}\n")

        if result.uncovered:
            labels = ", ".join(
                f"{n}:{self.big.nodes[n]}" for n in sorted(result.uncovered)
            )
            f.write(f"\n  uncovered: {{{labels}}}\n")

        f.flush()

        # Write the tiled graph DOT.
        dot = self.big.to_dot_with_tiling(result, "Tiled Graph")
        self._write_dot(dot, "tiled_graph")

    # ── Helpers ──

    def _snapshot(self, name: str, title: str):
        """Render the current exploration state (tile stack) as DOT/PNG."""
        self._snapshot_tiles(list(self._tile_stack), os.path.join("steps", name), title)

    def _snapshot_tiles(self, tiles: list[Tile], name: str, title: str):
        """Render an explicit list of tiles as DOT/PNG."""
        covered = frozenset().union(*(t.covered_nodes for t in tiles)) if tiles else frozenset()
        uncovered = frozenset(self.big.nodes) - covered
        tmp_result = TilingResult(
            tiles=tiles, uncovered=uncovered,
            coverage=len(covered), total_nodes=len(self.big.nodes),
        )
        dot = self.big.to_dot_with_tiling(tmp_result, title)
        self._write_dot(dot, name)

    def _write_dot(self, dot_str: str, name: str):
        dot_path = os.path.join(self.dir, f"{name}.dot")
        png_path = os.path.join(self.dir, f"{name}.png")
        with open(dot_path, "w") as f:
            f.write(dot_str)
        if shutil.which("dot"):
            subprocess.run(
                ["dot", "-Tpng", dot_path, "-o", png_path],
                check=False, capture_output=True,
            )

class _NoOpLogger:
    """Stub that does nothing — zero overhead when debug is off."""

    def log_candidate(self, *a, **kw): pass
    def log_candidates_done(self, *a, **kw): pass
    def log_library(self, *a, **kw): pass
    def log_pick_node(self, *a, **kw): pass
    def log_try_tile(self, *a, **kw): pass
    def log_new_best(self, *a, **kw): pass
    def log_skip_node(self, *a, **kw): pass
    def log_memo_hit(self, *a, **kw): pass
    def log_prune(self, *a, **kw): pass
    def push_tile(self, *a, **kw): pass
    def pop_tile(self, *a, **kw): pass
    def enter(self): pass
    def leave(self): pass
    def log_result(self, *a, **kw): pass
    def close(self): pass


# ── Public API ───────────────────────────────────────────────────


def tile(big: Graph, library: list[Graph], debug_dir: str | None = None) -> TilingResult:
    """Tile *big* using patterns drawn from *library*.

    Returns a :class:`TilingResult` with the best tiling found.

    If *debug_dir* is provided, creates that directory and writes detailed
    logs and DOT/PNG visualizations of each tiling phase into it:

    - ``candidates.txt`` — every candidate tile placement found
    - ``library/`` — DOT/PNG of each pattern graph in the library
    - ``input_graph.dot/.png`` — the input graph before tiling
    - ``search.txt`` — backtracking search decisions (with ``[step N]`` labels)
    - ``steps/`` — DOT/PNG snapshot for each labeled step in search.txt
    - ``result.txt`` — final coverage summary
    - ``tiled_graph.dot/.png`` — the graph with tile clusters colored
    """
    dbg: _DebugLogger | _NoOpLogger
    if debug_dir is not None:
        dbg = _DebugLogger(debug_dir, big)
    else:
        dbg = _NoOpLogger()

    try:
        return _tile_impl(big, library, dbg)
    finally:
        dbg.close()


def _tile_impl(big: Graph, library: list[Graph], dbg: _DebugLogger | _NoOpLogger) -> TilingResult:
    # ── Phase 1: enumerate every valid tile placement ─────────
    candidates: list[Tile] = []
    for pat_id, pattern in enumerate(library):
        for match in find_all_matches(big, pattern):
            covered = frozenset(match.values())
            t = Tile(pattern_id=pat_id, covered_nodes=covered, mapping=match)
            candidates.append(t)
            dbg.log_candidate(t, big, pattern)

    dbg.log_candidates_done(len(candidates))
    dbg.log_library(library)

    # Index: node -> tiles covering it, sorted largest-first.
    node_to_tiles: dict[str, list[Tile]] = {}
    for t in candidates:
        for n in t.covered_nodes:
            node_to_tiles.setdefault(n, []).append(t)
    for n in node_to_tiles:
        node_to_tiles[n].sort(key=lambda t: len(t.covered_nodes), reverse=True)

    all_nodes = frozenset(big.nodes)

    # ── Phase 2: backtracking search with memoisation ─────────
    # memo[uncovered] -> (coverage, num_tiles, tile_list)
    memo: dict[frozenset[str], tuple[int, int, list[Tile]]] = {}

    def solve(uncovered: frozenset[str]) -> tuple[int, int, list[Tile]]:
        if not uncovered:
            return (0, 0, [])
        if uncovered in memo:
            dbg.log_memo_hit(len(uncovered))
            return memo[uncovered]

        dbg.enter()

        # Pick the most-constrained node (fewest covering tiles → fail fast).
        node = min(uncovered, key=lambda n: len(node_to_tiles.get(n, [])))
        dbg.log_pick_node(node, len(node_to_tiles.get(node, [])), len(uncovered))

        # Option A: leave *node* uncovered (skip it).
        dbg.log_skip_node(node)
        best_cov, best_nt, best_tiles = solve(uncovered - {node})

        # Option B: cover *node* with a tile.
        for t in node_to_tiles.get(node, []):
            if t.covered_nodes <= uncovered:  # all tile nodes still uncovered
                dbg.push_tile(t)
                dbg.log_try_tile(t)
                sub_cov, sub_nt, sub_tiles = solve(uncovered - t.covered_nodes)
                cov = sub_cov + len(t.covered_nodes)
                nt = sub_nt + 1

                if cov > best_cov or (cov == best_cov and nt < best_nt):
                    best_cov = cov
                    best_nt = nt
                    best_tiles = sub_tiles + [t]
                    dbg.log_new_best(best_cov, best_nt, best_tiles)

                dbg.pop_tile()

                # Local pruning: everything in this sub-problem is covered.
                if best_cov == len(uncovered):
                    dbg.log_prune()
                    break

        memo[uncovered] = (best_cov, best_nt, best_tiles)
        dbg.leave()
        return (best_cov, best_nt, best_tiles)

    coverage, _num_tiles, tiles = solve(all_nodes)

    covered = frozenset().union(*(t.covered_nodes for t in tiles)) if tiles else frozenset()
    uncovered = all_nodes - covered

    result = TilingResult(
        tiles=tiles,
        uncovered=uncovered,
        coverage=coverage,
        total_nodes=len(all_nodes),
    )

    dbg.log_result(result)
    return result
