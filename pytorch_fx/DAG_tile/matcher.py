"""Subgraph pattern matching for DAG tiling.

Given a big graph and a small pattern graph, find every valid placement
(subgraph isomorphism on op-types) that also satisfies the output constraint:
any node inside the tile whose value escapes to a consumer outside the tile
must correspond to a declared output of the pattern.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graph import Graph

# Ops where input order does not matter.
COMMUTATIVE_OPS: frozenset[str] = frozenset({"add", "mul"})


# ── Public API ────────────────────────────────────────────────────

def find_all_matches(big: Graph, pattern: Graph) -> list[dict[str, str]]:
    """Return every valid placement of *pattern* inside *big*.

    Each match is a dict ``{pattern_node_id: big_graph_node_id}``.
    Matches are deduplicated by the set of big-graph nodes they cover
    (two mappings covering the same nodes are equivalent for tiling).
    """
    if not pattern.nodes:
        return []

    match_order = _get_match_order(pattern, big)
    results: list[dict[str, str]] = []

    def backtrack(idx: int, mapping: dict[str, str], used: set[str]):
        if idx == len(match_order):
            if _edges_match(big, pattern, mapping):
                if _output_constraint_ok(big, pattern, mapping):
                    results.append(dict(mapping))
            return

        p_node = match_order[idx]
        p_type = pattern.nodes[p_node]

        for g_node in _get_candidates(p_node, mapping, big, pattern):
            if g_node not in used and big.nodes.get(g_node) == p_type:
                mapping[p_node] = g_node
                used.add(g_node)
                backtrack(idx + 1, mapping, used)
                used.remove(g_node)
                del mapping[p_node]

    backtrack(0, {}, set())

    # Deduplicate by covered node set.
    seen: set[frozenset[str]] = set()
    deduped: list[dict[str, str]] = []
    for m in results:
        covered = frozenset(m.values())
        if covered not in seen:
            seen.add(covered)
            deduped.append(m)
    return deduped


# ── Internals ─────────────────────────────────────────────────────

def _pattern_neighbors(node: str, pattern: Graph) -> list[str]:
    """All neighbours of *node* in the pattern (inputs + consumers)."""
    nbrs: list[str] = list(pattern.inputs.get(node, []))
    for n, inp_list in pattern.inputs.items():
        if node in inp_list and n != node:
            nbrs.append(n)
    return nbrs


def _get_match_order(pattern: Graph, big: Graph) -> list[str]:
    """BFS traversal of *pattern* starting from the rarest op type."""
    type_counts = Counter(big.nodes.values())
    remaining = set(pattern.nodes)
    order: list[str] = []

    while remaining:
        # Seed from the rarest type among remaining nodes.
        start = min(remaining, key=lambda n: type_counts.get(pattern.nodes[n], 0))
        queue = [start]
        remaining.discard(start)
        order.append(start)

        while queue:
            cur = queue.pop(0)
            for nbr in _pattern_neighbors(cur, pattern):
                if nbr in remaining:
                    remaining.discard(nbr)
                    order.append(nbr)
                    queue.append(nbr)

    return order


def _get_candidates(
    p_node: str,
    mapping: dict[str, str],
    big: Graph,
    pattern: Graph,
) -> list[str]:
    """Candidate big-graph nodes for *p_node* based on already-matched neighbours."""
    candidate_sets: list[set[str]] = []

    # p_node feeds into an already-matched consumer.
    for consumer in pattern.consumers(p_node):
        if consumer in mapping:
            candidate_sets.append(set(big.inputs.get(mapping[consumer], [])))

    # An already-matched node feeds into p_node.
    for inp in pattern.inputs.get(p_node, []):
        if inp in mapping:
            candidate_sets.append(set(big.consumers(mapping[inp])))

    if not candidate_sets:
        # No neighbour constraints – fall back to type filtering.
        p_type = pattern.nodes[p_node]
        return [g for g in big.nodes if big.nodes[g] == p_type]

    result = candidate_sets[0]
    for cs in candidate_sets[1:]:
        result &= cs
    return list(result)


def _edges_match(big: Graph, pattern: Graph, mapping: dict[str, str]) -> bool:
    """Verify every pattern edge exists in the big graph (respecting positions)."""
    for p_node in pattern.nodes:
        g_node = mapping[p_node]
        p_inputs = pattern.inputs.get(p_node, [])
        if not p_inputs:
            continue

        g_inputs = big.inputs.get(g_node, [])
        mapped = [mapping[pi] for pi in p_inputs]

        if pattern.nodes[p_node] in COMMUTATIVE_OPS:
            # Each mapped input must appear *somewhere* in g_inputs.
            g_set = set(g_inputs)
            for mi in mapped:
                if mi not in g_set:
                    return False
        else:
            # Positional: pattern slot i must match big-graph slot i.
            for i, mi in enumerate(mapped):
                if i >= len(g_inputs) or g_inputs[i] != mi:
                    return False
    return True


def _output_constraint_ok(
    big: Graph, pattern: Graph, mapping: dict[str, str]
) -> bool:
    """Every value that escapes the tile must be a declared pattern output."""
    covered = set(mapping.values())
    pat_outputs = set(pattern.outputs)

    for p_node, g_node in mapping.items():
        for consumer in big.consumers(g_node):
            if consumer not in covered:
                # Value escapes — pattern node must be an output.
                if p_node not in pat_outputs:
                    return False
    return True
