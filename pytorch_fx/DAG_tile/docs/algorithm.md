# Algorithm Deep Dive

This document explains the three algorithmic phases of the DAG tiler in detail.
If you just want to use the tool, the [README](../README.md) is enough.
This is for anyone studying the code or extending it.

---

## Phase 1: Canonicalization (fx_adapter.py)

When working with PyTorch FX graphs, the same logical operation can appear
in several syntactic forms:

| Logical op | FX forms |
|---|---|
| add | `operator.add`, `torch.add`, `Tensor.add`, `Tensor.add_` |
| relu | `F.relu`, `torch.relu`, `nn.ReLU`, `x.relu()` |

The FX adapter maps all of these to a single canonical string (`"add"`,
`"relu"`, etc.) so the matcher only needs to compare strings.

The abstract `Graph` class used internally has no notion of PyTorch — it
stores `{node_id: op_type_string}` and `{node_id: [input_node_ids]}`.
This separation means the solver and matcher are pure graph algorithms,
independent of any ML framework.

---

## Phase 2: Pattern Matching (matcher.py)

Given a big graph **G** and a small pattern graph **P**, we want every
valid placement of **P** inside **G**. A valid placement is a
**subgraph isomorphism on op types**: an injective mapping
`m: P.nodes → G.nodes` such that:

1. **Type match**: `P.type(p) == G.type(m(p))` for every pattern node `p`.
2. **Edge match**: every edge in the pattern corresponds to an edge in **G**
   (see [Position matching](#position-matching) below).
3. **Output constraint**: see [Output constraint](#output-constraint) below.

### Search strategy

Brute-force enumeration of all injective mappings is `O(N^K)` where `N = |G|`
and `K = |P|`. We tame this with two techniques:

**a) Match ordering.**
Pattern nodes are processed in BFS order starting from the node whose op type
is *rarest* in **G** (fewest candidates → fail fast). Because the BFS ensures
every node (except the first) has at least one already-matched neighbour, the
candidate set for each subsequent node is tightly constrained.

**b) Neighbour-based candidate generation.**
When matching pattern node `p`:
- If `p` feeds into an already-matched consumer `q`, then `m(p)` must be
  among the inputs of `m(q)` in **G**.
- If an already-matched node `r` feeds into `p`, then `m(p)` must be
  among the consumers of `m(r)` in **G**.

These constraints are intersected, typically reducing the candidate set to
1–3 nodes.

### Position matching

Edges carry positional information: `inputs[node][i]` is the node at input
slot `i`. For **non-commutative** ops (matmul, linear, conv, ...) the
pattern's slot indices must match the big graph's exactly. For
**commutative** ops (add, mul) any permutation is accepted — we just check
set membership.

```
Pattern:  add(A, B)     — A at slot 0, B at slot 1
Big:      add(B', A')   — B' at slot 0, A' at slot 1

Non-commutative: FAIL (m(A) must be at slot 0, but it's at slot 1)
Commutative:     PASS (m(A) ∈ {A', B'} ✓, m(B) ∈ {A', B'} ✓)
```

### Output constraint

When a tile covers a subset of **G**'s nodes, some values may "escape" — a
node inside the tile feeds a consumer *outside* the tile. For the tile to be
valid, those escaping values must be **declared outputs** of the pattern.

```
Pattern P:  mm -> relu,  outputs = [relu]
Big G:      mm -> relu -> add,  mm -> add

Tile {mm, relu}:  mm's value escapes to add, but mm is NOT a pattern output.
→ INVALID.

Fix:  outputs = [mm, relu]  → mm's escape is allowed → VALID.
```

This is checked per-match during precomputation (not during the backtracking
search), so it has zero cost during the solve phase.

### Deduplication

Two different node-to-node mappings that cover the **same set** of big-graph
nodes are equivalent for tiling purposes. We deduplicate by `frozenset` of
covered nodes.

---

## Phase 3: Backtracking Solver (solver.py)

The solver takes the list of candidate tiles from Phase 2 and finds an optimal
tiling using backtracking with memoization.

### Objective (lexicographic)

1. **Maximize coverage** — cover as many nodes as possible.
2. **Minimize tile count** — among equal-coverage solutions, prefer fewer
   (i.e. larger) tiles.

### State representation

The state is the **set of uncovered nodes**, represented as a Python
`frozenset[str]`. Two search paths that arrive at the same uncovered set
are identical subproblems.

### Recurrence

```
solve(uncovered):
    if uncovered is empty:
        return (0 coverage, 0 tiles, [])

    pick node n from uncovered  (most-constrained-first heuristic)

    best = solve(uncovered - {n})          # Option A: skip n

    for each tile T covering n:            # Option B: cover n
        if T.covered_nodes ⊆ uncovered:
            sub = solve(uncovered - T.covered_nodes)
            candidate = (sub.coverage + |T|, sub.tiles + 1, ...)
            if candidate better than best:
                best = candidate
            if best.coverage == |uncovered|:
                break                      # local pruning: everything covered

    memo[uncovered] = best
    return best
```

### Heuristics and pruning

| Technique | What it does |
|---|---|
| Most-constrained-first | Pick the uncovered node with the fewest candidate tiles. If a node has 0 candidates, it's immediately skipped. |
| Largest-tile-first | Tiles for each node are sorted by size (descending). Larger tiles are tried first, increasing the chance of early full coverage. |
| Local pruning | If the current subproblem achieves full coverage, stop trying more tiles. |
| Memoization | `frozenset → (coverage, tile_count, tiles)`. Avoids re-solving identical subproblems reached via different paths. |

### Why not global branch-and-bound?

Global pruning (skip a subproblem if `coverage_so_far + |remaining| ≤ global_best`)
conflicts with memoization. The memo must store the **true optimum** for each
subproblem, but global pruning may cause early exit with a suboptimal answer
that then gets cached and reused incorrectly in a different context.

We use **memoization + local pruning** instead, which is always correct and
covers the most impactful optimizations.

### Complexity

| Metric | Worst case | Typical (LLM block) |
|---|---|---|
| States | `O(2^N)` | `O(2^B)` where `B` = block size (20–40) |
| Per state | `O(T)` where T = tiles per node | Small |
| Space | `O(2^N · N)` for memo | Feasible for B ≤ ~30 |

The exponential worst case is expected — the problem is a variant of
**set cover** (NP-hard). But real deep learning graphs are highly structured
and repetitive, so:

- **Graph decomposition**: An LLM is L identical transformer blocks.
  Tile one block, reuse for all L.
- **Reachable states ≪ 2^B**: Tiles are contiguous subgraphs, so only a
  tiny fraction of all subsets are reachable.
- **Early termination**: The library is typically designed to cover the model,
  so full coverage is found quickly.

---

## Worked example

```
Big graph:    mm -> relu -> mm -> relu
Library:      [mm], [relu], [mm -> relu]

Phase 2 finds these candidate tiles:
  T0: {1:mm, 2:relu}   (pattern: mm->relu)
  T1: {3:mm, 4:relu}   (pattern: mm->relu)
  T2: {1:mm}            (pattern: mm)
  T3: {2:relu}          (pattern: relu)
  T4: {3:mm}            (pattern: mm)
  T5: {4:relu}          (pattern: relu)

Phase 3 backtracking:
  solve({1,2,3,4})
    pick node with fewest tiles, say node 1 (has T0, T2)
    try T0 (largest first, size 2): covers {1,2}
      solve({3,4})
        pick node 3 (has T1, T4)
        try T1 (size 2): covers {3,4}
          solve({}) → (0, 0, [])
        → (2, 1, [T1])
        coverage = 2 == |{3,4}| → local pruning, stop
      → (2, 1, [T1])
    candidate: (2+2, 1+1, [T1, T0]) = (4, 2)
    4 == |{1,2,3,4}| → local pruning, stop
  → (4, 2, [T1, T0])

Result: 2 tiles of mm->relu, full coverage. ✓
```
