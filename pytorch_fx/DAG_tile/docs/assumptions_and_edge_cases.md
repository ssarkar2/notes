# Assumptions, Edge Cases, and Pitfalls

This document catalogues what the tiler assumes, what it handles correctly,
and what can trip it up.

---

## Assumptions

### 1. Op-type-only matching

The tiler compares nodes **only by their canonical op type string** (e.g.
`"mm"`, `"relu"`, `"add"`). It ignores:

- Tensor shapes and dtypes
- Constant values (bias terms, epsilon, etc.)
- Device placement
- Module configuration (e.g. `nn.Linear(4, 8)` vs `nn.Linear(16, 32)`)

**Implication**: two `linear` nodes always match, even if they have
different weight shapes. This is by design — the tiler finds structural
matches, not semantic ones. Shape validation is the caller's responsibility.

### 2. Static graph structure

The tiler operates on a fixed DAG. It does not handle:

- Dynamic control flow (`if`, loops, `torch.cond`)
- Data-dependent shapes
- In-place mutation side effects

PyTorch FX's `symbolic_trace` has the same limitation — it captures a
single execution trace. If your model uses `torch.cond` or Python `if`
on tensor values, tracing will fail or capture only one branch.

### 3. Connected patterns

Library patterns should be **connected graphs**. A disconnected pattern
(two unrelated subgraphs in one pattern) will technically work but is
unlikely to be useful and may produce unexpected matches.

### 4. Patterns are DAGs

Cycles in patterns are not supported (and don't make sense for compute
graphs). The big graph is also assumed to be a DAG.

---

## Edge cases handled correctly

### Commutativity (add, mul)

```
Pattern:  add(A, B)
Big:      add(B, A)   ← inputs swapped
```

The matcher recognises `add` and `mul` as commutative and accepts any
input ordering. All other ops require exact positional matching.

**To add more commutative ops**, edit `COMMUTATIVE_OPS` in `matcher.py`.

### Op aliasing (FX adapter)

The following all canonicalize to `"add"`:

- `operator.add` (Python `+`)
- `torch.add`
- `Tensor.add_` (in-place)
- `x.add(y)` (call_method)

Similarly, `relu` covers `F.relu`, `torch.relu`, `nn.ReLU`, `x.relu()`,
`F.relu_`.

The full mapping is in `fx_adapter.py` (`_FUNC_CANONICAL`,
`_METHOD_CANONICAL`, `_MODULE_CANONICAL`).

### Partial coverage

When the library cannot cover the entire graph, the solver still finds
the **maximum coverage** tiling and reports uncovered nodes.

```python
result = tile(big, lib)
result.fully_tiled   # False
result.coverage      # e.g. 8 out of 10
result.uncovered     # frozenset of node IDs
```

### Fan-out with output constraint

When a node inside a tile feeds consumers both inside and outside the tile,
the output constraint correctly rejects tiles where the escaping value
is not a declared pattern output.

```
Big:    mm -> relu,  mm -> sigmoid
Tile:   {mm, relu}

mm's value escapes to sigmoid.
If pattern outputs = [relu]     → REJECTED (mm not an output)
If pattern outputs = [mm, relu] → ACCEPTED
```

### Diamond / reconvergent paths

The matcher handles DAG patterns, not just chains. A pattern like:

```
A -> B -> D
A -> C -> D
```

correctly matches diamond structures in the big graph, including the
fan-out at A and reconvergence at D.

### Self-tiling

`tile(graph, [graph])` always produces exactly 1 tile with full coverage.
This is a useful sanity check.

---

## What can trip you up

### 1. Forgetting to set pattern outputs

This is the most common mistake. If your pattern covers nodes whose values
are used outside the tile, those nodes **must** be declared as outputs.

```python
# BAD — relu is the only auto-detected output
pattern = Graph.from_spec("1:linear->2:relu")

# GOOD — both exposed
pattern = Graph.from_spec("1:linear->2:relu", outputs=["1", "2"])
```

**Rule of thumb**: if a pattern is meant to be placed in the middle of a
larger graph (not at the very end), set `outputs` to include all nodes
that might fan out.

When in doubt, set all pattern nodes as outputs — this is maximally
permissive and the only cost is that the tiler won't reject placements
where values escape.

### 2. Input slot ordering for non-commutative ops

For non-commutative ops, the pattern's input slot order must match the
big graph's. The DSL assigns slots in declaration order:

```python
# Slot 0 = A, slot 1 = B
Graph.from_spec("1:A->3:matmul; 2:B->3:matmul")
```

If the big graph has B at slot 0 and A at slot 1, this won't match.
For commutative ops (add, mul) this doesn't matter.

**Mitigation**: when defining patterns for non-commutative ops, match the
input order used by PyTorch (e.g. `matmul(input, weight)` — input first,
weight second).

### 3. FX tracing limitations

Some modules can't be traced by `torch.fx.symbolic_trace`:

- Modules using `torch.cond`, `torch.vmap`, or Python control flow on
  tensor values
- Modules with `*args`/`**kwargs` forwarding in some configurations
- Custom autograd functions

For these, consider using `torch.compile` with `torch.export` instead of
`symbolic_trace`, or build the `Graph` manually using `from_spec`.

### 4. Placeholder and get_attr nodes are excluded

The FX adapter strips `placeholder` (function inputs) and `get_attr`
(parameter access) nodes. Only "real" compute nodes appear in the graph.

This means a `linear` node in the graph has **no inputs from its weight
and bias** — those are external. Its only input edges come from other
compute nodes (e.g., the previous layer's output).

This is usually what you want, but be aware if you're inspecting the
graph manually.

### 5. Scalability for large, non-decomposable subgraphs

The backtracking solver is exponential in the worst case. For graphs
where a single connected component has > ~30 nodes and the library
patterns don't cleanly cover it, the solver may be slow.

**Mitigations**:
- Decompose the big graph into independent blocks (e.g., one per
  transformer layer) and tile each separately.
- Ensure the library contains patterns that cleanly cover the model
  structure.
- For very large graphs, a greedy heuristic (not currently implemented)
  could serve as a fast fallback.

### 6. Associativity is not handled

```
(a + b) + c     vs     a + (b + c)
```

These produce different DAG structures and are **not** treated as equivalent.
A pattern matching one form will not match the other. If you need this,
add a canonicalization pass that normalizes associative chains to a
consistent form (e.g., left-leaning).

### 7. Duplicate edges in the DSL

The DSL appends to the input list every time an edge is declared:

```python
g = Graph.from_spec("1:A->2:B; 1:A->2:B")  # 2 edges from 1 to 2!
# g.inputs["2"] == ["1", "1"]
```

Avoid declaring the same edge twice. If you reference a node by `id:type`
again, it checks type consistency but still adds the edge.

### 8. Node ID collisions between pattern and big graph

Node IDs are just strings. The pattern and big graph use independent ID
namespaces — there's no collision risk. A pattern with node `"1"` can
match big-graph node `"42"`. The mapping dict tracks the correspondence.
