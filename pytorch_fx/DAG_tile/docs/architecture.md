# Architecture

Overview of how the codebase is structured and how the pieces connect.

---

## Design principle: two-layer separation

```
┌─────────────────────────────────────────────────────┐
│  Layer 2: PyTorch FX adapter  (fx_adapter.py)       │
│  - Traces nn.Module → torch.fx.Graph                │
│  - Canonicalizes op types                           │
│  - Converts to abstract Graph                       │
└──────────────────────┬──────────────────────────────┘
                       │  Graph objects
┌──────────────────────▼──────────────────────────────┐
│  Layer 1: Abstract graph solver                     │
│  - graph.py     — data structure + DSL + DOT output │
│  - matcher.py   — subgraph pattern matching         │
│  - solver.py    — backtracking tiler                │
└─────────────────────────────────────────────────────┘
```

**Layer 1** knows nothing about PyTorch. It works on graphs with string-typed
nodes. All core logic lives here, and all core tests use the compact DSL
to define graphs inline.

**Layer 2** is a thin adapter. It converts `torch.fx.GraphModule` into
Layer 1's `Graph` class. It handles the messiness of op aliasing
(`operator.add` vs `torch.add` vs `Tensor.add_`) so Layer 1 doesn't have to.

This separation means:
- Tests are fast and don't need GPU/PyTorch for the solver logic.
- The solver could be reused for non-PyTorch graph IRs (ONNX, TVM, etc.)
  by writing a different adapter.

---

## File map

```
DAG_tile3/
├── graph.py              Core data structure
├── matcher.py            Subgraph isomorphism + output constraint
├── solver.py             Backtracking tiler + TilingResult
├── fx_adapter.py         PyTorch FX → Graph conversion
├── example.py            Runnable walkthrough (start here)
├── test_solver.py        42 tests for graph/matcher/solver
├── test_fx.py            14 tests for FX adapter + end-to-end
├── docs/
│   ├── algorithm.md      Deep dive into matching + backtracking
│   ├── assumptions_and_edge_cases.md
│   └── architecture.md   This file
└── README.md             Quick-start guide
```

---

## Data flow

```
nn.Module
   │
   │  torch.fx.symbolic_trace
   ▼
torch.fx.GraphModule
   │
   │  fx_adapter.fx_to_graph()     ← op canonicalization happens here
   ▼
Graph  (big graph)
   │
   │  + list[Graph]  (library patterns, built via from_spec or fx_adapter)
   │
   │  matcher.find_all_matches()   ← Phase 2: enumerate candidate tiles
   ▼
list[Tile]  (each Tile = pattern_id + covered_nodes + mapping)
   │
   │  solver backtracking          ← Phase 3: optimize coverage + tile count
   ▼
TilingResult
   ├── .tiles           list of chosen Tile objects
   ├── .uncovered       frozenset of uncovered node IDs
   ├── .coverage        int
   ├── .fully_tiled     bool
   └── (feed to graph.to_dot_with_tiling() for visualization)
```

---

## Key classes

### `Graph` (graph.py)

The core data structure. A directed acyclic graph where:
- `nodes: dict[str, str]` — node ID → op type
- `inputs: dict[str, list[str]]` — node ID → ordered list of input node IDs
- `outputs: list[str]` — declared outputs (auto-detected as sink nodes if
  not set explicitly)

Construction:
```python
# From DSL
g = Graph.from_spec("1:mm->2:relu->3:add; 4:sigmoid->3:add")

# Programmatic
g = Graph()
g.add_node("1", "mm")
g.add_node("2", "relu")
g.add_edge("1", "2")

# From PyTorch
g = trace_to_graph(my_module)
```

### `Tile` (solver.py)

A single tile placement:
- `pattern_id: int` — which library pattern
- `covered_nodes: frozenset[str]` — big-graph nodes covered
- `mapping: dict[str, str]` — pattern node → big-graph node (for debugging)

### `TilingResult` (solver.py)

The solver's output:
- `tiles: list[Tile]`
- `uncovered: frozenset[str]`
- `coverage: int`
- `total_nodes: int`
- `fully_tiled: bool` (property)

---

## Visualization

The `Graph.to_dot_with_tiling(result)` method generates Graphviz DOT
format with:

- **Coloured clusters** for each tile (cycling through 10 pastel colours)
- **Dashed gray nodes** for uncovered nodes
- **All original edges** preserved

Render with:
```bash
dot -Tpng output.dot -o output.png
```

Or programmatically:
```python
dot = big.to_dot_with_tiling(result, title="My Tiling")
with open("output.dot", "w") as f:
    f.write(dot)
```

---

## Extending the tiler

### Adding new commutative ops

Edit `COMMUTATIVE_OPS` in `matcher.py`:
```python
COMMUTATIVE_OPS = frozenset({"add", "mul", "bitwise_or"})
```

### Adding new FX op mappings

Add entries to `_FUNC_CANONICAL`, `_METHOD_CANONICAL`, or
`_MODULE_CANONICAL` in `fx_adapter.py`.

### Supporting a new graph IR

Write an adapter function similar to `fx_to_graph()` that converts
your IR into a `Graph` object. The solver and matcher work unchanged.
