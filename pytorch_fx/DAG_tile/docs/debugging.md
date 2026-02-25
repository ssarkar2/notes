# Debug Mode

The tiler has a built-in debug mode that records every decision the solver
makes and produces DOT/PNG visualizations at each step.  Enable it by passing
`debug_dir` to `tile()`:

```python
from solver import tile

result = tile(big, library, debug_dir="my_debug")
```

When `debug_dir` is `None` (the default), a no-op logger is used and there is
zero overhead.

---

## Directory layout

After tiling completes, the debug directory contains:

```
my_debug/
  input_graph.dot        # DOT source for the input graph
  input_graph.png        # rendered PNG (requires Graphviz `dot`)
  candidates.txt         # every candidate tile placement found
  search.txt             # backtracking search trace
  result.txt             # final coverage summary
  tiled_graph.dot        # DOT source for the tiled graph (colored clusters)
  tiled_graph.png        # rendered PNG
  candidates/            # one DOT/PNG per candidate placement (on the big graph)
    cand_0000.dot
    cand_0000.png
    cand_0001.dot
    cand_0001.png
    ...
  library/               # one DOT/PNG pair per pattern in the library
    pat_00.dot
    pat_00.png
    ...
  steps/                 # one DOT/PNG snapshot per search step
    step_0001_try.dot
    step_0001_try.png
    step_0002_best.dot
    step_0002_best.png
    ...
```

If `dot` (Graphviz) is not on your `PATH`, the `.dot` files are still written
but the `.png` files are skipped.

---

## Output files explained

### `input_graph.dot / .png`

A visualization of the input graph before any tiling happens.  Useful as a
reference when reading the search trace.

### `candidates.txt`

Lists every valid tile placement the solver found during Phase 1 (candidate
enumeration).  Each line shows:

- which library pattern was matched (by index)
- the pattern's op sequence
- the set of input-graph nodes the placement covers

Example:

```
candidate 0: pattern 0 [1:linear -> 2:sigmoid -> 5:mul -> ...]  =>  {add:add, block1_down_proj:linear, ...}
candidate 1: pattern 0 [1:linear -> 2:sigmoid -> 5:mul -> ...]  =>  {add_1:add, block2_down_proj:linear, ...}

Total candidate placements: 2
```

Each candidate number corresponds to a visualization in `candidates/`
(e.g. `cand_0000.png`).

### `candidates/`

One DOT/PNG pair per candidate placement.  Each image shows the **full input
graph** with the candidate's covered nodes highlighted in a colored cluster
and the remaining nodes in grey.  This lets you visually verify where each
pattern matched on the big graph — useful for spotting unexpected or missing
matches before looking at the solver's search trace.

### `search.txt`

A full trace of the backtracking search (Phase 2).  The indentation reflects
recursion depth.  Key entries:

| Entry | Meaning |
|---|---|
| `pick node X:op (N covering tiles, M uncovered remain)` | The solver chose node X as the next branching point. N tiles cover it; M nodes are still uncovered. |
| `try tile pat=P {nodes...} [step K]` | The solver is trying tile placement P. Step K links to the snapshot in `steps/`. |
| `skip X:op` | The solver explores the branch where node X is left uncovered. |
| `-> new best: coverage=C, tiles=T [step K]` | A new best solution was found with coverage C using T tiles. |
| `memo hit (uncovered size=S)` | A memoized result was reused for a subproblem of size S. |
| `pruned (full sub-coverage)` | The remaining subproblem was fully covered, so no further branching is needed. |

The `[step K]` labels correspond to files in `steps/` (e.g. `step_0003_try.png`),
so you can visually follow the solver's decisions.

### `steps/`

Contains a DOT/PNG snapshot for every `[step K]` entry in `search.txt`.

- **`step_NNNN_try`** — shows the graph right after the solver *tries* a tile.
  Covered nodes are colored; uncovered nodes are grey.
- **`step_NNNN_best`** — shows the graph when a new best solution is recorded.

Stepping through these in order gives an animated view of the backtracking
search.

### `library/`

One DOT/PNG pair per pattern in the library (`pat_00`, `pat_01`, ...).
Each visualization labels the pattern's nodes with their ops so you can
cross-reference with the pattern indices in `candidates.txt` and `search.txt`.

### `result.txt`

A summary of the final tiling:

```
[FULL] coverage=14/14  tiles=2

  tile 0: pattern=0  {add_1:add, block2_down_proj:linear, ...}
  tile 1: pattern=0  {add:add, block1_down_proj:linear, ...}
```

- **FULL** means every node is covered; **PARTIAL** means some nodes are
  uncovered (listed at the bottom).
- Each tile line shows the library pattern index and the covered node set.

### `tiled_graph.dot / .png`

The final visualization with each tile drawn as a colored cluster.  Uncovered
nodes (if any) appear without a cluster border.

---

## Typical workflow

1. **Run with debug enabled:**
   ```python
   result = tile(big, library, debug_dir="debug_output")
   ```

2. **Check `result.txt`** — did the solver achieve full coverage?  If not,
   which nodes are uncovered?

3. **Open `tiled_graph.png`** — visually verify the tile boundaries make sense.

4. **If something looks wrong**, open `search.txt` and follow the step labels.
   For each `[step K]`, open the corresponding `steps/step_KKKK_*.png` to see
   exactly what the solver was considering at that point.

5. **Check `candidates.txt`** and **`candidates/`** — if a pattern you expected
   to match is missing, the issue is in Phase 1 (candidate enumeration /
   pattern matching), not the search.  Open `cand_NNNN.png` to see exactly
   which nodes each candidate covers on the big graph.
