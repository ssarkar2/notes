"""Minimal example that forces the solver to backtrack.

Run:  python example3.py
Debug output lands in debug_output3/.

The graph (8 nodes):

    1:conv -> 2:bn -> 3:relu -> 4:conv -> 5:bn -> 6:relu -> 7:custom
                      3:relu -> 8:pool

Library:
    pat 0: conv -> bn -> relu   (3 nodes)
    pat 1: relu -> pool         (2 nodes)
    pat 2: conv -> bn           (2 nodes)

Candidates found by the matcher:
    C0: pat 0  {1,2,3}   conv->bn->relu
    C1: pat 0  {4,5,6}   conv->bn->relu
    C2: pat 1  {3,8}     relu->pool
    C3: pat 2  {1,2}     conv->bn
    C4: pat 2  {4,5}     conv->bn

WHY BACKTRACKING IS NEEDED:

    C0 and C2 overlap on node 3:relu — picking one blocks the other.

    Greedy (largest first) picks C0 {1,2,3}, then C1 {4,5,6}.
    Coverage = 6/8.  Node 8:pool left uncovered.

    Optimal: C1 {4,5,6} + C2 {3,8} + C3 {1,2}.
    Coverage = 7/8.  Only 7:custom (unsupported) is uncovered.

    The solver must backtrack from C0 to discover that the smaller
    C3 + C2 combination covers one more node overall.
"""

from graph import Graph
from solver import tile

# ── Big graph ──

big = Graph.from_spec(
    "1:conv->2:bn->3:relu->4:conv->5:bn->6:relu->7:custom;"
    "3->8:pool",
    outputs=["7"],
)

print(f"Big graph: {len(big)} nodes")
print(f"Nodes: {big.nodes}")
print()

# ── Library ──

library = [
    Graph.from_spec("1:conv->2:bn->3:relu",    # pat 0 (3 nodes)
                    outputs=["1", "2", "3"]),
    Graph.from_spec("1:relu->2:pool",           # pat 1 (2 nodes)
                    outputs=["1", "2"]),
    Graph.from_spec("1:conv->2:bn",             # pat 2 (2 nodes)
                    outputs=["1", "2"]),
]

# ── Tile with debug output ──

result = tile(big, library, debug_dir="debug_output3")

status = "FULL" if result.fully_tiled else "PARTIAL"
print(f"[{status}] coverage={result.coverage}/{result.total_nodes}  tiles={len(result.tiles)}")
for i, t in enumerate(result.tiles):
    labels = ", ".join(f"{n}:{big.nodes[n]}" for n in sorted(t.covered_nodes))
    print(f"  tile {i}: pat={t.pattern_id}  {{{labels}}}")
uncov = ", ".join(f"{n}:{big.nodes[n]}" for n in sorted(result.uncovered))
print(f"  uncovered: {{{uncov}}}")

import os
steps_dir = os.path.join("debug_output3", "steps")
n_steps = len([f for f in os.listdir(steps_dir) if f.endswith(".svg")])
print(f"\nDebug: {n_steps} step snapshots in debug_output3/steps/")
print("Check debug_output3/search.txt for the full backtracking trace.")
