"""Example with partial coverage, competing patterns, and lots of search steps.

Run:  python example2.py
Debug output lands in debug_output2/.
"""

from graph import Graph
from solver import tile

# ── Big graph: ~25 nodes, branches, skips, uncoverable ops ──────
#
#   conv->bn->relu->conv->bn->relu->conv->bn->relu
#        skip: relu_1 ─────────────────┐
#                          relu_2 ─────┤
#                                      v
#                                     add_1
#                                      │
#                         ┌────────────┤
#                         v            v
#                   linear->relu->linear->add_2
#                                          │
#                  ┌───────────────────────┤
#                  v                       v
#             linear(q)──►matmul    linear(k)──►matmul
#                         │                     │
#                         softmax───────────►matmul
#                                              │
#                  linear(v)───────────────────►│
#                                              │
#                                    ┌─────────┘
#                                    v
#                               gather->scatter->custom
#                               softmax->log  (uncoverable)
#

big = Graph.from_spec(
    # 3x conv->bn->relu trunk
    "1:conv->2:bn->3:relu->4:conv->5:bn->6:relu->7:conv->8:bn->9:relu;"

    # skip connections into add
    "3->10:add; 6->10:add;"

    # FFN with residual
    "10->11:linear->12:relu->13:linear->140:log->14:add;"
    "10->14;"

    # attention-like Q/K/V
    "14->15:linear->18:matmul;"
    "14->16:linear->19:transpose->18;"
    "18->20:softmax->21:matmul;"
    "14->17:linear->21;"

    # uncoverable tail
    "21->22:gather->23:scatter->24:custom;"
    "20->25:log",

    outputs=["24", "25"],
)

print(f"Big graph: {len(big)} nodes")
print(f"Op types: {sorted(set(big.nodes.values()))}")


# ── Library: 6 patterns, mix of sizes, deliberate overlaps ──────

library = [
    # Large blocks (compete with each other and with singles)
    Graph.from_spec("1:conv->2:bn->3:relu",                          # pat 0 — matches 3x
                    outputs=["1", "2", "3"]),
    Graph.from_spec("1:conv->2:bn",                                  # pat 1 — overlaps with pat 0, matches 3x
                    outputs=["1", "2"]),
    Graph.from_spec("1:linear->2:relu->3:linear",                    # pat 2 — FFN, matches 1x
                    outputs=["1", "2", "3"]),
    Graph.from_spec("1:matmul->2:softmax->3:matmul",                 # pat 3 — attn core, matches 1x
                    outputs=["1", "2", "3"]),
    Graph.from_spec("1:linear->2:transpose->3:matmul",               # pat 4 — K projection, matches 1x
                    outputs=["1", "2", "3"]),

    # Singles (cover stragglers)
    Graph.from_spec("1:add"),                                        # pat 5
    Graph.from_spec("1:linear"),                                     # pat 6
    Graph.from_spec("1:relu"),                                       # pat 7
    Graph.from_spec("1:matmul"),                                     # pat 8
    Graph.from_spec("1:softmax"),                                    # pat 9
    Graph.from_spec("1:transpose"),                                  # pat 10

    # NOTE: no patterns for gather, scatter, custom, log — these stay uncovered.
]

print(f"Library: {len(library)} patterns")

result = tile(big, library, debug_dir="debug_output2")

status = "FULL" if result.fully_tiled else "PARTIAL"
print(f"\n[{status}] coverage={result.coverage}/{result.total_nodes}  tiles={len(result.tiles)}")
for i, t in enumerate(result.tiles):
    labels = ", ".join(f"{n}:{big.nodes[n]}" for n in sorted(t.covered_nodes))
    print(f"  tile {i}: pat={t.pattern_id}  {{{labels}}}")
uncov = ", ".join(f"{n}:{big.nodes[n]}" for n in sorted(result.uncovered))
print(f"  uncovered: {{{uncov}}}")

import os
steps_dir = os.path.join("debug_output2", "steps")
n_steps = len([f for f in os.listdir(steps_dir) if f.endswith(".png")])
print(f"\nDebug: {n_steps} step snapshots in debug_output2/steps/")
