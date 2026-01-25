from dag_tile_1 import tiler_greedy_maximal_munch
from dag_tile_2 import tiler_greedy_frontier_maximal_munch
from dag_tile_3 import tiler_backtracking_frontier_maximal_munch
from dag_tile_4 import tiler_backtracking_frontier_maximal_munch_cached
from dag_tile_5 import tiler_backtracking_frontier_maximal_munch_cached_withskip
from visualize import visualize_tiling
import networkx as nx

# --- Setup Example ---

# Create a Big DAG: A -> B -> C -> D
big = nx.DiGraph()
big.add_nodes_from([
    (1, {'op': 'add'}), (2, {'op': 'mul'}), 
    (3, {'op': 'add'}), (4, {'op': 'mul'})
])
big.add_edges_from([(1, 2), (2, 3), (3, 4)])

# Create Library: A small 'add -> mul' piece
lib_dag = nx.DiGraph()
lib_dag.add_nodes_from([('a', {'op': 'add'}), ('b', {'op': 'mul'})])
lib_dag.add_edges_from([('a', 'b')])

library = [lib_dag]


for algo in [tiler_greedy_maximal_munch, tiler_greedy_frontier_maximal_munch, tiler_backtracking_frontier_maximal_munch, tiler_backtracking_frontier_maximal_munch_cached, tiler_backtracking_frontier_maximal_munch_cached_withskip]:
    print(f"\n--- Running {algo.__name__} ---")
    # Run the Tiler
    result, missing = algo(big, library)
    visualize_tiling(big, library, result, missing, save_path=f"{algo.__name__}_result.png")

    if isinstance(result, str):
        print(result)
    else:
        print(f"Successfully tiled with {len(result)} pieces!")
        for i, match in enumerate(result):
            print(f"Piece {i+1}: Nodes {match['matched_nodes']}")