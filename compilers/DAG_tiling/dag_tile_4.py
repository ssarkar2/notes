
import networkx as nx
from dag_tile_2 import real_rooted_match


"""
tiler_backtracking_frontier_maximal_munch_with_skipping_memoized

This algorithm performs an exhaustive search to find the optimal tiling of a DAG
using a library of patterns. It is specifically optimized for deep, repetitive 
structures like DNNs (ResNet, Transformers), because it uses memoization

Key Algorithmic Features:
1. Frontier-Based Search: Processes nodes in a topological order to ensure 
   dependencies are respected and tiling flows from inputs to outputs.
2. Value-Returning Recursion: Unlike standard backtracking that returns True/False,
   each call returns the actual BEST tiling found for the remaining subgraph.
3. Subgraph Memoization: Uses a (node_index, covered_set) key to cache results.
   If the search reaches a previously seen "remaining problem," it instantly 
   returns the cached best solution. This is critical for DNNs where identical 
   blocks (e.g., Transformer layers) would otherwise be re-tiled millions of times.
4. Optimal Greedy Fallback: Prioritizes "Maximal Munching" (larger tiles), but
   efficiently falls back to skipping nodes if that leads to a higher overall 
   node coverage score.
5. Early Exit: If a branch achieves 100% coverage of the remaining nodes, it 
   terminates search for that sub-problem immediately, as no better result 
   is mathematically possible.

Input: 
    big_dag: The target DAG to be tiled.
    library: A list of DAG patterns to match.
    allow_skip: If False, the algo only returns solutions with 100% coverage.
    
Returns:
    final_sol: List of dictionaries containing matched library pieces and nodes.
    missing_nodes: List of nodes that could not be covered by the library.
"""

def tiler_backtracking_frontier_maximal_munch_with_skipping_memoized(big_dag, library, allow_skip=True):
    sorted_library = sorted(library, key=lambda g: g.number_of_nodes(), reverse=True)
    topo_order = list(nx.topological_sort(big_dag))
    all_nodes = set(big_dag.nodes())

    # Memo stores: (order_index, frozenset_covered) -> (best_sub_solution, total_nodes_covered)
    memo = {}

    def solve(covered_nodes, order_index):
        # 1. Check Memo
        state_key = (order_index, frozenset(covered_nodes))
        if state_key in memo:
            return memo[state_key]

        # 2. Base Case: Reached the end
        if order_index >= len(topo_order):
            return [], 0

        target_node = topo_order[order_index]

        # 3. Skip if already covered by a previous large tile
        if target_node in covered_nodes:
            return solve(covered_nodes, order_index + 1)

        best_sub_sol = []
        max_sub_count = 0

        # 4. Option A: Try all library matches
        for lib_dag in sorted_library:
            matches = real_rooted_match(big_dag, lib_dag, target_node)
            for match in matches:
                matched_set = set(match.keys())
                if not matched_set.intersection(covered_nodes):
                    # Recurse
                    sub_sol, sub_count = solve(covered_nodes | matched_set, order_index + 1)
                    
                    current_total = len(matched_set) + sub_count
                    if current_total > max_sub_count:
                        max_sub_count = current_total
                        best_sub_sol = [{"library_dag": lib_dag, "matched_nodes": matched_set}] + sub_sol
                        
                        # Optimization: If we hit 100% coverage for the remaining graph, stop searching
                        if max_sub_count == (len(all_nodes) - len(covered_nodes)):
                            memo[state_key] = (best_sub_sol, max_sub_count)
                            return best_sub_sol, max_sub_count

        # 5. Option B: Fallback Skip (only if 100% cover wasn't found above)
        if allow_skip:
            skip_sol, skip_count = solve(covered_nodes | {target_node}, order_index + 1)
            if skip_count > max_sub_count:
                max_sub_count = skip_count
                best_sub_sol = skip_sol

        # 6. Store in Memo and Return
        memo[state_key] = (best_sub_sol, max_sub_count)
        return best_sub_sol, max_sub_count

    # Initial call
    final_sol, final_count = solve(set(), 0)
    
    missing_nodes = list(all_nodes - {n for t in final_sol for n in t['matched_nodes']})
    return final_sol, missing_nodes