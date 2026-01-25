
import networkx as nx
from dag_tile_2 import real_rooted_match


"""
tiler_backtracking_frontier_maximal_munch_with_skipping: Maximum munch on frontier with skipping and backtracking


Logic:
1. Frontier Search: It follows the topological order of the DAG, ensuring
   it tiles from inputs toward outputs.
2. Maximal Munching: It prioritizes larger library pieces first
3. Recursive Backtracking: Instead of committing to a match like a greedy 
   tiler, it "tries" a match. If that choice makes it impossible to tile 
   the rest of the graph, it undos (backtracks) and tries a different piece.
4. The Skip Fallback: If no library piece fits 'target_node', or if existing 
   matches lead to dead ends, the algorithm "skips" the node. This treats the 
   node as a missing piece but allows the tiler to continue and finish 
   tiling the remainder of the graph.
5. Best-State Tracking: It uses a global 'best_state' to remember the most 
   complete tiling found during the entire search, ensuring it returns the 
   highest node coverage even if a 100% perfect cover is unreachable.
"""

def tiler_backtracking_frontier_maximal_munch_with_skipping(big_dag, library, allow_skip=True):
    sorted_library = sorted(library, key=lambda g: g.number_of_nodes(), reverse=True)
    topo_order = list(nx.topological_sort(big_dag))
    all_nodes = set(big_dag.nodes())

    # We use a nonlocal variable to keep track of the "Best" solution found
    # even if we can't get a 100% cover.
    best_state = {"sol": [], "missing": list(all_nodes), "count": 0}

    def solve(covered_nodes, current_solution, order_index):
        # Calculate how many nodes we've actually covered with tiles
        actual_covered_count = sum(len(t['matched_nodes']) for t in current_solution)
        
        # Update our "Best So Far" record
        if actual_covered_count > best_state["count"]:
            best_state["count"] = actual_covered_count
            best_state["sol"] = list(current_solution)
            best_state["missing"] = list(all_nodes - {n for t in current_solution for n in t['matched_nodes']})

        # Base case: reached the end of the graph
        if order_index >= len(topo_order):
            return actual_covered_count == len(all_nodes)

        target_node = topo_order[order_index]

        # If already covered by a previous tile, just move to the next index
        if target_node in covered_nodes:
            return solve(covered_nodes, current_solution, order_index + 1)

        # 1. Attempt to cover target_node with library pieces (Highest Priority)
        for lib_dag in sorted_library:
            matches = real_rooted_match(big_dag, lib_dag, target_node)
            for match in matches:
                matched_set = set(match.keys())
                if not matched_set.intersection(covered_nodes):
                    if solve(covered_nodes | matched_set, 
                             current_solution + [{"library_dag": lib_dag, "matched_nodes": matched_set}], 
                             order_index + 1):
                        return True

        if allow_skip:
            # 2. Fallback: Skip target_node (Lowest Priority)
            # This only executes if no library match (or orientation) was able to 
            # complete the tiling for the rest of the graph.
            return solve(covered_nodes | {target_node}, current_solution, order_index + 1)
        else:
            # We are not allowing skipping, so just return False here and end the recursion
            return False

    perfect_match = solve(set(), [], 0)

    assert (best_state['missing'] == []) == perfect_match == (best_state['count'] == len(all_nodes)) == (set().union(*[i['matched_nodes'] for i in best_state['sol']]) == all_nodes)


    # Returns the best possible solution it found, and the nodes it had to skip
    return best_state["sol"], best_state["missing"]



