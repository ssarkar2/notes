
import networkx as nx
from networkx.algorithms import isomorphism

'''
Greedy Maximal Munch DAG Tiling Algorithm

Loops over the library from largest to smallest DAG
Tries to find non-overlapping matches in the big DAG

'''


def tiler_greedy_maximal_munch(big_dag, library):
    # 1. Sort library by size (number of nodes) descending
    # We want to use the "largest" pieces first to maximize efficiency.
    sorted_library = sorted(library, key=lambda g: g.number_of_nodes(), reverse=True)
    
    nodes_to_cover = set(big_dag.nodes())
    tiling_solution = []
    
    # We define a node_match to ensure 'op' types match (e.g., 'add' matches 'add')
    def node_match(n1, n2):
        return n1.get('op') == n2.get('op')

    # 2. Iterate through library and try to find matches
    for small_dag in sorted_library:
        matcher = isomorphism.DiGraphMatcher(big_dag, small_dag, node_match=node_match)
        
        # Look for all instances of this small_dag
        for match in matcher.subgraph_isomorphisms_iter():
            # match is a dict: {big_node: small_node}
            matched_nodes = set(match.keys())
            
            # 3. Check if these nodes are already covered (no overlaps allowed)
            if matched_nodes.issubset(nodes_to_cover):
                # Consume these nodes
                nodes_to_cover -= matched_nodes
                tiling_solution.append({
                    "library_dag": small_dag,
                    "matched_nodes": matched_nodes
                })
                
                if not nodes_to_cover:
                    break
        if not nodes_to_cover:
            break

    # 4. Final Report
    missing = list(nodes_to_cover)
    
    return tiling_solution, missing
