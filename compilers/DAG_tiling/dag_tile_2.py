import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import isomorphism



'''
tiler_greedy_frontier: A DAG tiling algorithm that processes nodes in topological order,
attempting to cover the DAG from sources to sinks using the largest library pieces first.
'''

def node_match(n1, n2):
    return n1.get('op') == n2.get('op')


'''
This is "fake" because we are running full isomorphism.DiGraphMatcher, which is costly. its full match followed by filtering

real_rooted_match below is the efficient version of fake_rooted_match. We keep fake_rooted_match to 
explain the logic in simpler terms.
'''
def fake_rooted_match(big_dag, lib_dag, root_node):
    matcher = isomorphism.DiGraphMatcher(big_dag, lib_dag, node_match=node_match)

    # We only care about this match if it covers our current 'node'
    return [match for match in matcher.subgraph_isomorphisms_iter() if root_node in match]


'''
Performs a high-performance, anchored subgraph isomorphism check.

Efficiency:
- Unlike global matchers, it starts specifically at target_node and only 
  explores its local neighborhood.
- Short-circuits immediately if the root operation doesn't match.

Inputs:
- big_dag: The main computation graph.
- lib_dag: The pattern to match.
- target_node: The node in big_dag that must act as the root for this match.

Returns:
- A list containing a mapping dictionary {big_node: lib_node} if successful.
- An empty list [] if no match is found.
'''
def real_rooted_match(big_dag, lib_dag, target_node):
    """
    Efficiently finds if lib_dag can be rooted at target_node.
    Returns: A list containing the mapping dictionary, or an empty list.
    Format: [{big_node: lib_node, ...}] or []
    """
    # 1. Identify the 'root' of the library pattern
    # Using a list because we only need the first source node
    lib_topo = list(nx.topological_sort(lib_dag))
    if not lib_topo:
        return []
    lib_root = lib_topo[0]

    def _match_recursive(b_node, l_node, current_mapping):
        # Semantic check
        if big_dag.nodes[b_node].get('op') != lib_dag.nodes[l_node].get('op'):
            return None

        # Tentative mapping for this branch
        new_mapping = current_mapping.copy()
        new_mapping[b_node] = l_node

        # Structural check: All library children must find a unique big_dag child
        for l_child in lib_dag.successors(l_node):
            found_child_match = False
            for b_child in big_dag.successors(b_node):
                # Ensure we don't reuse a node already mapped in this pattern
                if b_child in new_mapping:
                    continue
                
                # Recursive call to check the next level of the pattern
                res = _match_recursive(b_child, l_child, new_mapping)
                if res:
                    new_mapping.update(res)
                    found_child_match = True
                    break
            
            if not found_child_match:
                return None
                
        return new_mapping

    # 2. Execute the recursive match starting from the anchor
    result_map = _match_recursive(target_node, lib_root, {})

    # 3. Return in the same format as fake_rooted_match (a list of matches)
    return [result_map] if result_map is not None else []




def tiler_greedy_frontier_maximal_munch(big_dag, library, rooted_search=real_rooted_match):
    # Sort library by size so we "munch" the biggest pieces first
    sorted_library = sorted(library, key=lambda g: g.number_of_nodes(), reverse=True)
    
    covered_nodes = set()
    tiling_solution = []
    
    # Process nodes in topological order (source to sink)
    # This ensures we satisfy dependencies as we go
    order = list(nx.topological_sort(big_dag))
    
    for node in order:
        if node in covered_nodes:
            continue
            
        found_match = False
        for lib_dag in sorted_library:
            # We look for a subgraph isomorphism, but do a "rooted" match on 'node'
            matches = rooted_search(big_dag, lib_dag, node)

            # Find all matches for this library piece
            for match in matches:
                if not set(match.keys()).intersection(covered_nodes):
                    # Found a valid, non-overlapping fit for this piece!
                    matched_set = set(match.keys())
                    tiling_solution.append({
                        "library_dag": lib_dag,
                        "matched_nodes": matched_set
                    })
                    covered_nodes.update(matched_set)
                    found_match = True
                    break # Break out of isomorphisms
            if found_match:
                break # Move to next uncovered node in topological order
                
    missing = set(big_dag.nodes()) - covered_nodes
    return tiling_solution, list(missing)