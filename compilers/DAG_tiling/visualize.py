
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import isomorphism
import random

def visualize_tiling(big_dag, library, tiling_results, missing_nodes, save_path="dag_tiling_result.png"):
    # 1. Create a figure with subplots: one for the library, one for the big DAG
    num_lib_items = len(library)
    fig = plt.figure(figsize=(12, 8))
    
    # Subplot for Library Components
    for i, lib_dag in enumerate(library):
        ax = fig.add_subplot(2, max(num_lib_items, 2), i + 1)
        pos = nx.spring_layout(lib_dag)
        labels = nx.get_node_attributes(lib_dag, 'op')
        nx.draw(lib_dag, pos, with_labels=True, labels=labels, node_color='lightblue', 
                node_size=800, font_size=10, ax=ax)
        ax.set_title(f"Library Piece {i+1}")

    # 2. Assign colors to each tile found
    # Generate a random color for each successful tile
    color_map = []
    node_to_color = {}
    
    for i, tile in enumerate(tiling_results):
        color = "#%06x" % random.randint(0, 0xFFFFFF)
        for node in tile['matched_nodes']:
            node_to_color[node] = color

    # Default color for untiled nodes
    for node in big_dag.nodes():
        if node not in node_to_color:
            node_to_color[node] = '#A0A0A0' # Gray

    # 3. Plot the Big DAG with Tiling Colors
    ax_big = fig.add_subplot(2, 1, 2)
    pos_big = nx.shell_layout(big_dag) # Shell or Spring layout
    
    colors = [node_to_color[n] for n in big_dag.nodes()]
    labels_big = {n: f"{n}\n({big_dag.nodes[n].get('op')})" for n in big_dag.nodes()}
    
    nx.draw(big_dag, pos_big, with_labels=True, labels=labels_big, 
            node_color=colors, node_size=1500, font_size=8, ax=ax_big)
    
    status = "SUCCESS" if not missing_nodes else f"FAILED (Missing: {missing_nodes})"
    ax_big.set_title(f"Big DAG Tiling Result - {status}")
    
    plt.tight_layout()
    #plt.show()

    plt.savefig(save_path)
    plt.close(fig)