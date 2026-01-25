import networkx as nx
import re


"""
Parses a custom graph notation string and returns a directed graph.

Each edge in the notation should be of the form:
    <src_id>_<src_op>-><dst_id>_<dst_op>
Multiple edges are separated by semicolons or newlines.

Example:
    notation = "0_input->1_conv; 1_conv->2_relu; 2_relu->3_fc"
    G = parse_graph(notation)

Returns:
    networkx.DiGraph: Directed graph with nodes labeled by id and op attribute.
"""

def parse_graph(notation):
    G = nx.DiGraph()
    edges = []
    nodes = {}

    # Split on semicolons or newlines
    edge_strs = re.split(r'[;\n]+', notation.strip())
    for edge_str in edge_strs:
        # Remove comments
        edge_str = edge_str.split('#', 1)[0].strip()
        if not edge_str:
            continue
        # Support chains: 1_a->2_b->3_c
        chain = [s.strip() for s in edge_str.split('->')]
        node_ids_ops = []
        for node in chain:
            node_id, node_op = node.split('_', 1)
            # Check for redefinition
            if node_id in nodes and nodes[node_id] != node_op:
                raise ValueError(f"Node {node_id} redefined: '{nodes[node_id]}' vs '{node_op}'")
            nodes[node_id] = node_op
            node_ids_ops.append((int(node_id), node_op))
        # Add edges for the chain
        for i in range(len(node_ids_ops) - 1):
            edges.append((node_ids_ops[i][0], node_ids_ops[i+1][0]))

    # Add nodes with op attribute
    for node_id, op in nodes.items():
        G.add_node(int(node_id), op=op)
    # Add edges
    G.add_edges_from(edges)
    return G


# Example Attention-like network graph notation (simplified Transformer encoder block)
# input -> linear_q, linear_k, linear_v -> attention -> add -> norm1 -> linear1 -> relu -> linear2 -> add -> norm2
attention_notation = (
    "0_input->1_linear_q; "
    "0_input->2_linear_k; "
    "0_input->3_linear_v; "
    "1_linear_q->4_attention; "
    "2_linear_k->4_attention; "
    "3_linear_v->4_attention; "
    "4_attention->5_add; "
    "0_input->5_add; "  # skip connection
    "5_add->6_norm1; "
    "6_norm1->7_linear1; "
    "7_linear1->8_relu; "
    "8_relu->9_linear2; "
    "9_linear2->10_add; "
    "6_norm1->10_add; "  # skip connection
    "10_add->11_norm2"
)

# Example Attention-like network graph notation (simplified Transformer encoder block)
# input -> linear_q, linear_k, linear_v -> attention -> add -> norm1 -> linear1 -> relu -> linear2 -> add -> norm2
attention_notation = (
    "0_input->1_linear; "
    "0_input->2_linear; "
    "0_input->3_linear; "
    "1_linear->4_matmul; "
    "2_linear->4_matmul; "
    "4_matmul->5_softmax; "
    "5_softmax->6_matmul; "
    "3_linear->6_matmul; "
    "6_matmul->7_add; "
    "0_input->7_add; "  # skip connection
    "7_add->8_layernorm; "
    "8_layernorm->9_linear; "
    "9_linear->10_relu; "
    "10_relu->11_linear; "
    "11_linear->12_add; "
    "8_layernorm->12_add; "  # skip connection
    "12_add->13_layernorm"
)
# Example BERT-like network graph notation for 2 encoder layers
# Each encoder layer: input -> linear_q, linear_k, linear_v -> matmul1 (QK^T) -> softmax -> matmul2 (attn*V) -> add (skip) -> layernorm -> linear1 -> relu -> linear2 -> add (skip) -> layernorm

bert_notation = (
    # Encoder Layer 1
    "0_input->1_linear; "
    "0_input->2_linear; "
    "0_input->3_linear; "
    "1_linear->4_matmul; "
    "2_linear->4_matmul; "
    "4_matmul->5_softmax; "
    "5_softmax->6_matmul; "
    "3_linear->6_matmul; "
    "6_matmul->7_add; "
    "0_input->7_add; "  # skip connection
    "7_add->8_layernorm; "
    "8_layernorm->9_linear; "
    "9_linear->10_relu; "
    "10_relu->11_linear; "
    "11_linear->12_add; "
    "8_layernorm->12_add; "  # skip connection
    "12_add->13_layernorm; "

    # Encoder Layer 2
    "13_layernorm->14_linear; "
    "13_layernorm->15_linear; "
    "13_layernorm->16_linear; "
    "14_linear->17_matmul; "
    "15_linear->17_matmul; "
    "17_matmul->18_softmax; "
    "18_softmax->19_matmul; "
    "16_linear->19_matmul; "
    "19_matmul->20_add; "
    "13_layernorm->20_add; "  # skip connection
    "20_add->21_layernorm; "
    "21_layernorm->22_linear; "
    "22_linear->23_relu; "
    "23_relu->24_linear; "
    "24_linear->25_add; "
    "21_layernorm->25_add; "  # skip connection
    "25_add->26_layernorm; "

    # pooler
    "26_layernorm->27_slice; "
    "27_slice->28_linear; " 
    "28_linear->29_tanh;"

    # classifier
    "29_tanh->30_linear;"
    "30_linear->31_softmax"
)


if __name__ == "__main__":
    vgg_graph = parse_graph(vgg_notation)
    resnet18_graph = parse_graph(resnet18_notation)
    attention_graph = parse_graph(attention_notation)
    bert_graph = parse_graph(bert_notation)
    import matplotlib.pyplot as plt


    # Draw DNN with node labels as op names
    def draw_dnn(G, filename="model.png"):
        plt.figure(figsize=(12, 18))  # DNNs are usually tall/long

        # Use 'dot' layout (hierarchical)
        pos = nx.nx_pydot.graphviz_layout(G, prog='dot')

        # Use op attribute as label
        labels = {n: G.nodes[n]['op'] for n in G.nodes}

        nx.draw(G, pos,
                labels=labels,
                with_labels=True,
                node_size=1000,
                node_color='skyblue',
                font_size=8,
                arrows=True,
                arrowsize=15)

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()  # Do not display, just save

    draw_dnn(vgg_graph, "vgg_graph.png")
    draw_dnn(resnet18_graph, "resnet18_graph.png")
    draw_dnn(attention_graph, "attention_graph.png")
    draw_dnn(bert_graph, "bert_graph.png")