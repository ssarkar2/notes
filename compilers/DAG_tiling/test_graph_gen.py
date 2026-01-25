from graph_gen import parse_graph
import pytest


def helper(G):
    # Check nodes
    expected_nodes = {
        1: {'op': 'add'},
        2: {'op': 'mul'},
        3: {'op': 'mul'},
        4: {'op': 'gelu'}
    }
    for node_id, attrs in expected_nodes.items():
        assert G.nodes[node_id] == attrs

    # Check edges
    expected_edges = [(1, 2), (2, 4), (2, 3)]
    assert set(G.edges()) == set(expected_edges)

def test_parse_graph_semicolon():
    notation = "1_add->2_mul; 2_mul->4_gelu; 2_mul->3_mul"
    G = parse_graph(notation)

    helper(G)

def test_parse_graph_newline():
    notation = '''1_add->2_mul
    2_mul->4_gelu
    2_mul->3_mul'''
    G = parse_graph(notation)

    helper(G)
    

def test_parse_graph_mixed():
    notation = '''1_add->2_mul
    2_mul->4_gelu;2_mul->3_mul'''
    G = parse_graph(notation)

    helper(G)

def test_parse_graph_mixed_comment():
    notation = '''1_add->2_mul # hello
    2_mul->4_gelu;2_mul->3_mul'''
    G = parse_graph(notation)

    helper(G)

def test_parse_graph_chain():
    notation = '''1_add->2_mul->4_gelu;
    2_mul->3_mul'''
    G = parse_graph(notation)

    helper(G)

def test_bad_graph_redefinition():
    notation = '''1_add->2_mul # hello
    1_mul->4_gelu;2_mul->3_mul'''
    with pytest.raises(ValueError, match="Node 1 redefined: 'add' vs 'mul'"):
        parse_graph(notation)

    