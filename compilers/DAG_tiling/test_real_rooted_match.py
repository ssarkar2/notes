from graph_gen import parse_graph


from dag_tile_2 import real_rooted_match
import networkx as nx


'''
test real_rooted_match
'''

def test_real_rooted_match_simple_match():
    big_dag = parse_graph('1_add->2_mul')
    lib_dag = parse_graph('10_add->11_mul')
    matches = real_rooted_match(big_dag, lib_dag, 1)
    assert matches
    mapping = matches[0]
    assert mapping[1] == 10
    assert mapping[2] == 11

def test_real_rooted_match_no_match_wrong_op():
    big_dag = parse_graph('1_add->2_mul')
    lib_dag = parse_graph('10_mul->11_add')
    matches = real_rooted_match(big_dag, lib_dag, 1)
    assert matches == []

def test_real_rooted_match_partial_match_fails():
    big_dag = parse_graph('1_add->2_mul; 1_add->3_mul')
    lib_dag = parse_graph('10_add->11_mul; 11_mul->12_gelu')
    matches = real_rooted_match(big_dag, lib_dag, 1)
    assert matches == []

def test_real_rooted_match_multiple_children():
    big_dag = parse_graph('1_add->2_mul; 1_add->3_mul')
    lib_dag = parse_graph('10_add->11_mul; 10_add->12_mul')
    matches = real_rooted_match(big_dag, lib_dag, 1)
    assert matches
    mapping = matches[0]
    assert mapping[1] == 10
    assert set([mapping[2], mapping[3]]) == {11, 12}

def test_real_rooted_match_non_root_anchor():
    big_dag = parse_graph('1_add->2_mul; 2_mul->3_gelu')
    lib_dag = parse_graph('10_mul->11_gelu')
    matches = real_rooted_match(big_dag, lib_dag, 2)
    assert matches
    mapping = matches[0]
    assert mapping[2] == 10
    assert mapping[3] == 11
    matches = real_rooted_match(big_dag, lib_dag, 1)
    assert matches == []

def test_real_rooted_match_empty_lib():
    big_dag = parse_graph('1_add->2_mul')
    lib_dag = nx.DiGraph()
    matches = real_rooted_match(big_dag, lib_dag, 1)
    assert matches == []