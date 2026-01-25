from dag_tile_1 import tiler_greedy_maximal_munch
from dag_tile_2 import tiler_greedy_frontier_maximal_munch, fake_rooted_match
from dag_tile_3 import tiler_backtracking_frontier_maximal_munch_with_skipping
from dag_tile_4 import tiler_backtracking_frontier_maximal_munch_with_skipping_memoized
import pytest
from graph_gen import parse_graph


'''
This test shows tiler_greedy_frontier_maximal_munch is faster than tiler_greedy_maximal_munch for large graphs
'''

def large_linear_graph(num_nodes, library_piece_size):
    def helper(n):
        str = ''
        for i in range(0, n-1):
            str += f"{i}_op->{i+1}_op; "
        return str

    return parse_graph(helper(num_nodes)), [parse_graph(helper(library_piece_size))]

def test_dt1_dt2_performance():
    big_dag, library = large_linear_graph(200, 5)

    import time

    start = time.time()
    result1, missing1 = tiler_greedy_maximal_munch(big_dag, library)
    duration1 = time.time() - start
    print(f"tiler_greedy_maximal_munch took {duration1:.4f} seconds with {len(result1)} pieces.")

    start = time.time()
    result2, missing2 = tiler_greedy_frontier_maximal_munch(big_dag, library)
    assert missing2 == [] #note missing1 can be non empty. I cant emphasize enough how bad tiler_greedy_maximal_munch could be
    duration2 = time.time() - start
    print(f"tiler_greedy_frontier_maximal_munch took {duration2:.4f} seconds with {len(result2)} pieces.")


    start = time.time()
    result3, missing3 = tiler_greedy_frontier_maximal_munch(big_dag, library, fake_rooted_match)
    assert missing3 == []
    duration3 = time.time() - start
    print(f"tiler_greedy_frontier_maximal_munch with fake_rooted_match took {duration3:.4f} seconds with {len(result3)} pieces.")

    assert duration2 < duration1, "tiler_greedy_frontier_maximal_munch should be faster than tiler_greedy_maximal_munch"
    assert duration2 < duration3, "tiler_greedy_frontier_maximal_munch with real rooted match should be faster than with fake rooted match"
    assert duration1 < duration3, "tiler_greedy_maximal_munch should be faster than tiler_greedy_frontier_maximal_munch with fake rooted match"
    # because one is a loop over the library and the other is a double loop (nodes in big DAG x library)




def test_tile_1():
    big_dag = parse_graph('1_a->2_b->3_c->4_d')
    lib_dag1 = parse_graph('10_a->11_b')
    lib_dag2 = parse_graph('20_c->21_d')
    library = [lib_dag1, lib_dag2]

    for tiler in [tiler_greedy_maximal_munch, tiler_greedy_frontier_maximal_munch]:
        result, missing = tiler(big_dag, library)
        assert missing == []
        assert len(result) == 2  # Expecting 2 pieces to cover the big DAG

        covered_nodes = set()
        for piece in result:
            covered_nodes.update(piece['matched_nodes'])
        assert covered_nodes == set(big_dag.nodes()), "All nodes should be covered"

def test_tile_2_greedy_fail_backtrack_pass():
    big_dag = parse_graph('0_A->1_B->2_C->3_D')

    lib_dag1 = parse_graph('0_A->1_B')
    lib_dag2 = parse_graph('0_A->1_B->2_C')
    lib_dag3 = parse_graph('2_C->3_D')
    library = [lib_dag1, lib_dag2, lib_dag3]

    for tiler in [tiler_greedy_maximal_munch, tiler_greedy_frontier_maximal_munch]:
        result, missing = tiler(big_dag, library)
        assert missing == [3] # Greedy should fail to cover all nodes
        assert len(result) == 1
        assert result[0]['matched_nodes'] == {0, 1, 2}
    

    for tiler in [tiler_backtracking_frontier_maximal_munch_with_skipping, tiler_backtracking_frontier_maximal_munch_with_skipping_memoized]:
        result_bt, missing_bt = tiler(big_dag, library, allow_skip=False)
        assert missing_bt == []  # Backtracking should succeed
        assert len(result_bt) == 2
        assert result_bt[0]['matched_nodes'].union(result_bt[1]['matched_nodes']) == set(big_dag.nodes())

def test_tile_2_backtrack_fail_but_skipbacktrack_pass():
    big_dag = parse_graph('0_A->1_B->2_C->3_D')

    lib_dag1 = parse_graph('1_B->2_C->3_D')
    library = [lib_dag1]

    for tiler in [tiler_backtracking_frontier_maximal_munch_with_skipping, tiler_backtracking_frontier_maximal_munch_with_skipping_memoized]:
        result, missing = tiler(big_dag, library, allow_skip=False)
        assert result == []
        assert missing == [0, 1, 2, 3]

        result_skip, missing_skip = tiler(big_dag, library, allow_skip=True)
        assert missing_skip == [0]  # Skipping should succeed in tiling some nodes
        assert len(result_skip) == 1
        assert result_skip[0]['matched_nodes'] == {1, 2, 3}


def test_tile_vgg():
    # Example VGG-like network graph notation with additional layers
    # input -> conv1 -> relu1 -> conv2 -> relu2 -> conv3 -> relu3 -> fc1 -> relu4 -> fc2 -> softmax
    vgg_notation = (
        "1_conv->2_relu; "
        "2_relu->3_conv; "
        "3_conv->4_relu; "
        "4_relu->5_conv; "
        "5_conv->6_relu; "
        "6_relu->7_fc; "
        "7_fc->8_relu; "
        "8_relu->9_fc; "
        "9_fc->10_softmax"
    )
    big_dag = parse_graph(vgg_notation)
    lib_dag1 = parse_graph("0_conv->1_relu")
    lib_dag2 = parse_graph("0_fc")
    lib_dag3 = parse_graph("1_relu")
    lib_dag4 = parse_graph("0_softmax")
    lib_dag5 = parse_graph("0_fc->1_relu")
    

    for tiler in [tiler_backtracking_frontier_maximal_munch_with_skipping, tiler_backtracking_frontier_maximal_munch_with_skipping_memoized]:
        library = [lib_dag1, lib_dag2, lib_dag3, lib_dag4, lib_dag5]
        result, missing = tiler(big_dag, library)
        assert missing == []  # Expecting full coverage
        assert len(result) == 6  # Expecting 6 pieces to cover the big DAG
        assert result[0]['matched_nodes'] == {1, 2}  # conv -> relu
        assert result[1]['matched_nodes'] == {3, 4}  # conv -> relu
        assert result[2]['matched_nodes'] == {5, 6}  # conv -> relu
        assert result[3]['matched_nodes'] == {7, 8}  # fc -> relu
        assert result[4]['matched_nodes'] == {9}  # fc
        assert result[5]['matched_nodes'] == {10}  # softmax

        # if we remove fc -> relu from library, it should still work but use more pieces
        library.remove(lib_dag5)
        result2, missing2 = tiler(big_dag, library)
        assert missing2 == []  # Expecting full coverage
        assert len(result2) == 7  # Now expecting 7 pieces to cover the big DAG without fc->relu piece

        # if we remove individual fc and relu, then it should fail to cover fc and relu nodes
        library.remove(lib_dag2)
        library.remove(lib_dag3)
        result3, missing3 = tiler(big_dag, library)
        assert set(missing3) == {7, 8, 9}  # Expecting missing fc and relu nodes
        assert len(result3) == 4  # Only conv->relu and softmax pieces used


def get_resnet():
    # Example ResNet-18-like network graph notation
    # input -> conv1 -> bn1 -> relu1 -> maxpool
    # Then 4 residual blocks (2 blocks per stage, 4 stages), each block: conv->bn->relu->conv->bn + skip connection -> relu
    resnet18_notation = (
    "1_conv->2_bn; "
    "2_bn->3_relu; "
    "3_relu->4_maxpool; "

    # Stage 1, Block 1
    "4_maxpool->5_conv; "
    "5_conv->6_bn; "
    "6_bn->7_relu; "
    "7_relu->8_conv; "
    "8_conv->9_bn; "
    "4_maxpool->9_bn; "  # skip connection
    "9_bn->10_relu; "

    # Stage 1, Block 2
    "10_relu->11_conv; "
    "11_conv->12_bn; "
    "12_bn->13_relu; "
    "13_relu->14_conv; "
    "14_conv->15_bn; "
    "10_relu->15_bn; "  # skip connection
    "15_bn->16_relu; "

    # Stage 2, Block 1
    "16_relu->17_conv; "
    "17_conv->18_bn; "
    "18_bn->19_relu; "
    "19_relu->20_conv; "
    "20_conv->21_bn; "
    "16_relu->21_bn; "  # skip connection
    "21_bn->22_relu; "

    # Stage 2, Block 2
    "22_relu->23_conv; "
    "23_conv->24_bn; "
    "24_bn->25_relu; "
    "25_relu->26_conv; "
    "26_conv->27_bn; "
    "22_relu->27_bn; "  # skip connection
    "27_bn->28_relu; "

    # Stage 3, Block 1
    "28_relu->29_conv; "
    "29_conv->30_bn; "
    "30_bn->31_relu; "
    "31_relu->32_conv; "
    "32_conv->33_bn; "
    "28_relu->33_bn; "  # skip connection
    "33_bn->34_relu; "

    # Stage 3, Block 2
    "34_relu->35_conv; "
    "35_conv->36_bn; "
    "36_bn->37_relu; "
    "37_relu->38_conv; "
    "38_conv->39_bn; "
    "34_relu->39_bn; "  # skip connection
    "39_bn->40_relu; "

    # Stage 4, Block 1
    "40_relu->41_conv; "
    "41_conv->42_bn; "
    "42_bn->43_relu; "
    "43_relu->44_conv; "
    "44_conv->45_bn; "
    "40_relu->45_bn; "  # skip connection
    "45_bn->46_relu; "

    # Stage 4, Block 2
    "46_relu->47_conv; "
    "47_conv->48_bn; "
    "48_bn->49_relu; "
    "49_relu->50_conv; "
    "50_conv->51_bn; "
    "46_relu->51_bn; "  # skip connection
    "51_bn->52_relu; "

    # Final layers
    "52_relu->53_avgpool; "
    "53_avgpool->54_fc; "
    "54_fc->55_softmax"
    )
    return resnet18_notation


# this test needs `pip install pytest-timeout`
@pytest.mark.timeout(5)
@pytest.mark.xfail(reason="tiler_backtracking_frontier_maximal_munch_with_skipping should run longer than 5 seconds for this long graph")
def test_tile_resnet():
    big_dag = parse_graph(get_resnet())
    lib_dag1 = parse_graph("0_conv->1_bn->2_relu")
    lib_dag2 = parse_graph("0_conv->1_bn")
    lib_dag3 = parse_graph("0_fc->1_softmax")
    library = [lib_dag1, lib_dag2, lib_dag3]

    result, missing = tiler_backtracking_frontier_maximal_munch_with_skipping(big_dag, library)
    
def test_tile_resnet_fast():
    big_dag = parse_graph(get_resnet())
    lib_dag1 = parse_graph("0_conv->1_bn->2_relu")
    lib_dag2 = parse_graph("0_conv->1_bn")
    lib_dag3 = parse_graph("0_fc->1_softmax")
    library = [lib_dag1, lib_dag2, lib_dag3]

    result, missing = tiler_backtracking_frontier_maximal_munch_with_skipping_memoized(big_dag, library)
    assert missing == [4, 53]  # maxpool and avgpool are not covered


def test_tile_resnet_perfect_match():
    big_dag = parse_graph(get_resnet())
    lib_dag1 = parse_graph("0_conv->1_bn->2_relu")
    lib_dag2 = parse_graph("0_conv->1_bn")
    lib_dag3 = parse_graph("0_fc->1_softmax")
    lib_dag4 = parse_graph("0_avgpool")
    lib_dag5 = parse_graph("1_maxpool")
    library = [lib_dag1, lib_dag2, lib_dag3, lib_dag4, lib_dag5]

    result, missing = tiler_backtracking_frontier_maximal_munch_with_skipping_memoized(big_dag, library)
    assert missing == []  # Now we have perfect coverage

    result, missing = tiler_backtracking_frontier_maximal_munch_with_skipping(big_dag, library)
    assert missing == []  # Now we have perfect coverage, and the non-memoized slow one also works (because its perfectly covered, so it exits early)
