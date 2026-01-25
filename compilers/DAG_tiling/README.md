# Introduction

Consider a case where you have a big DAG of computations, and a library of smaller DAGs of computations. Lets say we have compiled programs for the smaller DAGs and we want to decompose/tile the big DAG in terms of the smaller DAGs

Here we fall back to some sensible heuristics to perform a quick tiling.

## Heuristics
We assume the bigger DAGs in the DAG library are preferred over smaller ones.

"Bigger is better" is just a stand-in; One can assign priorities to the DAGs in the library (maybe based on their performance), and in that case we can prioritize the library DAGs based on that order. 

## Notation
We'll use the format `nodeid(integer)_type` to denote a node. Edges will be written as `0_A->1_B` (this means Node `0` of type `A` links to node `1` of type `B`). See `graph_gen.py` for code to convert this notation to a an `nx` graph.

# Greedy Maximal Munch
Sort the library in descending order of size. Start from the largest DAG in the library and find all of its instances in the big DAG. Mark those nodes as "covered". Continue the process with the next DAG from the library.

See `dag_tile_1.py`

## Properties
1. Greedy
2. Prioritizes larger subgraphs from the library
3. Not guaranteed to find a covering even if it exists
4. Fast-ish, but still slow (Faster than full search)

## Why its not good
### Functionally
This is not a particularly good algorithm. For example, if we have a simple chain as the big DAG, `0_A->1_A->2_A->3_A->4_A->5_A`, and a library `[A->A]`, it might chomp off `1_A->2_A` and then `4_A->5_A`, leaving 2 uncovered nodes `0_A` and `3_A`.

### Performance
The `isomorphism.DiGraphMatcher` call is costly. Also `dag_tile_1.py` is pretty egregiously implemented, where we do not remove already covered nodes from the big DAG (to make the problem smaller in subsequent iterations), instead each call to `isomorphism.DiGraphMatcher` takes the whole big DAG. A low hanging optimization would be to keep subtracting already covered nodes from the big DAG



# Frontier Maximal Munch

The previous algorithm was a starting point, but it doesnt get us far, both in terms of functionality and performance. In this section we'll try to rectify the following:
1. Performance: Try to avoid `isomorphism.DiGraphMatcher` call
2. Functionality: Provide an answer if it exists

## Greedy

Topo-sort the big DAG, start from the source nodes. The idea is to peel away maximum number of nodes from the frontier depending on the largest match we can get in the library.

Like the previous approach, sort the library in descending order, to enforce the preference of large chunks.

Match the largest subgraph in the library to the big DAG which has the frontier big DAG node

This logic is present in `dag_tile_2.py`.

### A note about rooted match
In `dag_tile_2.py`, we have 2 functions, `fake_rooted_match` and `real_rooted_match`. The fake one uses `isomorphism.DiGraphMatcher` and then filters out matches based on presence of the frontier node. This is inefficient (because we are still calling `isomorphism.DiGraphMatcher`), however the function shows the logic succintly

`real_rooted_match` does an actual "rooted local search", which means its much faster.



### Properties
1. Greedy
2. Prioritizes larger subgraphs from the library
3. Not guaranteed to find a covering even if it exists
4. Faster (because a "rooted" sub-graph match is much faster than `isomorphism.DiGraphMatcher`)

### Why its not good
Being a greedy algorithm, with no recovery mechanism, it can still fall flat on its face. For example: big DAG = `0_A->1_B->2_C->3_D` and the library. =`[0_A->1_B,0_A->1_B->2_C,2_C->3_D]` it will choose to chomp off `0_A->1_B->2_C`, which leaves it stranded with an uncovered node `3_D`. See `test_tile_2_greedy_fail`


## Backtracking

In `dag_tile_3.py` we introduce backtracking. This allows us to go back and try other options if we run into a dead end (uncovered nodes left). `test_tile_2_greedy_fail_backtrack_pass` shows a case where backtracking passes while greedy approaches fail.

### Adding skipping
Sometimes the algo might fail to match the current node with any pattern. At that point instead of giving up, we can skip. `tiler_backtracking_frontier_maximal_munch_with_skipping` in `dag_tile_3.py` can be configured to skip or not using `allow_skip`

### Optimizations

As with any backtracking algorithm, we can throw some bog standard optimizations at it

#### Pruning
While searching for a tiling, if the best tile has 2 missing nodes, and right now I am on a path where I have already got 3 skipped node, this path cannot be the best path, so stop the search

#### Memoization
Consider a diamond, with a long tail, a kite if you will: `A->B; A->C; B->D0; C->D0; D0->D1; D1->D2; D2->D3`

Lets say we choose to tile `A->B` and then we explire the tail `D` and we do not find a perfect tiling. However we have a "best" tiling for the tail.
Now it tries `A->C`, but because it has no memory, it will retile the tail again

Memoization prevents this. It stores results it has seen using a key such as: `(order_index, frozenset(covered_nodes))`. `order_index` tell us where we are in the topo sort, and `covered_nodes` tells us which nodes have already been tiled/munched. The cache's values are a tuple `(best_sub_solution, total_covered_in_subgraph)`


#### Using both
TODO, write why both together kind of conflict, and which might be more useful.
Pruning: maybe is we have time estimates etc
Memoization: long, repeated tails

#### Current status
Lets choose to do memoization. See `test_tile_resnet` as an example which take a long time if we do not memoize.

`dag_tile_4.py` implements the memoized version

Note in case of a perfect both the memoized and non-memoized versions can be fast enough (see `test_tile_resnet_perfect_match`)


# Run tests

```
pip install -r requirements.txt
pytest -s
```