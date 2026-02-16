import torch
from torch.fx import symbolic_trace
from collections import Counter

def get_node_sig(node, gm):
    if node.op == 'call_module':
        mod = gm.get_submodule(node.target)
        return f"{node.op}:{type(mod).__name__}"
    return f"{node.op}:{node.target}"


def detect_block_boundaries(gm: torch.fx.GraphModule, num_layers: int, min_len=None, max_len=None):
    nodes = [n for n in gm.graph.nodes if n.op not in ['placeholder', 'output']]
    sig_sequence = [get_node_sig(n, gm) for n in nodes]

    if min_len is None:
        min_len = 1
    if max_len is None:
        max_len = int((len(nodes) // num_layers) * 1.1)
    max_len = max(max_len, len(nodes))

    final_pattern = []
    final_count = 0
    
    # Iterate from smallest to largest length
    for length in range(min_len, max_len + 1):
        patterns = []
        for i in range(len(sig_sequence) - length + 1):
            patterns.append(tuple(sig_sequence[i:i+length]))
        
        if not patterns:
            continue
            
        counts = Counter(patterns)
        # Get the most common pattern for this specific length
        pattern, count = counts.most_common(1)[0]
        

        if count == num_layers:
            # if we find a pattern that fits num_layers, try to find the largest one
            if len(pattern) > len(final_pattern):
                final_pattern = pattern
                final_count = count
        elif count < num_layers:
            # length increases, count decreases, so we can break early
            break


    return final_pattern, final_count



def detect_block_boundaries_binary(gm, num_layers, min_len=1, max_len=None):
    nodes = [n for n in gm.graph.nodes if n.op not in ['placeholder', 'output']]
    sig_sequence = [get_node_sig(n, gm) for n in nodes]

    if max_len is None:
        max_len = len(nodes) // num_layers

    low = min_len
    high = max_len
    best_pattern = []
    
    while low <= high:
        mid = (low + high) // 2
        if mid == 0: 
            low = mid + 1
            continue

        # Check most common pattern at this specific length
        patterns = [tuple(sig_sequence[i:i+mid]) for i in range(len(sig_sequence) - mid + 1)]
        if not patterns:
            high = mid - 1
            continue
            
        counter = Counter(patterns)
        pattern, count = counter.most_common(1)[0]

        if count >= num_layers:
            # This length works! Save it and try to find a longer one.
            best_pattern = pattern
            low = mid + 1
        else:
            # Too long, pattern doesn't repeat num_layers times.
            high = mid - 1

    return best_pattern, num_layers