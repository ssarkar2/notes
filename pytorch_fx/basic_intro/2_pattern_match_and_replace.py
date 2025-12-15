import torch
import torch.nn as nn
import torch.fx as fx
import operator


# Original graph
class M(nn.Module):
    def forward(self, x, y):
        return torch.relu(x + y)


# Fused graph
def fused_add_relu(x, y):
    return torch.relu(x + y)


# Trace original
m = M()
gm = fx.symbolic_trace(m)
graph = gm.graph

# Traverse and find the pattern
for node in graph.nodes:
    # Look for a relu node whose input is an add
    if node.op == "call_function" and node.target == torch.relu:
        input_node = node.args[0]
        if input_node.op == "call_function" and input_node.target == operator.add:
            add_node = input_node
            relu_node = node
            break

# Insert the fused node after add_node's inputs
with graph.inserting_after(add_node.args[0]):
    fused_node = graph.call_function(
        fused_add_relu, 
        args=add_node.args,  # x and y
        kwargs={}
    )

relu_node.replace_all_uses_with(fused_node)


graph.eliminate_dead_code()
graph.lint()
gm.recompile()
