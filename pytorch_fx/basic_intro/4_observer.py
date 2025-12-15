import torch
import torch.nn as nn
import torch.fx as fx

# Example module
class M(nn.Module):
    def forward(self, x):
        a = x + 1
        b = a * 2
        c = torch.relu(b)
        return c

m = M()
gm = fx.symbolic_trace(m)
graph = gm.graph

x = torch.randn(3, 4)
y = gm(x)
print(y)

# Find the node we want to observe
target_node = None
for node in gm.graph.nodes:
    if node.op == "call_function" and "mul" in str(node.target):
        target_node = node
        break

shapes = []

'''
The observer takes in a tensor and returns it unchanged
'''
def observer_fn(x):
    shapes.append(x.shape)  # collect shape
    return x                # pass tensor through unchanged

with graph.inserting_after(target_node):
    obs_node = gm.graph.call_function(observer_fn, args=(target_node,))

target_node.replace_all_uses_with(obs_node)
# Make the observer use target_node as input
obs_node.replace_input_with(obs_node, target_node)
#obs_node.args = (target_node,) # alternatively

gm.graph.lint()
gm.recompile()


y = gm(x)
print(y)
x = torch.randn(30, 40)
gm(x)


print(shapes)
'''
[torch.Size([3, 4]), torch.Size([30, 40])]
'''