import torch

import torch.nn as nn
import torch.fx as fx

# Define a simple PyTorch module
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 2)
        self.linear2 = nn.Linear(2, 2)

    def forward(self, x):
        return torch.nn.functional.gelu(self.linear2(torch.relu(self.linear1(x))))

# Instantiate the module
module = MyModule()

# Wrap the module with torch.fx symbolic tracing
traced = fx.symbolic_trace(module)

# Print the FX graph
print("FX Graph:")
print(traced.graph)

# You can also run the traced module
x = torch.randn(1, 4)
output = traced(x)
print("Output:", output)

# Show all nodes in the graph
print("\nNodes in the FX graph:")
for node in traced.graph.nodes:
    print(node)
'''
Nodes in the FX graph:
x
linear1
relu
linear2
gelu
output
'''


print(traced.graph.print_tabular())
'''
opcode         name     target                                                args        kwargs
-------------  -------  ----------------------------------------------------  ----------  --------
placeholder    x        x                                                     ()          {}
call_module    linear1  linear1                                               (x,)        {}
call_function  relu     <built-in method relu of type object at 0x103940a50>  (linear1,)  {}
call_module    linear2  linear2                                               (relu,)     {}
call_function  gelu     <built-in function gelu>                              (linear2,)  {}
output         output   output                                                (gelu,)     {}
None
'''


nodes = list(traced.graph.nodes)
print(traced.graph.python_code(nodes[0]))
print(traced.graph.python_code(nodes[2]))

print((traced.graph.python_code(nodes[0]).src))

'''
def forward(self, x):
    linear1 = x.linear1(x);  x = None
    relu = torch.relu(linear1);  linear1 = None
    linear2 = x.linear2(relu);  relu = None
    gelu = torch._C._nn.gelu(linear2);  linear2 = None
    return gelu
'''



print(traced.graph.find_nodes(op="call_module"))
'''
[linear1, linear2]
'''

print(traced.graph.find_nodes(op="call_function", target=torch.relu))
'''
[relu]

target must be specified for call_function op
'''
breakpoint()
print()