import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx import subgraph_rewriter
import operator


class M(nn.Module):
    def forward(self, x):
        return x + 0


inp = torch.tensor([1,2,3])

m = M()
ref0 = m(inp)
gm = fx.symbolic_trace(m)
ref1 = gm(inp)
print("Before:")
print(gm.graph)

class RemoveAddZero(fx.Transformer):
    def call_function(self, target, args, kwargs):
        if target == operator.add:
            lhs, rhs = args
            if isinstance(rhs, fx.Proxy) is False and rhs == 0:
                return lhs
        return super().call_function(target, args, kwargs)
    
new_gm = RemoveAddZero(gm).transform()
print("After (using Transformer):")
print(new_gm.graph)
ref2 = new_gm(inp)
print('-------')


m1 = M()
gm = fx.symbolic_trace(m1)
ref3 = m1(inp)
ref4 = gm(inp)

print("Before:")
print(gm.graph)
def add_zero_pattern(x):
    return x + 0
def add_zero_replacement(x):
    return x
subgraph_rewriter.replace_pattern(
    gm,
    add_zero_pattern,
    add_zero_replacement
)
print("After (using subgraph_rewriter):")
print(gm.graph)
ref5 = gm(inp)

assert torch.equal(ref0, ref1)
assert torch.equal(ref0, ref2)
assert torch.equal(ref0, ref3)
assert torch.equal(ref0, ref4)
assert torch.equal(ref0, ref5)

'''
Before:
graph():
    %x : [num_users=1] = placeholder[target=x]
    %add : [num_users=1] = call_function[target=operator.add](args = (%x, 0), kwargs = {})
    return add
After (using Transformer):
graph():
    %x : [num_users=1] = placeholder[target=x]
    return x
-------
Before:
graph():
    %x : [num_users=1] = placeholder[target=x]
    %add : [num_users=1] = call_function[target=operator.add](args = (%x, 0), kwargs = {})
    return add
After (using subgraph_rewriter):
graph():
    %x : [num_users=1] = placeholder[target=x]
    return x
'''