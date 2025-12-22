import torch

import torch.nn as nn
import torch.fx as fx

# Define QuickGELU activation
class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)



# FX pass to replace ReLU with QuickGELU
def replace_relu_with_quickgelu(gm: fx.GraphModule):
    for node in gm.graph.nodes:
        if node.op == 'call_module':
            target_mod = dict(gm.named_modules())[node.target]
            if isinstance(target_mod, nn.ReLU):
                # Replace with QuickGELU
                new_name = node.target + "_quick"
                gm.add_submodule(new_name, QuickGELU())
                node.target = new_name
    gm.recompile()
    return gm

# FX pass to replace GELU with ReLU
def replace_gelu_with_relu(gm: fx.GraphModule):
    for node in gm.graph.nodes:
        if node.op == 'call_module':
            target_mod = dict(gm.named_modules())[node.target]
            if isinstance(target_mod, nn.GELU):
                # Replace with ReLU
                new_name = node.target + "_relu"
                gm.add_submodule(new_name, nn.ReLU())
                node.target = new_name
    gm.recompile()
    return gm

# FX pass to replace GELU(approximate=None) with GELU(approximate='tanh')
def replace_gelu_approximate_with_tanh(gm: fx.GraphModule):
    for node in gm.graph.nodes:
        if node.op == 'call_module':
            target_mod = dict(gm.named_modules())[node.target]
            if isinstance(target_mod, nn.GELU) and (target_mod.approximate is None or target_mod.approximate == 'none'):
                # Edit the existing module in-place
                target_mod.approximate = 'tanh'
    gm.recompile()
    return gm

if __name__ == "__main__":
    # Example model using GELU
    class MyModel(nn.Module):
        def __init__(self, approx='none'):
            super().__init__()
            self.act = nn.GELU(approximate=approx)

        def forward(self, x):
            return self.act(x)

    # Instantiate and trace the model
    model = MyModel()
    model1 = MyModel('tanh')
    traced = fx.symbolic_trace(model)

    transformations = [
        replace_gelu_approximate_with_tanh,
        replace_gelu_with_relu,
        replace_relu_with_quickgelu,
    ]

    x = torch.randn(200, 400)
    orig = model(x)
    orig1 = model1(x)
    print("Original:", orig)
    print("Original (with tanh):", orig1)
    for transform in transformations:
        traced = transform(traced)
        print(f"After applying {transform.__name__}:")
        print(traced(x))

   

