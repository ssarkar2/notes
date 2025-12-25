
import torch
from datasets import load_dataset
import torch.fx as fx
import torch.nn as nn
from pytorch_fx.utils import get_device
from pytorch_fx.utils_resnet18 import get_resnet18_model, eval


def fold_bn_into_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    assert not conv.training
    assert not bn.training

    W = conv.weight
    if conv.bias is None:
        b = torch.zeros(W.size(0), device=W.device)
    else:
        b = conv.bias

    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    std = torch.sqrt(var + eps)
    scale = gamma / std

    W_fused = W * scale.reshape(-1, 1, 1, 1)
    b_fused = (b - mean) * scale + beta

    return W_fused, b_fused

def fuse_conv_bn(gm: fx.GraphModule):
    gm.eval()
    modules = dict(gm.named_modules())
    graph = gm.graph

    for node in gm.graph.nodes:
        if node.op != "call_module":
            continue

        bn = modules.get(node.target)
        if not isinstance(bn, nn.BatchNorm2d):
            continue

        prev = node.args[0]
        if prev.op != "call_module":
            continue

        conv = modules.get(prev.target)

        if not isinstance(conv, nn.Conv2d):
            continue

        W_fused, b_fused = fold_bn_into_conv(conv, bn)

        # Update Conv in-place
        conv.weight.data.copy_(W_fused)
        if conv.bias is None:
            conv.bias = nn.Parameter(b_fused)
        else:
            conv.bias.data.copy_(b_fused)

        # Redirect BN users to Conv output
        node.replace_all_uses_with(prev)

        # Remove BN node from graph
        assert len(node.users) == 0
        graph.erase_node(node)
        gm.delete_submodule(node.target)
    graph.lint()
    gm.recompile()
    return gm



def run_expt(fuse):
    device = get_device()
    model, preprocess = get_resnet18_model(device)
    print(f"Model initialized and moved to: {device}")
    dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
    num_imgs = 10000
    eval(dataset, model, preprocess, device, num_imgs)

    model = fx.symbolic_trace(model)
    if fuse:
        model = fuse_conv_bn(model)
    eval(dataset, model, preprocess, device, num_imgs)

    model_compiled = torch.compile(model)
    eval(dataset, model_compiled, preprocess, device, num_imgs)

if __name__ == "__main__":
    run_expt(fuse=False)
    run_expt(fuse=True)


'''
MPS:
Accuracy on 10000 streamed images: 70.41%. Time taken: 0.017031 seconds per image
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [02:57<00:00, 56.28it/s]
Accuracy on 10000 streamed images: 70.41%. Time taken: 0.017173 seconds per image
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [01:53<00:00, 87.80it/s]
Accuracy on 10000 streamed images: 70.41%. Time taken: 0.010710 seconds per image


Accuracy on 10000 streamed images: 70.41%. Time taken: 0.017394 seconds per image
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [02:03<00:00, 80.93it/s]
Accuracy on 10000 streamed images: 70.41%. Time taken: 0.011603 seconds per image
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [01:57<00:00, 85.14it/s]
Accuracy on 10000 streamed images: 70.41%. Time taken: 0.010871 seconds per image
'''