import torch
import torch.nn as nn
from observer import ObserverModule
from quant_params import TensorStats, QuantDtype, QuantParams
from quantization_simulate import compute_quant_params
from int_modules import IntAdd, IntMul, IntLinear, IntReLU
from quantization_simulate import QuantizationSimulator
from torchvision.models import ResNet18_Weights
import torchvision.models as models
from tqdm import tqdm
import time
from datasets import load_dataset
import itertools
import torch.nn.utils.fusion as fusion
import torch.fx as fx

class ToyModel(nn.Module):
    """A toy model with supported and unsupported operations."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)  # Unsupported op
        self.fc = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)  # Unsupported - will trigger dequant before, quant after
        x = x.flatten(1)
        x = self.fc(x)
        return x

def test_observer_basic():
    obs = ObserverModule()
    x1 = torch.tensor([[-1.0, 0.5, 2.0]])
    x2 = torch.tensor([[3.0, -0.2]])
    # call twice to aggregate min/max
    obs(x1)
    obs(x2)
    stats = obs.get_stats()
    assert isinstance(stats, TensorStats)
    assert stats.min_val <= -1.0
    assert stats.max_val >= 3.0

def test_compute_quant_params_nonzero():
    stats = TensorStats(min_val=-2.0, max_val=2.0)
    qp = compute_quant_params(stats, QuantDtype.INT8)
    assert isinstance(qp, QuantParams)
    assert qp.scale > 0
    assert qp.zero_point == 0

def test_compute_quant_params_zeros():
    # all zeros should avoid divide-by-zero
    stats = TensorStats(min_val=0.0, max_val=0.0)
    qp = compute_quant_params(stats, QuantDtype.INT8)
    assert qp.scale > 0


def make_qp(scale=1.0):
    return QuantParams(scale=scale, zero_point=0, dtype=QuantDtype.INT8)

def test_int_add_mul_basic():
    a = torch.tensor([[1, 2]], dtype=torch.int8)
    b = torch.tensor([[3, -1]], dtype=torch.int8)
    a_q = make_qp(0.1)
    b_q = make_qp(0.2)
    out_q = make_qp(0.05)
    add = IntAdd(a_q, b_q, out_q)
    res = add(a, b)
    assert res.dtype == torch.int8
    mul = IntMul(a_q, b_q, out_q)
    res2 = mul(a, b)
    assert res2.dtype == torch.int8

def test_int_linear_forward():
    lin = nn.Linear(4, 2)
    # simple input
    x = torch.randn(1, 4)
    # quant params (scale chosen arbitrarily)
    in_q = make_qp(0.1)
    w_q = make_qp(0.05)
    out_q = make_qp(0.2)
    int_lin = IntLinear(lin, in_q, w_q, out_q)
    # simulate int input (int8)
    x_int = torch.round(x / in_q.scale).to(torch.int8)
    y = int_lin(x_int)
    assert y.dtype == torch.int8

def test_simulator_smoke_linear():
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    sim = QuantizationSimulator(model, dtype=QuantDtype.INT8)
    sim.prepare_for_observation()
    # small calibration set
    calib = [torch.randn(4, 8) for _ in range(3)]
    sim.calibrate(calib)
    q = sim.convert_to_quantized()
    # run compare_outputs on single input
    metrics = sim.compare_outputs(torch.randn(2, 8))
    assert 'mse' in metrics
    assert metrics['mse'] >= 0.0


def test_int_mul_exact_same_scale():
    # Both inputs and output use the same scale 1.0 and zero_point 0
    a = torch.tensor([[1, 2, -3]], dtype=torch.int32)
    b = torch.tensor([[4, -1, 3]], dtype=torch.int32)

    a_q = make_qp(scale=1.0)
    b_q = make_qp(scale=1.0)
    out_q = make_qp(scale=1.0)

    mul = IntMul(a_q, b_q, out_q)
    res = mul(a, b)

    # Expected: elementwise integer multiplication
    expected = torch.tensor([[4, -2, -9]], dtype=res.dtype)

    assert torch.equal(res, expected)


def test_int_mul_different_scales():
    # a has scale 0.5, b has scale 1.0, output scale 0.25
    # real product scale = a_scale * b_scale = 0.5
    # output scale = 0.25 -> factor = 0.5 / 0.25 = 2
    a = torch.tensor([[2, -2]], dtype=torch.int32)  # represents [1.0, -1.0] if scale 0.5
    b = torch.tensor([[3, 1]], dtype=torch.int32)   # represents [3.0, 1.0] if scale 1.0

    a_q = make_qp(scale=0.5)
    b_q = make_qp(scale=1.0)
    out_q = make_qp(scale=0.25)

    mul = IntMul(a_q, b_q, out_q)
    res = mul(a, b)

    # Compute expected manually:
    # real product: [1.0*3.0, -1.0*1.0] = [3.0, -1.0]
    # output int at scale 0.25: [3.0/0.25, -1.0/0.25] = [12, -4]
    expected = torch.tensor([[12, -4]], dtype=res.dtype)

    assert torch.equal(res, expected)

def test_int_add_exact_same_scale():
    # Both inputs and output use the same scale 1.0 and zero_point 0
    a = torch.tensor([[1, 2, -3]], dtype=torch.int8)
    b = torch.tensor([[4, -1, 3]], dtype=torch.int8)

    a_q = make_qp(scale=1.0)
    b_q = make_qp(scale=1.0)
    out_q = make_qp(scale=1.0)

    add = IntAdd(a_q, b_q, out_q)
    res = add(a, b)

    # Expected: elementwise integer addition, clamped to qmin/qmax
    expected = torch.tensor([[5, 1, 0]], dtype=res.dtype)

    assert torch.equal(res, expected)




def test_int_add_different_scales():
    # a has scale 0.5, b has scale 1.0, output scale 0.5
    # So a is finer; to align to output scale 0.5, a_scale/out_scale = 1, b_scale/out_scale = 2
    a = torch.tensor([[2, -2]], dtype=torch.int8)  # represents [1.0, -1.0] if scale 0.5
    b = torch.tensor([[3, 1]], dtype=torch.int8)   # represents [3.0, 1.0] if scale 1.0

    a_q = make_qp(scale=0.5)
    b_q = make_qp(scale=1.0)
    out_q = make_qp(scale=0.5)

    add = IntAdd(a_q, b_q, out_q)
    res = add(a, b)

    # Compute expected manually:
    # a real: [2*0.5, -2*0.5] = [1.0, -1.0]
    # b real: [3*1.0, 1*1.0] = [3.0, 1.0]
    # sum: [4.0, 0.0]
    # output int at scale 0.5: [4.0 / 0.5, 0.0 / 0.5] = [8, 0]
    expected = torch.tensor([[8, 0]], dtype=res.dtype)

    assert torch.equal(res, expected)



def node_name_repr(node):
    # Return a string representation for the node's target for easier assertions
    if node.op == 'call_module':
        return str(node.target)
    elif node.op == 'call_function':
        try:
            return node.target.__name__
        except Exception:
            return str(node.target)
    else:
        return f"{node.op}:{node.target}"


def test_quantize_dequantize_boundary():
    # Build model A->B->C->D where C (pool) is unsupported
    model = ToyModel()
    sim = QuantizationSimulator(model, dtype=QuantDtype.INT8)

    sim.prepare_for_observation()
    # small calibration dataset
    calib = [torch.randn(2, 3, 32, 32) for _ in range(3)]
    sim.calibrate(calib)

    qmod = sim.convert_to_quantized()
    nodes = list(qmod.graph.nodes)
    names = [node_name_repr(n) for n in nodes]

    # Find a dequantize node corresponding to the boundary before 'pool'
    # We expect pattern: ... int_* , dequantize_*, pool, quantize_*, int_* ...
    found = False
    for i in range(len(names) - 5):
        n0 = names[i] #int
        n1 = names[i + 1] #dequant
        n2 = names[i + 2] #pool
        n3 = names[i + 3] #flatten
        n4 = names[i + 4] #quant
        n5 = names[i + 5] #int

        if n0.startswith('int_') and n1.startswith('dequantize_') and n2 == 'pool' and n4.startswith('quantize_') and n5.startswith('int_'):
            found = True
            break

    assert found, f"Did not find expected quantize/dequantize boundary pattern in graph. Nodes: {names}"




class BranchingModel(nn.Module):
    """A model with a branching path where `e` is an unsupported op.

    Graph:
        A -> B -> C -> D
        A -> E -> D

    E is nn.AdaptiveAvgPool2d (treated as unsupported) so we expect a
    dequantize before E and a quantize after E in the converted graph.
    """
    def __init__(self):
        super().__init__()
        self.a = nn.Conv2d(3, 16, 3, padding=1)
        self.b = nn.Conv2d(16, 16, 3, padding=1)
        self.c = nn.Conv2d(16, 8, 3, padding=1)
        self.e = nn.AdaptiveAvgPool2d(1)  # Unsupported op -> should create dequant/quant boundary
        # D consumes pooled C (8) and pooled A via E (16) -> total 24 features
        self.d = nn.Linear(8 + 16, 10)

    def forward(self, x):
        a = self.a(x)           # A
        b = self.b(a)           # B
        c = self.c(b)           # C
        c_pooled = c.mean(dim=(2, 3))          # (N, 8)
        # E is an unsupported module (AdaptiveAvgPool2d)
        e = self.e(a)            # E (unsupported)
        e = torch.flatten(e, 1)  # (N, 16)
        # merge branches and feed D
        merged = torch.cat([c_pooled, e], dim=1)
        out = self.d(merged)     # D
        return out


def test_quantize_dequantize_branch():
    model = BranchingModel()
    sim = QuantizationSimulator(model, dtype=QuantDtype.INT8)

    sim.prepare_for_observation()
    calib = [torch.randn(2, 3, 32, 32) for _ in range(3)]
    sim.calibrate(calib)

    qmod = sim.convert_to_quantized()
    nodes = list(qmod.graph.nodes)
    names = [node_name_repr(n) for n in nodes]

    assert names == ['placeholder:x', 'quantize_0', 'int_conv2d_1', 'int_conv2d_2', 'int_conv2d_3', 'dequantize_4', 'call_method:mean', 'dequantize_5', 'e', 'flatten', 'cat', 'quantize_6', 'int_linear_7', 'dequantize_8', 'output:output']






def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device



def get_resnet18_model(device):
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()
    return model.to(device), weights.transforms()



def eval(dataset, model, preprocess, device, num_imgs = 20):
    correct = 0
    with torch.no_grad():
        for i, example in enumerate(tqdm(dataset, total=num_imgs)):
            if i==10: # warmup
                start_time = time.time()
            if i >= num_imgs: break # Stop after num_imgs images

            image = preprocess(example['image'].convert('RGB')).unsqueeze(0).to(device)
            output = model(image)

            if example['label'] == output.argmax(1).item():
                correct += 1
        end_time = time.time()
    print(f"Accuracy on {num_imgs} streamed images: {100 * correct / num_imgs:.2f}%. Time taken: {(end_time - start_time)/(num_imgs - 10):.6f} seconds per image")
    return (100 * correct) / num_imgs



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


def test_resnet_quantization_accuracy():
    device = 'cpu'#get_device()
    model, preprocess = get_resnet18_model(device)
    model = fx.symbolic_trace(model)
    model = fuse_conv_bn(model)

    # Calibration: 100 streaming training images
    train_ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
    calib_inputs = []
    for ex in itertools.islice(train_ds, 100):
        img = preprocess(ex["image"].convert("RGB")).unsqueeze(0).to(device)
        calib_inputs.append(img)

    sim = QuantizationSimulator(model, dtype=QuantDtype.INT8)
    # prepare with one example and run calibration
    sim.prepare_for_observation()
    sim.calibrate(calib_inputs)
    qmodel = sim.convert_to_quantized()
    qmodel.to(device)

    # Validation: 100 streaming validation images
    val_ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)

    orig_acc = eval(val_ds, model, preprocess, device, num_imgs=100)
    q_acc = eval(val_ds, qmodel, preprocess, device, num_imgs=100)

    # Allow some degradation but ensure quantized model is reasonably close
    assert abs(orig_acc - q_acc) <= 10.0, f"Original acc: {orig_acc}, Quantized acc: {q_acc}"
    #print(orig_acc, q_acc)


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.b = 5
    def forward(self, x, y):
        a = x + y
        b = a * x
        c = a - b
        #d = c + 4 # todo this wont work now, is a bug here adding const
        #return d + self.b
        return c
    

def test_quantize_mymodel_basic():
    torch.manual_seed(0)
    model = MyModel()
    sim = QuantizationSimulator(model, dtype=QuantDtype.INT8)

    # prepare and calibrate with a few example input tuples
    sim.prepare_for_observation()
    calib = []
    for _ in range(3):
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        calib.append((x, y))
    sim.calibrate(calib)

    qmodel = sim.convert_to_quantized()

    # verify quantization nodes are present in the converted graph
    nodes = list(qmodel.graph.nodes)
    names = [node_name_repr(n) for n in nodes]

    has_quant = any(n.startswith("quantize_") for n in names)
    has_dequant = any(n.startswith("dequantize_") for n in names)
    has_int = any(n.startswith("int_") for n in names)

    assert has_quant, f"No quantize nodes found in qmodel graph. Nodes: {names}"
    assert has_dequant or has_int, f"No dequantize or int nodes found in qmodel graph. Nodes: {names}"

    # Run a test input through both models and compare outputs
    x_t = torch.randn(4, 3)
    y_t = torch.randn(4, 3)
    out_ref = model(x_t, y_t)
    out_q = qmodel(x_t, y_t)

    assert out_ref.shape == out_q.shape
    assert torch.isfinite(out_q).all()
    # Allow some tolerance due to quantization; ensure difference is reasonable
    diff = (out_ref - out_q).abs().mean().item()
    assert diff < 1.0