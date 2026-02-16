import torch
from torch.fx import symbolic_trace
import time
import torch.nn as nn

from block_detect import detect_block_boundaries_binary, detect_block_boundaries
import pytest

class SimpleBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))

class RepetitiveModel(torch.nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList([SimpleBlock() for _ in range(num_layers)])
        self.final = torch.nn.Linear(10, 10)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@pytest.mark.parametrize("block_detector", [detect_block_boundaries_binary, detect_block_boundaries])
@pytest.mark.parametrize("num_layers", [1, 2, 4, 10])
def test_resnet(block_detector, num_layers):
    # Run detection
    model = RepetitiveModel(num_layers).eval()

    gm = symbolic_trace(model)

    start = time.time()
    pattern, count = block_detector(gm, num_layers)
    print(f"Time to run block detection in RepetitiveModel: num_layers={num_layers}, algo={block_detector.__name__}, time={time.time() - start:.6f}s")
    assert pattern == ('call_module:Conv2d', 'call_module:ReLU')
    assert count == num_layers


class MiniAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        # Split QKV and reshape for multi-head
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MiniAttention(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class LocalTransformer(nn.Module):
    def __init__(self, vocab_size=1000, dim=128, heads=4, layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(layers)])
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


@pytest.mark.parametrize("block_detector", [detect_block_boundaries_binary, detect_block_boundaries])
@pytest.mark.parametrize("num_layers", [1, 2, 4, 10, 50])
def test_transformer(block_detector, num_layers):
    model = LocalTransformer(layers=num_layers)
    model.eval()
    gm = symbolic_trace(model)
    start = time.time()
    pattern, count = block_detector(gm, num_layers)
    print(f"Time to run block detection in LocalTransformer: num_layers={num_layers}, algo={block_detector.__name__}, time={time.time() - start:.6f}s")
    if num_layers == 1:
        assert pattern == ('call_module:Embedding', 'call_module:LayerNorm', 'call_function:<built-in function getattr>', 'call_function:<built-in function getitem>', 'call_function:<built-in function getitem>', 'call_function:<built-in function getitem>', 'call_module:Linear', 'call_function:<built-in function floordiv>', 'call_method:reshape', 'call_method:permute', 'call_function:<built-in function getitem>', 'call_function:<built-in function getitem>', 'call_function:<built-in function getitem>', 'call_method:transpose', 'call_function:<built-in function matmul>', 'call_function:<built-in function mul>', 'call_method:softmax', 'call_function:<built-in function matmul>', 'call_method:transpose', 'call_method:reshape', 'call_module:Linear', 'call_function:<built-in function add>', 'call_module:LayerNorm', 'call_module:Linear', 'call_module:ReLU', 'call_module:Linear', 'call_function:<built-in function add>', 'call_module:Linear')
    else:
        assert pattern == ('call_module:LayerNorm', 'call_function:<built-in function getattr>', 'call_function:<built-in function getitem>', 'call_function:<built-in function getitem>', 'call_function:<built-in function getitem>', 'call_module:Linear', 'call_function:<built-in function floordiv>', 'call_method:reshape', 'call_method:permute', 'call_function:<built-in function getitem>', 'call_function:<built-in function getitem>', 'call_function:<built-in function getitem>', 'call_method:transpose', 'call_function:<built-in function matmul>', 'call_function:<built-in function mul>', 'call_method:softmax', 'call_function:<built-in function matmul>', 'call_method:transpose', 'call_method:reshape', 'call_module:Linear', 'call_function:<built-in function add>', 'call_module:LayerNorm', 'call_module:Linear', 'call_module:ReLU', 'call_module:Linear', 'call_function:<built-in function add>')
    assert count == num_layers



