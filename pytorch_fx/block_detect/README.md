# Block detection

Sometimes we need to detect block boundaries in models (maybe because we want ot apply blockwise quantization or some other reason)

If a block is just a repeated pattern, then this is just detecting the "time period" of the lowest frequency in a discrete series

Also we know the "number of layers" in most cases, so we can try to leverage that, so we are trying to find the largest block that fits "number of layers" times. Note that we can get non unique answers in some cases, like:
"ABABABA" might give us (AB) or (BA) as the block

## Sliding window

Implements a Bottom-Up Greedy Motif Search

1. Normalization: : Linearizes the graph into a sequence of Signatures (Op types + Module classes).
2. Iterative Expansion: Starts with a window size of min_len and incrementally increases it.
3. Frequency Filtering: If a pattern occurs exactly num_layers, it is tracked as a potential block. The algorithm continues to "grow" the window to find the largest possible block size.
4. Early Exit: Terminate search once frequency drops below num_layers, ensuring optimal performance on deep graphs.

### Optimization
Instead of searching linearly, one could potentially "binary search" the space of motif_len by getting feedback from current count. Check `detect_block_boundaries_binary`



## Testing

```
pytest
```

Binary search version is faster for larger models, here are some comparisions:

```
test_model.py Time to run block detection in RepetitiveModel: num_layers=1, algo=detect_block_boundaries_binary, time=0.000028s
.Time to run block detection in RepetitiveModel: num_layers=1, algo=detect_block_boundaries, time=0.000015s
.Time to run block detection in RepetitiveModel: num_layers=2, algo=detect_block_boundaries_binary, time=0.000020s
.Time to run block detection in RepetitiveModel: num_layers=2, algo=detect_block_boundaries, time=0.000020s
.Time to run block detection in RepetitiveModel: num_layers=4, algo=detect_block_boundaries_binary, time=0.000030s
.Time to run block detection in RepetitiveModel: num_layers=4, algo=detect_block_boundaries, time=0.000032s
.Time to run block detection in RepetitiveModel: num_layers=10, algo=detect_block_boundaries_binary, time=0.000062s
.Time to run block detection in RepetitiveModel: num_layers=10, algo=detect_block_boundaries, time=0.000066s
.Time to run block detection in LocalTransformer: num_layers=1, algo=detect_block_boundaries_binary, time=0.000052s
.Time to run block detection in LocalTransformer: num_layers=1, algo=detect_block_boundaries, time=0.000125s
.Time to run block detection in LocalTransformer: num_layers=2, algo=detect_block_boundaries_binary, time=0.000108s
.Time to run block detection in LocalTransformer: num_layers=2, algo=detect_block_boundaries, time=0.000297s
.Time to run block detection in LocalTransformer: num_layers=4, algo=detect_block_boundaries_binary, time=0.000257s
.Time to run block detection in LocalTransformer: num_layers=4, algo=detect_block_boundaries, time=0.000770s
.Time to run block detection in LocalTransformer: num_layers=10, algo=detect_block_boundaries_binary, time=0.000738s
.Time to run block detection in LocalTransformer: num_layers=10, algo=detect_block_boundaries, time=0.002135s
.Time to run block detection in LocalTransformer: num_layers=50, algo=detect_block_boundaries_binary, time=0.004225s
.Time to run block detection in LocalTransformer: num_layers=50, algo=detect_block_boundaries, time=0.011872s
.
```