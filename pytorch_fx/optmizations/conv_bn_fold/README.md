# Setup

Running inference on resnet 18 for 10000 images for some variants like:
1. Original model
2. Original model, fx traced
3. Original model, fx traced, torch compiled
4. fx traced model with batch norm weights fused into conv
5. fx traced model with batch norm weights fused into conv, and torch compiled
```bash
HF_TOKEN=xxx python resnet_conv_bn_fold.py
```

Note we need a HF_TOKEN with approprate permissions as the dataset is gated. In all cases the accuracy is 70.41%.


# Results

| Fused bn? | traced? | time per image | 
|------------|--------------------|--------------------|
|     Not fused       |     None   |       0.017031           |
|     Not fused       |     fx traced   |       0.017173             |
|     Not fused       |    torch compiled   |       0.010710      |
|     Fused       |    fx traced   |       0.011603      |
|     Fused       |    torch compiled   |      0.010871      |


The perf numbers are for MPS backend in torch. We can see a large speedup in the fx traced models with and without the batchnorm fusion. The torch compiled times are similar for both, because this basic optimization is being done in both cases in the background.