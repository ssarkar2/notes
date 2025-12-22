
# Setup

A simple 3 layer conv model is setup to train on cifar10 dataset. Gelu is used as the non linearity. After training we replace the Gelu with different variants and see the effect on inference performance and accuracy

```bash
python train_model.py
```

## Variants:
1. Original model, with Gelu(approx='none')
2. Traced model
    1. with Gelu(approx='none')
    2. with Gelu(approx='tanh'): Expect it to be faster than 2.1, similar accuracy as 2.1
    $$
    0.5 \times x \times \left(1 + \tanh\left(\frac{2}{\pi} \times \left(x + 0.044715 \times x^3\right)\right)\right)
    $$
    3. Relu(): Expect it to be faster than 2.2, but bad accuracy
    4. Quickgelu: Expect it to be faster than 2.1, similar accuracy as 2.1
    $$
    x \times \sigma(1.702 \times x)
    $$

# Results

For MPS:

| Model Type | Non Linearity Type | Time Taken to Eval | Eval Accuracy |
|------------|--------------------|--------------------|---------------|
|     Original       |     Gelu(approx='none')               |       1.44             |         54.36%      |
|      Traced      |        Gelu(approx='none')            |          1.65          |      54.36%         |
|     Traced       |        Gelu(approx='tanh')            |           1.31         |         54.33%      |
|     Traced       |        Relu            |           1.17         |         27.59%      |
|     Traced       |        Quickgelu            |           1.61         |         54.22%      |


For CPU:

| Model Type | Non Linearity Type | Time Taken to Eval | Eval Accuracy |
|------------|--------------------|--------------------|---------------|
|     Original       |     Gelu(approx='none')               |       5.55             |         54.36%      |
|      Traced      |        Gelu(approx='none')            |          5.82          |      54.36%         |
|     Traced       |        Gelu(approx='tanh')            |           6.26         |         54.33%      |
|     Traced       |        Relu            |           5.25         |         27.59%      |
|     Traced       |        Quickgelu            |           5.67         |         54.22%      |

## Observations
1. Obviously Relu is the fastest, but the one with the worst accuracy (Expected)
2. The approximate gelus have about the same accuracy as the original (Expected)
3. MPS
    1. the approximate gelus are faster than the original gelu (Expected).
    2. Quickgelu is not as fast. This might be because there are dedicated fused kernels for gelu(none) and gelu(tanh)
4. CPU
    1. Gelu(tanh) is slower than Gelu(none). This maybe because erf is more optimized than tanh
    2. quickgelu is a bit faster than gelu(none) (Expected)


# Conclusions
Approximate Gelus might not actually be faster than the original. Depends on what fused kernels are available.
