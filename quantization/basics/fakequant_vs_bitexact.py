import numpy as np
import argparse

def get_qmin_qmax(nbits, symmetric):
    """Return (qmin, qmax) for given bitwidth and symmetric/asymmetric quantization."""
    if symmetric:
        qmin = - (2 ** (nbits - 1)) +1
        qmax = (2 ** (nbits - 1)) - 1
        assert np.abs(qmin) == np.abs(qmax)
    else:
        qmin = 0
        qmax = (2 ** nbits) - 1
    assert qmax > qmin
    return qmin, qmax

def quantize(x, scale, zero_point, symmetric):
    """Quantize float array x to uint8."""
    assert isinstance(zero_point, np.int8) or isinstance(zero_point, np.uint8), "zero_point must be np.int8 or np.uint8"
    qx = np.round(x / scale + zero_point.astype(np.float32)) # float32
    qmin, qmax = get_qmin_qmax(nbits=8, symmetric=symmetric)
    qx = np.clip(qx, qmin, qmax)
    if symmetric:
        qx = qx.astype(np.int8)
    else:
        qx = qx.astype(np.uint8)
    return qx

def dequantize(qx, scale, zero_point):
    """Dequantize uint8 array qx to float."""
    return scale * (qx.astype(np.float32) - zero_point.astype(np.float32))


def calc_scale_zero(min_val, max_val, symmetric):
    """
    Calculate scale and zero_point for quantization given min/max and qmin/qmax.
    Returns:
        scale (float), zero_point (int)
    """
    qmin, qmax = get_qmin_qmax(nbits=8, symmetric=symmetric)
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = int(np.round(qmin - min_val / scale))
    zero_point = np.clip(zero_point, qmin, qmax)
    if symmetric:
        assert zero_point == 0, f'For symmetric quantization, zero_point should be 0, but got {zero_point}'
        zero_point = zero_point.astype(np.int8)
    else:
        zero_point = zero_point.astype(np.uint8)
    return scale, zero_point



def will_overflow_int16_on_addition(a, b, dtype):
    assert dtype in [np.int16, np.int32], f"dtype must be np.int16 or np.int32, but got {dtype}"
    INT_MIN = np.iinfo(dtype).min
    INT_MAX = np.iinfo(dtype).max

    a = int(a)
    b = int(b)

    if a > 0 and b > 0:
        return a > INT_MAX - b
    if a < 0 and b < 0:
        return a < INT_MIN - b
    return False

def int_wide_matmul(a, b, widen_type):
    """
    Perform matrix multiplication where inputs are int8 and output is int32.
    Each multiplication is done in int8, accumulation in int32.
    a: np.ndarray of shape (M, K), dtype=int8
    b: np.ndarray of shape (K, N), dtype=int8
    Returns:
        np.ndarray of shape (M, N), dtype=int32
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    assert a.dtype == widen_type and b.dtype == widen_type, f"Inputs must be {widen_type}, but got {a.dtype} and {b.dtype}"
    out = np.zeros((M, N), dtype=widen_type)
    overflow = np.zeros((M, N), dtype=bool)
    for m in range(M):
        for n in range(N):
            acc = widen_type(0)
            for k in range(K):
                # Multiplication in int8, result upcast to int32 for accumulation
                mul = a[m, k]*b[k, n]
                if not overflow[m, n] and will_overflow_int16_on_addition(acc, mul, widen_type): # detect the first overflow, and print warning
                    #print(f"Warning: {widen_type} overflow in accumulation at ({m},{n}): {acc} + {mul}. Expected {np.int64(acc) + np.int64(mul)}, overflow after addn {acc + widen_type(mul)}, at k={k}")
                    overflow[m, n] = True
                acc += widen_type(mul)
            out[m, n] = acc
    print('Percent of output elements with overflow during accumulation:', 100.0 * np.sum(overflow) / (M*N))
    return out, overflow


scale = 5.0

# Quantization parameters
# why choose -3/3 ? torch.rand generates std normal, and ~99.7% values lie within +-3 stddev
x_min = -3.0*scale
x_max = 3.0*scale
w_min = -3.0*scale
w_max = 3.0*scale




def main(M, K, N, symmetric, num_iters, widen_type):
    if widen_type == 16:
        widen_type = np.int16
    else:
        widen_type = np.int32
    fakequant_vs_bitexact_max_diffs = []
    true_vs_fakequant_max_diffs = []
    true_vs_bitexact_max_diffs = []

    fakequant_vs_bitexact_mean_diffs = []
    true_vs_fakequant_mean_diffs = []
    true_vs_bitexact_mean_diffs = []
    for _ in range(num_iters):
        x = np.random.randn(M, K).astype(np.float32)*scale
        w = np.random.randn(K, N).astype(np.float32)*scale

        sx, zx = calc_scale_zero(x_min, x_max, symmetric)
        sw, zw = calc_scale_zero(w_min, w_max, symmetric)

        # Method 1: FakeQuant
        x1 = dequantize(quantize(x, sx, zx, symmetric=symmetric), sx, zx)
        w1 = dequantize(quantize(w, sw, zw, symmetric=symmetric), sw, zw)
        mm1 = np.matmul(x1, w1)

        # Method 2: Bit-exact
        xq = quantize(x, sx, zx, symmetric=symmetric)
        wq = quantize(w, sw, zw, symmetric=symmetric)

        xq_int32_centered = (xq.astype(widen_type) - int(zx))
        wq_int32_centered = (wq.astype(widen_type) - int(zw))
        mm_int32, overflow = int_wide_matmul(xq_int32_centered, wq_int32_centered, widen_type=widen_type)
        mm_1 = sx * sw * mm_int32

        # Actual result
        mm_true = np.matmul(x, w)

        # Compare results
        print("FakeQuant matmul result:\n", mm1)
        print("Bit-exact matmul result:\n", mm_1)
        print("True matmul result:\n", mm_true)
        print("Difference (max abs):", np.max(np.abs(mm1 - mm_1)))
        print("Difference (mean):", np.mean(np.abs(mm1 - mm_1)))
        print("FakeQuant vs True difference (max abs):", np.max(np.abs(mm1 - mm_true)))
        print("FakeQuant vs True difference (mean):", np.mean(np.abs(mm1 - mm_true)))
        print("Bit-exact vs True difference (max abs):", np.max(np.abs(mm_1 - mm_true)))
        print("Bit-exact vs True difference (mean):", np.mean(np.abs(mm_1 - mm_true)))
        print("-----")
        fakequant_vs_bitexact_max_diffs.append(np.max(np.abs(mm1 - mm_1)))
        true_vs_fakequant_max_diffs.append(np.max(np.abs(mm1 - mm_true)))
        true_vs_bitexact_max_diffs.append(np.max(np.abs(mm_1 - mm_true)))

        fakequant_vs_bitexact_mean_diffs.append(np.mean(np.abs(mm1 - mm_1)))
        true_vs_fakequant_mean_diffs.append(np.mean(np.abs(mm1 - mm_true)))
        true_vs_bitexact_mean_diffs.append(np.mean(np.abs(mm_1 - mm_true)))

    print("Average FakeQuant vs Bit-exact difference (max abs):", np.mean(fakequant_vs_bitexact_max_diffs))
    print("Average True vs FakeQuant difference (max abs):", np.mean(true_vs_fakequant_max_diffs))
    print("Average True vs Bit-exact difference (max abs):", np.mean(true_vs_bitexact_max_diffs))
    print("-----")
    print("Max FakeQuant vs Bit-exact difference (max abs):", np.max(fakequant_vs_bitexact_max_diffs))
    print("Max True vs FakeQuant difference (max abs):", np.max(true_vs_fakequant_max_diffs))
    print("Max True vs Bit-exact difference (max abs):", np.max(true_vs_bitexact_max_diffs))

    print("-----")
    print("-----")

    print("Average FakeQuant vs Bit-exact difference (mean abs):", np.mean(fakequant_vs_bitexact_mean_diffs))
    print("Average True vs FakeQuant difference (mean abs):", np.mean(true_vs_fakequant_mean_diffs))
    print("Average True vs Bit-exact difference (mean abs):", np.mean(true_vs_bitexact_mean_diffs))
    print("-----")
    print("Max FakeQuant vs Bit-exact difference (mean abs):", np.max(fakequant_vs_bitexact_mean_diffs))
    print("Max True vs FakeQuant difference (mean abs):", np.max(true_vs_fakequant_mean_diffs))
    print("Max True vs Bit-exact difference (mean abs):", np.max(true_vs_bitexact_mean_diffs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare FakeQuant and Bit-exact quantized matmul.")
    parser.add_argument('--symmetric', action='store_true', help='Use symmetric quantization (default: False)')
    parser.add_argument('--num-iters', type=int, default=10, help='Number of iterations (default: 10)')
    parser.add_argument('--M', type=int, default=40, help='Number of rows of input matrix x (default: 40)')
    parser.add_argument('--K', type=int, default=80, help='Number of columns of input matrix x / rows of weight matrix w (default: 80)')
    parser.add_argument('--N', type=int, default=40, help='Number of columns of weight matrix w (default: 40)')
    parser.add_argument('--widen-type', type=int, choices=[16, 32], default=32, help='Widening type for accumulation: 16 or 32 (default: 32)')
    args = parser.parse_args()
    main(args.M, args.K, args.N, args.symmetric, args.num_iters, args.widen_type)