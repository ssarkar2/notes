import torch
import torch.nn as nn


class FakeQuantBase:
    def __init__(self, num_bits=8, symmetric=True, min_val=None, max_val=None, scale=None, zero_point=None):
        self.num_bits = num_bits
        self.symmetric = symmetric

        if symmetric:
            self.qmin = -(2 ** (num_bits - 1)) + 1
            self.qmax = 2 ** (num_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** num_bits - 1

        min_max_specified = (min_val is not None and max_val is not None)
        scale_zp_specified = (scale is not None and zero_point is not None)
        assert min_max_specified or scale_zp_specified, "Either min_val/max_val or scale/zero_point must be specified."
        if min_max_specified == scale_zp_specified:
            raise ValueError("Specify either min_val/max_val or scale/zero_point, but not both.")

        if scale_zp_specified:
            assert isinstance(scale, (float, int)), "scale must be a float."
            self.scale = scale
            assert isinstance(zero_point, int), "zero_point must be an integer."
            self.zero_point = zero_point
            
            self.min_val = self.scale * (self.qmin - self.zero_point)
            self.max_val = self.scale * (self.qmax - self.zero_point)
        elif min_max_specified:
            assert isinstance(min_val, (float, int)), "min_val must be a float."
            assert isinstance(max_val, (float, int)), "max_val must be a float."
            self.min_val = min_val
            self.max_val = max_val
            if symmetric:
                largest_abs = max(abs(self.min_val), abs(self.max_val))
                self.min_val = -largest_abs
                self.max_val = largest_abs
            self.scale = (self.max_val - self.min_val) / (self.qmax - self.qmin)
            zero_point = round(self.qmin - self.min_val / self.scale)
            self.zero_point = max(self.qmin, min(self.qmax, zero_point))
        else:
            raise ValueError("Either min_val/max_val or scale/zero_point must be provided.")
        
        if symmetric:
            assert self.zero_point == 0, "For symmetric quantization, zero_point must be 0."


class FakeQuantize(FakeQuantBase, nn.Module):
    def __init__(self, num_bits=8, symmetric=True, min_val=None, max_val=None, scale=None, zero_point=None):
        nn.Module.__init__(self)
        FakeQuantBase.__init__(self, num_bits, symmetric, min_val, max_val, scale, zero_point)

    def forward(self, x):
        return ((x / self.scale) + self.zero_point).round().clamp(self.qmin, self.qmax)

class FakeDequantize(FakeQuantBase, nn.Module):
    def __init__(self, num_bits=8, symmetric=True, min_val=None, max_val=None, scale=None, zero_point=None):
        nn.Module.__init__(self)
        FakeQuantBase.__init__(self, num_bits, symmetric, min_val, max_val, scale, zero_point)

    def forward(self, x_int):
        return (x_int - self.zero_point) * self.scale


class QuantizedMatmul(nn.Module):
    def __init__(self, z_A, z_B):
        super().__init__()
        self.z_A = z_A
        self.z_B = z_B


    def forward(self, A_q, B_q):
        # note the zero points subtracted here. we cant just do a matmul, we need this zp adjustment as well
        return torch.matmul(A_q - self.z_A, B_q - self.z_B)

def test_fake_quant_dequant_matmul(symmetric, num_bits):
    # Create 2D float tensors
    A = torch.randn(4, 3)
    B = torch.randn(3, 5)

    # Golden reference
    C = torch.matmul(A, B)

    # Find min/max
    min_A, max_A = A.min().item(), A.max().item()
    min_B, max_B = B.min().item(), B.max().item()

    # Fake quantize A and B
    fq_A = FakeQuantize(symmetric=symmetric, num_bits=num_bits, min_val=min_A, max_val=max_A)
    fq_B = FakeQuantize(symmetric=symmetric, num_bits=num_bits, min_val=min_B, max_val=max_B)
    A_q = fq_A(A)
    B_q = fq_B(B)

    qm = QuantizedMatmul(fq_A.zero_point, fq_B.zero_point)

    # Matmul quantized tensors
    C_q = qm(A_q, B_q)

    # Fake dequantize result
    fd_C = FakeDequantize(symmetric=symmetric, num_bits=num_bits, scale=fq_A.scale * fq_B.scale, zero_point=0)
    C_dequant = fd_C(C_q)

    # Compare with golden reference
    print("Golden reference C:\n", C)
    print("Dequantized C:\n", C_dequant)
    error = torch.abs(C - C_dequant).mean().item()
    print("Mean absolute error:", error)
    tol_dict = {16: 0.001, 8: 0.1, 4: 0.5}
    assert error < tol_dict[num_bits], "Dequantized result deviates significantly from reference."

if __name__ == "__main__":
    test_fake_quant_dequant_matmul(symmetric=True, num_bits=16)
    test_fake_quant_dequant_matmul(symmetric=False, num_bits=16)
    test_fake_quant_dequant_matmul(symmetric=True, num_bits=8)
    test_fake_quant_dequant_matmul(symmetric=False, num_bits=8)
    test_fake_quant_dequant_matmul(symmetric=True, num_bits=4)
    test_fake_quant_dequant_matmul(symmetric=False, num_bits=4)