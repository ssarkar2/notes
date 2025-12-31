
import torch.nn as nn
from quant_params import QuantParams
import torch


# TODO, move some stuff to a base class?
# TODO requant is now done in Int model, we could attach a specific requant module

# TODO ensure inp scales are same. also requant out
class IntAdd1(nn.Module):
    """Integer addition with rescaling."""
    
    def __init__(self, a_params: QuantParams, b_params: QuantParams, out_params: QuantParams):
        super().__init__()
        self.a_scale = a_params.scale
        self.b_scale = b_params.scale
        self.out_scale = out_params.scale
        self.out_zp = out_params.zero_point
        self.qmin = out_params.qmin
        self.qmax = out_params.qmax
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Perform add in higher precision, then requantize
        assert a.dtype == torch.int8 and b.dtype == torch.int8, "Inputs must be int8 tensors"
        a_scaled = a.to(torch.int32) * round(self.a_scale / self.out_scale * 1024)
        b_scaled = b.to(torch.int32) * round(self.b_scale / self.out_scale * 1024)
        result = (a_scaled + b_scaled) // 1024
        result = torch.clamp(result + self.out_zp, self.qmin, self.qmax)
        return result.to(torch.int32)



class IntAdd(nn.Module):
    """
    Integer addition of two quantized tensors.
    Handles different input scales by rescaling to a common int32 domain.
    Requantizes to output quantization parameters.
    """

    def __init__(self, a_params, b_params, out_params):
        super().__init__()
        self.a_zp = a_params.zero_point
        self.b_zp = b_params.zero_point
        self.out_zp = out_params.zero_point

        self.a_scale = a_params.scale
        self.b_scale = b_params.scale
        self.out_scale = out_params.scale

        self.qmin = out_params.qmin
        self.qmax = out_params.qmax

    def forward(self, a_int: torch.Tensor, b_int: torch.Tensor) -> torch.Tensor:
        assert a_int.dtype == torch.int8 and b_int.dtype == torch.int8, "Inputs must be int8 tensors"

        # move b to a's scale:
        b_int_rescaled = (self.b_scale / self.a_scale)*(b_int.to(torch.int32) - self.b_zp) + self.a_zp
        y_int32 = a_int.to(torch.int32) - self.a_zp + b_int_rescaled.to(torch.int32) - self.b_zp

        scale = self.a_scale / self.out_scale
        y_rescaled = torch.round(y_int32.float() * scale).to(torch.int32)

        y_rescaled = y_rescaled + self.out_zp
        y_rescaled = torch.clamp(y_rescaled, self.qmin, self.qmax)

        return y_rescaled.to(torch.int8)


        # Convert to int32
        a_int32 = a_int.to(torch.int32) - self.a_zp
        b_int32 = b_int.to(torch.int32) - self.b_zp

        # Rescale inputs to output scale
        a_rescaled = torch.round(a_int32.float() * (self.a_scale / self.out_scale)).to(torch.int32)
        b_rescaled = torch.round(b_int32.float() * (self.b_scale / self.out_scale)).to(torch.int32)

        # Integer addition
        y_int32 = a_rescaled + b_rescaled

        # Add output zero-point and clamp
        y_int32 = y_int32 + self.out_zp
        y_int32 = torch.clamp(y_int32, self.qmin, self.qmax)

        return y_int32

class IntSub1(nn.Module):
    """Integer subtraction with rescaling."""
    
    def __init__(self, a_params: QuantParams, b_params: QuantParams, out_params: QuantParams):
        super().__init__()
        self.a_scale = a_params.scale
        self.b_scale = b_params.scale
        self.out_scale = out_params.scale
        self.out_zp = out_params.zero_point
        self.qmin = out_params.qmin
        self.qmax = out_params.qmax
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_scaled = a.to(torch.int64) * round(self.a_scale / self.out_scale * 1024)
        b_scaled = b.to(torch.int64) * round(self.b_scale / self.out_scale * 1024)
        result = (a_scaled - b_scaled) // 1024
        result = torch.clamp(result + self.out_zp, self.qmin, self.qmax)
        return result.to(torch.int32)


class IntSub(nn.Module):
    """
    Integer subtraction using IntAdd.
    Computes: y = a - b
    """

    def __init__(self, a_params, b_params, out_params):
        super().__init__()
        self.int_add = IntAdd(a_params, b_params, out_params)
        self.b_params = b_params

    def forward(self, a_int: torch.Tensor, b_int: torch.Tensor) -> torch.Tensor:
        # Negate b relative to its zero-point
        # b_int is uint/int8 with zero-point, so negation around zero-point:
        b_neg = (2 * self.b_params.zero_point - b_int).to(torch.int8)
        # Use IntAdd to compute a + (-b)
        y = self.int_add(a_int, b_neg)
        return y


class IntMul1(nn.Module):
    """Integer multiplication with rescaling."""
    
    def __init__(self, a_params: QuantParams, b_params: QuantParams, out_params: QuantParams):
        super().__init__()
        self.combined_scale = (a_params.scale * b_params.scale) / out_params.scale
        self.out_zp = out_params.zero_point
        self.qmin = out_params.qmin
        self.qmax = out_params.qmax
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        result = a.to(torch.int64) * b.to(torch.int64)
        # Scale down using fixed point arithmetic
        scale_int = round(self.combined_scale * 1024)
        result = (result * scale_int) // 1024
        result = torch.clamp(result + self.out_zp, self.qmin, self.qmax)
        return result.to(torch.int32)



class IntMul(nn.Module):
    """
    Integer multiplication of two quantized tensors.
    Handles different input scales by converting to float, multiplies in int32,
    and requantizes to output quantization parameters.
    """

    def __init__(self, a_params, b_params, out_params):
        super().__init__()
        self.a_zp = a_params.zero_point
        self.b_zp = b_params.zero_point
        self.out_zp = out_params.zero_point

        self.a_scale = a_params.scale
        self.b_scale = b_params.scale
        self.out_scale = out_params.scale

        self.qmin = out_params.qmin
        self.qmax = out_params.qmax

    def forward(self, a_int: torch.Tensor, b_int: torch.Tensor) -> torch.Tensor:
        # Convert inputs to int32 and subtract zero points
        a_int32 = (a_int.to(torch.int32) - self.a_zp)
        b_int32 = (b_int.to(torch.int32) - self.b_zp)

        # Multiply (int32)
        y_int32 = a_int32 * b_int32  # int32 multiplication

        # Requantize to output scale
        # Output scale = input_a_scale * input_b_scale / out_scale
        scale = (self.a_scale * self.b_scale) / self.out_scale
        y_rescaled = torch.round(y_int32.float() * scale).to(torch.int32)

        # Add output zero-point and clamp
        y_rescaled = y_rescaled + self.out_zp
        y_rescaled = torch.clamp(y_rescaled, self.qmin, self.qmax)

        return y_rescaled.to(torch.int8)


class IntLinear(nn.Module):
    """Integer linear layer."""
    
    def __init__(self, float_linear: nn.Linear, input_params: QuantParams, 
                 weight_params: QuantParams, output_params: QuantParams):
        super().__init__()
        # Quantize weights
        self.weight_int = nn.Parameter(
            torch.round(float_linear.weight.data / weight_params.scale).to(torch.int8),
            requires_grad=False
        )
        if float_linear.bias is not None:
            bias_scale = input_params.scale * weight_params.scale
            self.bias_int = nn.Parameter(
                torch.round(float_linear.bias.data / bias_scale).to(torch.int32),
                requires_grad=False
            )
        else:
            self.register_parameter('bias_int', None)
        
        self.input_scale = input_params.scale
        self.input_zp = input_params.zero_point
        self.weight_scale = weight_params.scale
        self.weight_zp = weight_params.zero_point
        self.output_scale = output_params.scale
        self.output_zp = output_params.zero_point
        self.qmin = output_params.qmin
        self.qmax = output_params.qmax
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Integer matmul
        assert x.dtype == torch.int8, "Input must be int8 tensor"

        result = torch.nn.functional.linear(
            (x.to(torch.int32) - self.input_zp), 
            (self.weight_int.to(torch.int32) - self.weight_zp),
            self.bias_int
        )
        # Rescale to output quantization
        combined_scale = (self.input_scale * self.weight_scale) / self.output_scale
        scale_int = round(combined_scale * 1024)
        result = (result.to(torch.int64) * scale_int) // 1024
        result = torch.clamp(result + self.output_zp, self.qmin, self.qmax)
        return result.to(torch.int8)


class IntConv2d(nn.Module):
    """Integer 2D convolution."""
    
    def __init__(self, float_conv: nn.Conv2d, input_params: QuantParams,
                 weight_params: QuantParams, output_params: QuantParams):
        super().__init__()
        self.weight_int = nn.Parameter(
            torch.round(float_conv.weight.data / weight_params.scale).to(torch.int8),
            requires_grad=False
        )
        if float_conv.bias is not None:
            bias_scale = input_params.scale * weight_params.scale
            self.bias_int = nn.Parameter(
                torch.round(float_conv.bias.data / bias_scale).to(torch.int32),
                requires_grad=False
            )
        else:
            self.register_parameter('bias_int', None)
        
        self.stride = float_conv.stride
        self.padding = float_conv.padding
        self.dilation = float_conv.dilation
        self.groups = float_conv.groups
        
        self.input_scale = input_params.scale
        self.weight_scale = weight_params.scale
        self.input_zp = input_params.zero_point
        self.weight_zp = weight_params.zero_point
        self.output_scale = output_params.scale
        self.output_zp = output_params.zero_point
        self.qmin = output_params.qmin
        self.qmax = output_params.qmax
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == torch.int8, "Input must be int8 tensor"
        result = torch.nn.functional.conv2d(
            (x.to(torch.int32) - self.input_zp),
            (self.weight_int.to(torch.int32) - self.weight_zp),
            self.bias_int,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )
        combined_scale = (self.input_scale * self.weight_scale) / self.output_scale
        scale_int = round(combined_scale * 1024)
        result = (result.to(torch.int64) * scale_int) // 1024
        result = torch.clamp(result + self.output_zp, self.qmin, self.qmax)
        return result.to(torch.int8)


# for symmetric, we can just use nn.Relu. for asymmetric, need to shift by zero_point
class IntReLU(nn.Module):
    """Integer ReLU - just clamps negative values to zero."""
    
    def __init__(self, zero_point: int = 0):
        super().__init__()
        self.zero_point = zero_point
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=self.zero_point).to(torch.int8)

