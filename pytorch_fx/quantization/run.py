import torch

import torch.nn as nn
from quant_params import QuantDtype


from quantization_simulate import QuantizationSimulator


# ===================== TOY MODEL AND DEMO =====================

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


class SimplerToyModel(nn.Module):
    """An even simpler model for clear demonstration."""
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 10)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class ModelWithAddMul(nn.Module):
    """Model demonstrating add/mul quantization."""
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(10, 20)
        self.linear3 = nn.Linear(20, 10)
    
    def forward(self, x):
        a = self.linear1(x)
        b = self.linear2(x)
        c = torch.add(a, b)  # Quantized add
        d = self.linear3(c)
        return d


def main():
    print("=" * 60)
    print("PyTorch FX Quantization Simulation Framework Demo")
    print("=" * 60)
    
    # Demo 1: Simple Linear Model
    print("\n--- Demo 1: Simple Linear Model ---")
    model1 = SimplerToyModel()
    model1.eval()
    
    simulator1 = QuantizationSimulator(model1, dtype=QuantDtype.INT8)
    
    # Prepare for observation
    observed_model = simulator1.prepare_for_observation()
    
    # Calibrate with some data
    calibration_data = [torch.randn(8, 10) for _ in range(10)]
    simulator1.calibrate(calibration_data)
    
    print(f"Collected stats for {len(simulator1.node_stats)} nodes")
    
    # Convert to quantized
    quantized_model = simulator1.convert_to_quantized()
    
    print("\nQuantized model graph:")
    print(quantized_model.graph)
    
    # Compare outputs
    test_input = torch.randn(4, 10)
    metrics = simulator1.compare_outputs(test_input)
    print(f"\nAccuracy comparison:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    
    # Demo 2: Model with Add operation
    print("\n" + "=" * 60)
    print("--- Demo 2: Model with Add Operation ---")
    model2 = ModelWithAddMul()
    model2.eval()
    
    simulator2 = QuantizationSimulator(model2, dtype=QuantDtype.INT8)
    observed_model2 = simulator2.prepare_for_observation()
    simulator2.calibrate(calibration_data)
    quantized_model2 = simulator2.convert_to_quantized()
    
    print("\nQuantized model graph:")
    print(quantized_model2.graph)
    
    metrics2 = simulator2.compare_outputs(test_input)
    print(f"\nAccuracy comparison:")
    for k, v in metrics2.items():
        print(f"  {k}: {v:.6f}")
    
    # Demo 3: Conv Model with unsupported op (pool)
    print("\n" + "=" * 60)
    print("--- Demo 3: Conv Model with Unsupported Op (AdaptiveAvgPool) ---")
    model3 = ToyModel()
    model3.eval()
    
    simulator3 = QuantizationSimulator(model3, dtype=QuantDtype.INT8)
    conv_example = torch.randn(1, 3, 32, 32)
    observed_model3 = simulator3.prepare_for_observation()
    
    conv_calibration = [torch.randn(4, 3, 32, 32) for _ in range(10)]
    simulator3.calibrate(conv_calibration)
    quantized_model3 = simulator3.convert_to_quantized()
    
    print("\nQuantized model graph:")
    print(quantized_model3.graph)
    
    test_conv_input = torch.randn(4, 3, 32, 32)
    metrics3 = simulator3.compare_outputs(test_conv_input)
    print(f"\nAccuracy comparison:")
    for k, v in metrics3.items():
        print(f"  {k}: {v:.6f}")
    
    # Show the quantization/dequantization pattern
    print("\n" + "=" * 60)
    print("Summary: Quantization Pattern")
    print("=" * 60)
    print("""
    For pattern A -> B -> C -> D where A, B, D are supported and C is not:
    
    Original:   A(float) -> B(float) -> C(float) -> D(float)
    
    Quantized:  quant -> A(int) -> B(int) -> dequant -> C(float) -> quant -> D(int) -> dequant
    
    This ensures:
    - Bit-exact integer operations for supported ops
    - Automatic insertion of quant/dequant at boundaries
    - Proper scale handling between consecutive quantized ops
    """)
    
    print("\nDone!")


if __name__ == "__main__":
    main()