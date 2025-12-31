import torch
import torch.nn as nn
from typing import Optional
from quant_params import TensorStats

class ObserverModule(nn.Module):
    """Module to observe min/max of tensors during calibration."""
    
    def __init__(self):
        super().__init__()
        self.min_val: Optional[float] = None
        self.max_val: Optional[float] = None
        self.enabled = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            with torch.no_grad():
                curr_min = x.min().item()
                curr_max = x.max().item()
                if self.min_val is None:
                    self.min_val = curr_min
                    self.max_val = curr_max
                else:
                    self.min_val = min(self.min_val, curr_min)
                    self.max_val = max(self.max_val, curr_max)
        return x
    
    def get_stats(self) -> Optional[TensorStats]:
        if self.min_val is not None and self.max_val is not None:
            return TensorStats(self.min_val, self.max_val)
        return None
