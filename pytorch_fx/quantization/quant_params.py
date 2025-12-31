from enum import Enum
from dataclasses import dataclass


@dataclass
class TensorStats:
    min_val: float
    max_val: float


class QuantDtype(Enum):
    INT8 = 8
    INT16 = 16

@dataclass
class QuantParams:
    scale: float
    zero_point: int
    dtype: QuantDtype = QuantDtype.INT8
    
    @property
    def qmin(self):
        bits = self.dtype.value
        return -(2 ** (bits - 1))
    
    @property
    def qmax(self):
        bits = self.dtype.value
        return 2 ** (bits - 1) - 1
