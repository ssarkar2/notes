import operator
import torch
import torch.fx as fx
from enum import Enum, auto

class Parity(Enum):
    EVEN = auto()
    ODD = auto()
    UNKNOWN = auto()

def add_parity(a, b):
    if Parity.UNKNOWN in (a, b):
        return Parity.UNKNOWN
    return Parity.EVEN if a == b else Parity.ODD

def mul_parity(a, b):
    if a == Parity.EVEN or b == Parity.EVEN:
        return Parity.EVEN
    if a == Parity.ODD and b == Parity.ODD:
        return Parity.ODD
    return Parity.UNKNOWN


class AbsInterpreter(fx.Interpreter):
    def __init__(self, gm, input_parity):
        super().__init__(gm)
        self.input_parity = input_parity
        self._current_node = None

    # Override run_node to keep track of the current node
    def run_node(self, node):
        self._current_node = node
        return super().run_node(node)

    def placeholder(self, target, args, kwargs):
        parity = self.input_parity[target]
        self._current_node.meta["parity"] = parity
        return parity
    
    # TODO override get_attr to handle tensors coming from init (self.x)
    # for simple ints, it works (see MyModel.b below)

    def call_function(self, target, args, kwargs):
        left, right = args
        # if left/right are ints, convert to Parity
        if isinstance(left, int):
            left = Parity.EVEN if left % 2 == 0 else Parity.ODD
        if isinstance(right, int):
            right = Parity.EVEN if right % 2 == 0 else Parity.ODD
        if target in (operator.add, torch.add):
            parity = add_parity(left, right)
        elif target in (operator.sub, torch.sub):
            parity = add_parity(left, right)
        elif target in (operator.mul, torch.mul):
            parity = mul_parity(left, right)
        else:
            parity = Parity.UNKNOWN

        self._current_node.meta["parity"] = parity
        return parity

    # TODO call_method might need to be overridded too to handle __add__, __sub__, __mul__ etc

    def output(self, target, args, kwargs):
        return args[0] # abstract value computed by the previous node
    
if __name__ == "__main__":
    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.b = 5
        def forward(self, x, y):
            a = x + y
            b = a * x
            c = a - b
            d = c + 4
            return d + self.b
        

    gm = fx.symbolic_trace(MyModel())

    print(gm.graph)

    for x_parity, y_parity in [
        (Parity.EVEN, Parity.EVEN),
        (Parity.EVEN, Parity.ODD),
        (Parity.ODD, Parity.EVEN),
        (Parity.ODD, Parity.ODD),
        (Parity.UNKNOWN, Parity.ODD),
    ]:
        print(f"Running abstract interpretation with x: {x_parity}, y: {y_parity}")

        interp = AbsInterpreter(
            gm,
            input_parity={
                "x": x_parity,
                "y": y_parity,
            },
        )

        result = interp.run()
        print("Final parity:", result)

        for n in gm.graph.nodes:
            print(n.name, "→", n.meta.get("parity"))

        if x_parity == Parity.UNKNOWN or y_parity == Parity.UNKNOWN:
            concrete_parity = Parity.UNKNOWN
        else:
            concrete_x = 2 if x_parity == Parity.EVEN else 3
            concrete_y = 4 if y_parity == Parity.EVEN else 5
            concrete_result = gm(concrete_x, concrete_y)
            concrete_parity = Parity.EVEN if concrete_result % 2 == 0 else Parity.ODD
        print("Concrete result:", concrete_result, "→", concrete_parity)
        assert concrete_parity == result, f"{concrete_parity} != {result}"
        print('-----')
