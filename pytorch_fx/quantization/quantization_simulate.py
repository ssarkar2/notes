
import torch
from torch.fx import symbolic_trace
from typing import Dict, Set, Tuple, Optional, List
from dataclasses import dataclass

import torch.nn as nn
import torch.fx as fx
from int_modules import IntLinear, IntConv2d, IntReLU, IntAdd, IntSub, IntMul
from quant_params import QuantParams, QuantDtype, TensorStats
from observer import ObserverModule
import graphviz
import operator

# Doing symmetric quantization for simplicity
def compute_quant_params(stats: TensorStats, dtype: QuantDtype = QuantDtype.INT8) -> QuantParams:
    """Compute scale and zero_point for symmetric quantization."""
    bits = dtype.value
    qmax = 2 ** (bits - 1) - 1
    
    max_abs = max(abs(stats.min_val), abs(stats.max_val))
    if max_abs == 0:
        max_abs = 1e-8
    
    scale = max_abs / qmax
    zero_point = 0  # Symmetric quantization
    
    return QuantParams(scale=scale, zero_point=zero_point, dtype=dtype)


# Supported operations for quantization
SUPPORTED_OPS = {
    operator.add,
    operator.sub,
    operator.mul,
    torch.add,
    torch.sub,
    torch.mul,
    torch.nn.functional.linear,
    torch.nn.functional.conv2d,
    torch.nn.functional.relu,
    torch.nn.functional.max_pool2d,
    'add',
    'sub',
    'mul',
}

SUPPORTED_MODULES = {
    nn.Linear,
    nn.Conv2d,
    nn.ReLU,
    nn.MaxPool2d,
}


def is_quantizable_node(node: fx.Node, modules: Dict[str, nn.Module]) -> bool:
    """Check if a node can be quantized."""
    if node.op == 'call_function':
        return node.target in SUPPORTED_OPS
    elif node.op == 'call_method':
        return node.target in SUPPORTED_OPS
    elif node.op == 'call_module':
        module = modules.get(node.target)
        return type(module) in SUPPORTED_MODULES
    return False


class QuantizeTensor(nn.Module):
    """Quantize float tensor to int tensor."""
    
    def __init__(self, quant_params: QuantParams):
        super().__init__()
        self.scale = quant_params.scale
        self.zero_point = quant_params.zero_point
        self.qmin = quant_params.qmin
        self.qmax = quant_params.qmax
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_int = torch.round(x / self.scale).to(torch.int8) + self.zero_point
        x_int = torch.clamp(x_int, self.qmin, self.qmax)
        return x_int


class DequantizeTensor(nn.Module):
    """Dequantize int tensor back to float tensor."""
    
    def __init__(self, quant_params: QuantParams):
        super().__init__()
        self.scale = quant_params.scale
        self.zero_point = quant_params.zero_point
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x.to(torch.float32) - self.zero_point) * self.scale)




class QuantizationSimulator:
    """Main framework for simulating quantization accuracy loss."""
    
    def __init__(self, model: nn.Module, dtype: QuantDtype = QuantDtype.INT8):
        self.original_model = model
        self.dtype = dtype
        self.traced_model: Optional[fx.GraphModule] = None
        self.observed_model: Optional[fx.GraphModule] = None
        self.quantized_model: Optional[fx.GraphModule] = None
        self.node_stats: Dict[str, TensorStats] = {}
        self.node_quant_params: Dict[str, QuantParams] = {}
        self.observers: Dict[str, ObserverModule] = {}
        self._quant_module_idx = 0
    
    def _trace_model(self) -> fx.GraphModule:
        """Trace the model using FX."""
        if isinstance(self.original_model, fx.GraphModule):
            return self.original_model
        traced = symbolic_trace(self.original_model)
        return traced
    
    def prepare_for_observation(self, ) -> fx.GraphModule:
        """Insert observers after each node to collect min/max statistics."""
        self.traced_model = self._trace_model()
        
        # Create a new graph with observers
        new_graph = fx.Graph()
        env: Dict[fx.Node, fx.Node] = {}
        modules = dict(self.traced_model.named_modules())
        
        observer_idx = 0
        
        for node in self.traced_model.graph.nodes:
            new_node = new_graph.node_copy(node, lambda x: env[x])
            env[node] = new_node
            
            # Insert observer after meaningful operations
            if node.op in ['call_function', 'call_method', 'call_module', 'placeholder']:
                observer_name = f'observer_{observer_idx}'
                observer = ObserverModule()
                self.observers[observer_name] = observer
                modules[observer_name] = observer
                
                observer_node = new_graph.call_module(observer_name, (new_node,))
                env[node] = observer_node
                observer_idx += 1
        
        new_graph.lint()
        self.observed_model = fx.GraphModule(modules, new_graph)
        return self.observed_model
    
    def calibrate(self, calibration_data: List[Tuple[torch.Tensor, ...]]):
        """Run calibration data through observed model to collect statistics."""
        if self.observed_model is None:
            raise RuntimeError("Must call prepare_for_observation first")
        
        self.observed_model.eval()
        with torch.no_grad():
            for inputs in calibration_data:
                if isinstance(inputs, torch.Tensor):
                    inputs = (inputs,)
                self.observed_model(*inputs)
        
        # Collect statistics
        for name, observer in self.observers.items():
            stats = observer.get_stats()
            if stats is not None:
                self.node_stats[name] = stats
                self.node_quant_params[name] = compute_quant_params(stats, self.dtype)
    
    def _find_quantizable_regions(self) -> List[List[fx.Node]]:
        """Find contiguous regions of quantizable operations."""
        if self.traced_model is None:
            raise RuntimeError("Model not traced")
        
        modules = dict(self.traced_model.named_modules())
        
        return set(node for node in self.traced_model.graph.nodes if is_quantizable_node(node, modules))
    
    def convert_to_quantized(self) -> fx.GraphModule:
        """Convert model to use integer operations where possible."""
        if self.traced_model is None or not self.node_quant_params:
            raise RuntimeError("Must trace and calibrate first")
        
        modules = dict(self.traced_model.named_modules())
        quantizable_nodes = self._find_quantizable_regions()
        
        new_graph = fx.Graph()
        env: Dict[fx.Node, fx.Node] = {}
        new_modules = dict(modules)
        
        # Track which nodes output quantized (int) tensors
        is_quantized: Dict[fx.Node, bool] = {}
        node_to_observer: Dict[fx.Node, str] = {}
        
        # Map nodes to their observers
        observer_idx = 0
        for node in self.traced_model.graph.nodes:
            if node.op in ['call_function', 'call_method', 'call_module', 'placeholder']:
                node_to_observer[node] = f'observer_{observer_idx}'
                observer_idx += 1
        
        def get_quant_params(node: fx.Node) -> Optional[QuantParams]:
            obs_name = node_to_observer.get(node)
            return self.node_quant_params.get(obs_name)
        
        def add_quantize(input_node: fx.Node, params: QuantParams) -> fx.Node:
            name = f'quantize_{self._quant_module_idx}'
            self._quant_module_idx += 1
            new_modules[name] = QuantizeTensor(params)
            return new_graph.call_module(name, (input_node,))
        
        def add_dequantize(input_node: fx.Node, params: QuantParams) -> fx.Node:
            name = f'dequantize_{self._quant_module_idx}'
            self._quant_module_idx += 1
            new_modules[name] = DequantizeTensor(params)
            return new_graph.call_module(name, (input_node,))
        
        for node in self.traced_model.graph.nodes:
            if node.op == 'placeholder':
                new_node = new_graph.node_copy(node, lambda x: env[x])
                env[node] = new_node
                is_quantized[new_node] = False
                
            elif node.op == 'output':
                # Ensure output is dequantized
                args = []
                for arg in node.args[0] if isinstance(node.args[0], tuple) else [node.args[0]]:
                    if isinstance(arg, fx.Node):
                        mapped = env[arg]
                        if is_quantized.get(mapped, False):
                            params = get_quant_params(arg)
                            if params:
                                mapped = add_dequantize(mapped, params)
                                is_quantized[mapped] = False
                        args.append(mapped)
                    else:
                        args.append(arg)
                new_graph.output(args[0] if len(args) == 1 else tuple(args))
                
            elif node in quantizable_nodes:
                # This is a quantizable operation
                out_params = get_quant_params(node)
                
                if node.op == 'call_module':
                    module = modules[node.target]
                    
                    # Get input and ensure it's quantized
                    input_arg = node.args[0]
                    input_node = env[input_arg]
                    input_params = get_quant_params(input_arg)
                    
                    if not is_quantized.get(input_node, False) and input_params:
                        input_node = add_quantize(input_node, input_params)
                        is_quantized[input_node] = True
                    
                    if isinstance(module, nn.Linear):
                        # Create weight quant params
                        weight_stats = TensorStats(
                            module.weight.min().item(),
                            module.weight.max().item()
                        )
                        weight_params = compute_quant_params(weight_stats, self.dtype)
                        
                        int_name = f'int_linear_{self._quant_module_idx}'
                        self._quant_module_idx += 1
                        new_modules[int_name] = IntLinear(module, input_params, weight_params, out_params)
                        new_node = new_graph.call_module(int_name, (input_node,))
                        is_quantized[new_node] = True
                        
                    elif isinstance(module, nn.Conv2d):
                        weight_stats = TensorStats(
                            module.weight.min().item(),
                            module.weight.max().item()
                        )
                        weight_params = compute_quant_params(weight_stats, self.dtype)
                        
                        int_name = f'int_conv2d_{self._quant_module_idx}'
                        self._quant_module_idx += 1
                        new_modules[int_name] = IntConv2d(module, input_params, weight_params, out_params)
                        new_node = new_graph.call_module(int_name, (input_node,))
                        is_quantized[new_node] = True
                        
                    elif isinstance(module, nn.ReLU):
                        int_name = f'int_relu_{self._quant_module_idx}'
                        self._quant_module_idx += 1
                        zp = input_params.zero_point if input_params else 0
                        new_modules[int_name] = IntReLU(zp)
                        new_node = new_graph.call_module(int_name, (input_node,))
                        is_quantized[new_node] = True
                    elif isinstance(module, nn.MaxPool2d):
                        int_name = f'int_maxpool2d_{self._quant_module_idx}'
                        self._quant_module_idx += 1
                        #zp = input_params.zero_point if input_params else 0
                        new_modules[int_name] = nn.MaxPool2d(module.kernel_size, module.stride, module.padding, module.dilation, module.return_indices, module.ceil_mode)
                        new_node = new_graph.call_module(int_name, (input_node,))
                        is_quantized[new_node] = True
                        # we can just retain the old module since maxpool doesn't change for float vs int
                    else:
                        new_node = new_graph.node_copy(node, lambda x: env[x])
                        is_quantized[new_node] = False
                    
                    env[node] = new_node
                    
                elif node.op == 'call_function':
                    # Handle binary ops like add, sub, mul
                    #or node.target in [operator.add, operator.sub, operator.mul]
                    if node.target in [torch.add, torch.sub, torch.mul] or node.target in [operator.add, operator.sub, operator.mul] \
                       or (hasattr(node.target, '__name__') and node.target.__name__ in ['add', 'sub', 'mul']):
                        
                        a_arg, b_arg = node.args[0], node.args[1]
                        a_node = env[a_arg] if isinstance(a_arg, fx.Node) else a_arg
                        b_node = env[b_arg] if isinstance(b_arg, fx.Node) else b_arg
                        
                        a_params = get_quant_params(a_arg) if isinstance(a_arg, fx.Node) else None
                        b_params = get_quant_params(b_arg) if isinstance(b_arg, fx.Node) else None
                        
                        # Quantize inputs if needed
                        if isinstance(a_arg, fx.Node) and not is_quantized.get(a_node, False) and a_params:
                            a_node = add_quantize(a_node, a_params)
                            is_quantized[a_node] = True
                        
                        if isinstance(b_arg, fx.Node) and not is_quantized.get(b_node, False) and b_params:
                            b_node = add_quantize(b_node, b_params)
                            is_quantized[b_node] = True
                        
                        if a_params and b_params and out_params:
                            if node.target == torch.add or (hasattr(node.target, '__name__') and node.target.__name__ == 'add'):
                                int_name = f'int_add_{self._quant_module_idx}'
                                new_modules[int_name] = IntAdd(a_params, b_params, out_params)
                            elif node.target == torch.sub or (hasattr(node.target, '__name__') and node.target.__name__ == 'sub'):
                                int_name = f'int_sub_{self._quant_module_idx}'
                                new_modules[int_name] = IntSub(a_params, b_params, out_params)
                            else:  # mul
                                int_name = f'int_mul_{self._quant_module_idx}'
                                new_modules[int_name] = IntMul(a_params, b_params, out_params)
                            
                            self._quant_module_idx += 1
                            new_node = new_graph.call_module(int_name, (a_node, b_node))
                            is_quantized[new_node] = True
                        else:
                            new_node = new_graph.node_copy(node, lambda x: env[x])
                            is_quantized[new_node] = False
                        
                        env[node] = new_node
                        
                    elif node.target == torch.nn.functional.relu:
                        input_arg = node.args[0]
                        input_node = env[input_arg]
                        input_params = get_quant_params(input_arg)
                        
                        if not is_quantized.get(input_node, False) and input_params:
                            input_node = add_quantize(input_node, input_params)
                            is_quantized[input_node] = True
                        
                        int_name = f'int_relu_{quant_module_idx}'
                        quant_module_idx += 1
                        zp = input_params.zero_point if input_params else 0
                        new_modules[int_name] = IntReLU(zp)
                        new_node = new_graph.call_module(int_name, (input_node,))
                        is_quantized[new_node] = True
                        env[node] = new_node
                    elif node.target == torch.nn.functional.max_pool2d:
                        assert False, "MaxPool2d quantization for functional not implemented yet"
                    else:
                        new_node = new_graph.node_copy(node, lambda x: env[x])
                        is_quantized[new_node] = False
                        env[node] = new_node
                else:
                    new_node = new_graph.node_copy(node, lambda x: env[x])
                    is_quantized[new_node] = False
                    env[node] = new_node
                    
            else:
                # Non-quantizable operation - need to dequantize inputs
                new_args = []
                for arg in node.args:
                    if isinstance(arg, fx.Node):
                        mapped = env[arg]
                        if is_quantized.get(mapped, False):
                            params = get_quant_params(arg)
                            if params:
                                mapped = add_dequantize(mapped, params)
                                is_quantized[mapped] = False
                        new_args.append(mapped)
                    elif isinstance(arg, (list, tuple)):
                        new_inner = []
                        for a in arg:
                            if isinstance(a, fx.Node):
                                m = env[a]
                                if is_quantized.get(m, False):
                                    p = get_quant_params(a)
                                    if p:
                                        m = add_dequantize(m, p)
                                        is_quantized[m] = False
                                new_inner.append(m)
                            else:
                                new_inner.append(a)
                        new_args.append(type(arg)(new_inner))
                    else:
                        new_args.append(arg)
                
                new_kwargs = {}
                for k, v in node.kwargs.items():
                    if isinstance(v, fx.Node):
                        mapped = env[v]
                        if is_quantized.get(mapped, False):
                            params = get_quant_params(v)
                            if params:
                                mapped = add_dequantize(mapped, params)
                                is_quantized[mapped] = False
                        new_kwargs[k] = mapped
                    else:
                        new_kwargs[k] = v
                
                with new_graph.inserting_after(list(new_graph.nodes)[-1] if new_graph.nodes else None):
                    if node.op == 'call_function':
                        new_node = new_graph.call_function(node.target, tuple(new_args), new_kwargs)
                    elif node.op == 'call_method':
                        new_node = new_graph.call_method(node.target, tuple(new_args), new_kwargs)
                    elif node.op == 'call_module':
                        new_node = new_graph.call_module(node.target, tuple(new_args), new_kwargs)
                    elif node.op == 'get_attr':
                        new_node = new_graph.get_attr(node.target)
                    else:
                        new_node = new_graph.node_copy(node, lambda x: env.get(x, x))
                
                is_quantized[new_node] = False
                env[node] = new_node
        
        new_graph.lint()
        self.quantized_model = fx.GraphModule(new_modules, new_graph)

        self.dump_quantized_graph_dot()
        return self.quantized_model
    
    def dump_quantized_graph_dot(self, file_prefix: str = None) -> str:
            """Dump the current quantized fx graph to a .dot file and optionally render a .png.
            Returns the path to the written file (png if rendered, otherwise .dot).
            """
            if file_prefix is None:
                file_prefix = type(self.original_model).__name__
            if self.quantized_model is None:
                raise RuntimeError("quantized_model is None. Call convert_to_quantized first.")
            graph = self.quantized_model.graph

            def _iter_nodes(obj):
                if isinstance(obj, fx.Node):
                    yield obj
                elif isinstance(obj, (list, tuple)):
                    for o in obj:
                        yield from _iter_nodes(o)
                elif isinstance(obj, dict):
                    for o in obj.values():
                        yield from _iter_nodes(o)

            lines = ['digraph G {', '  rankdir=LR;']
            for n in graph.nodes:
                label = f"{n.op}\\n{repr(n.target)}"
                # make sure label is safe for DOT
                label = label.replace('"', "'")
                try:
                    is_qdq = isinstance(self.quantized_model.get_submodule(n.target), QuantizeTensor) or isinstance(self.quantized_model.get_submodule(n.target), DequantizeTensor)
                except Exception:
                    is_qdq = False
                shape = 'oval' if is_qdq else 'box'
                lines.append(f'  "{n.name}" [label="{label}", shape={shape}];')

            for n in graph.nodes:
                for src in _iter_nodes(n.args):
                    lines.append(f'  "{src.name}" -> "{n.name}";')
                for v in (n.kwargs.values() if isinstance(n.kwargs, dict) else []):
                    for src in _iter_nodes(v):
                        lines.append(f'  "{src.name}" -> "{n.name}";')

            lines.append('}')
            dot = "\n".join(lines)

            dot_path = f"{file_prefix}.dot"
            with open(dot_path, "w") as f:
                f.write(dot)

            src = graphviz.Source(dot)
            png_path = src.render(filename=file_prefix, format="png", cleanup=True)
            return png_path, dot_path


    def compare_outputs(self, inputs: Tuple[torch.Tensor, ...]) -> Dict[str, float]:
        """Compare outputs between original and quantized models."""
        if self.quantized_model is None:
            raise RuntimeError("Must convert to quantized first")
        
        self.original_model.eval()
        self.quantized_model.eval()
        
        with torch.no_grad():
            if isinstance(inputs, torch.Tensor):
                inputs = (inputs,)
            orig_out = self.original_model(*inputs)
            quant_out = self.quantized_model(*inputs)
        
        if isinstance(orig_out, torch.Tensor):
            orig_out = orig_out.float()
            quant_out = quant_out.float()
            
            mse = torch.mean((orig_out - quant_out) ** 2).item()
            mae = torch.mean(torch.abs(orig_out - quant_out)).item()
            max_err = torch.max(torch.abs(orig_out - quant_out)).item()
            
            # Relative error
            rel_err = (torch.mean(torch.abs(orig_out - quant_out) / (torch.abs(orig_out) + 1e-8))).item()
            
            return {
                'mse': mse,
                'mae': mae,
                'max_error': max_err,
                'relative_error': rel_err
            }
        
        return {}