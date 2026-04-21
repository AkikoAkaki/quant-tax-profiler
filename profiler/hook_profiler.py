"""Core layer-wise profiler using PyTorch forward hooks + CUDA Events."""

import torch
import torch.nn as nn
import pandas as pd

from .metrics import CUDATimer, get_memory_mb, get_peak_memory_mb, reset_peak_memory


class HookProfiler:
    """Attaches pre/post forward hooks to target layers and records per-layer metrics.

    Args:
        model: A PyTorch nn.Module (HuggingFace model).
        target_types: Tuple of nn.Module subclasses to profile.
    """

    def __init__(self, model: nn.Module,
                 target_types=(nn.Linear, nn.LayerNorm)):
        self.model = model
        self.target_types = target_types
        self.records = []
        self._handles = []
        self._timers = {}  # layer_name -> CUDATimer
        self._pre_mem = {}  # layer_name -> mem_before_mb

        # Externally set before each forward pass
        self.current_phase = "unknown"
        self.current_decode_step = None
        self.run_id = "single-run"
        self.recording = True

    def attach(self):
        """Register hooks on all target layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, self.target_types):
                h_pre = module.register_forward_pre_hook(
                    self._make_pre_hook(name)
                )
                h_post = module.register_forward_hook(
                    self._make_post_hook(name, module)
                )
                self._handles.append(h_pre)
                self._handles.append(h_post)

    def detach(self):
        """Remove all hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._timers.clear()
        self._pre_mem.clear()

    def clear_records(self):
        """Clear collected data."""
        self.records.clear()

    def _make_pre_hook(self, layer_name: str):
        def hook(module, inputs):
            if not self.recording:
                return
            reset_peak_memory()
            self._pre_mem[layer_name] = get_memory_mb()
            timer = CUDATimer()
            timer.start()
            self._timers[layer_name] = timer
        return hook

    def _make_post_hook(self, layer_name: str, module: nn.Module):
        def hook(mod, inputs, output):
            if not self.recording:
                return
            timer = self._timers.get(layer_name)
            if timer is None:
                return

            elapsed_ms = timer.stop()
            mem_after = get_memory_mb()
            mem_peak = get_peak_memory_mb()
            mem_before = self._pre_mem.get(layer_name, 0.0)

            # Extract shapes
            input_shape = _extract_shape(inputs)
            output_shape = _extract_shape(output)

            self.records.append({
                "run_id": self.run_id,
                "layer_name": layer_name,
                "layer_type": module.__class__.__name__,
                "phase": self.current_phase,
                "decode_step": self.current_decode_step,
                "time_ms": elapsed_ms,
                "mem_before_mb": mem_before,
                "mem_after_mb": mem_after,
                "mem_peak_mb": mem_peak,
                "input_shape": str(input_shape),
                "output_shape": str(output_shape),
            })

            # Cleanup
            self._timers.pop(layer_name, None)
            self._pre_mem.pop(layer_name, None)
        return hook

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def export_csv(self, path: str):
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        print(f"Exported {len(df)} records to {path}")


def _extract_shape(tensor_or_tuple):
    """Get shape from a tensor, or from the first tensor in a tuple."""
    if isinstance(tensor_or_tuple, torch.Tensor):
        return tuple(tensor_or_tuple.shape)
    if isinstance(tensor_or_tuple, (tuple, list)) and len(tensor_or_tuple) > 0:
        first = tensor_or_tuple[0]
        if isinstance(first, torch.Tensor):
            return tuple(first.shape)
    return None
