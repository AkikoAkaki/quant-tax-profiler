"""CUDA Event timing and memory tracking utilities."""

import torch


class CUDATimer:
    """Measures GPU kernel time using CUDA Events (not wall-clock time).

    Usage:
        timer = CUDATimer()
        timer.start()
        # ... GPU work ...
        elapsed_ms = timer.stop()
    """

    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        self.start_event.record()

    def stop(self) -> float:
        """Records end event, synchronizes, and returns elapsed time in ms."""
        self.end_event.record()
        torch.cuda.synchronize()
        return self.start_event.elapsed_time(self.end_event)


def get_memory_mb() -> float:
    """Current GPU memory allocated in MB."""
    return torch.cuda.memory_allocated() / 1e6


def get_peak_memory_mb() -> float:
    """Peak GPU memory allocated in MB."""
    return torch.cuda.max_memory_allocated() / 1e6


def reset_peak_memory():
    """Reset peak memory stats so next peak is layer-local."""
    torch.cuda.reset_peak_memory_stats()
