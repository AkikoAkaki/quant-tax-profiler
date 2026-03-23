"""Prefill / Decode phase detection based on input tensor shape."""


class PhaseDetector:
    """Tracks the current inference phase.

    - Prefill: input_ids has seq_len > 1 (processing the full prompt)
    - Decode:  input_ids has seq_len == 1 (generating one token at a time)
    """

    def __init__(self):
        self.current_phase = "unknown"

    def update(self, input_ids):
        """Call before each forward pass with the input_ids tensor."""
        seq_len = input_ids.shape[1] if input_ids.dim() >= 2 else input_ids.shape[0]
        self.current_phase = "prefill" if seq_len > 1 else "decode"

    @property
    def phase(self) -> str:
        return self.current_phase
