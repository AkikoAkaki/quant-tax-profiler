"""Shared helpers for the learning-first Phase 3 workflow."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_LAYER_PATH = "model.layers.0.self_attn.k_proj"


@dataclass
class Linear4bitArtifacts:
    """Small bundle of tensors and metadata from one bitsandbytes Linear4bit layer."""

    layer_path: str
    in_features: int
    out_features: int
    packed_weight: torch.Tensor
    bias: torch.Tensor | None
    absmax: torch.Tensor
    code: torch.Tensor
    blocksize: int
    quant_type: str
    quant_dtype: str
    weight_device: str
    packed_weight_shape: tuple[int, ...]
    logical_weight_shape: tuple[int, int]


def load_int4_model(model_id: str = DEFAULT_MODEL_ID):
    """Load the repo's current 4-bit baseline exactly as Phase 2 used it."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="cuda",
    )
    model.eval()
    return model, tokenizer


def resolve_module(root, dotted_path: str):
    """Resolve a dotted path like 'model.layers.0.self_attn.k_proj'."""
    current = root
    for part in dotted_path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def extract_linear4bit_artifacts(layer, layer_path: str) -> Linear4bitArtifacts:
    """Capture the tensors and metadata the fused prototype will eventually consume."""
    if layer.__class__.__name__ != "Linear4bit":
        raise TypeError(f"{layer_path} is {layer.__class__.__name__}, expected Linear4bit")

    quant_state = layer.weight.quant_state
    return Linear4bitArtifacts(
        layer_path=layer_path,
        in_features=int(layer.in_features),
        out_features=int(layer.out_features),
        packed_weight=layer.weight.data,
        bias=layer.bias,
        absmax=quant_state.absmax,
        code=quant_state.code,
        blocksize=int(quant_state.blocksize),
        quant_type=str(quant_state.quant_type),
        quant_dtype=str(quant_state.dtype),
        weight_device=str(layer.weight.device),
        packed_weight_shape=tuple(int(dim) for dim in layer.weight.data.shape),
        logical_weight_shape=(int(layer.out_features), int(layer.in_features)),
    )


def make_decode_input(
    in_features: int,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float16,
    seed: int = 0,
) -> torch.Tensor:
    """Create one decode-like activation tensor with shape [1, 1, in_features]."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn((1, 1, in_features), device=device, dtype=dtype, generator=generator)


def summarize_artifacts(artifacts: Linear4bitArtifacts) -> dict[str, object]:
    """Convert layer metadata into a stable printable dictionary."""
    return {
        "layer_path": artifacts.layer_path,
        "logical_weight_shape": artifacts.logical_weight_shape,
        "packed_weight_shape": artifacts.packed_weight_shape,
        "packed_weight_dtype": str(artifacts.packed_weight.dtype),
        "bias_shape": None if artifacts.bias is None else tuple(int(dim) for dim in artifacts.bias.shape),
        "bias_dtype": None if artifacts.bias is None else str(artifacts.bias.dtype),
        "quant_type": artifacts.quant_type,
        "blocksize": artifacts.blocksize,
        "quant_dtype": artifacts.quant_dtype,
        "absmax_shape": tuple(int(dim) for dim in artifacts.absmax.shape),
        "absmax_dtype": str(artifacts.absmax.dtype),
        "code_shape": tuple(int(dim) for dim in artifacts.code.shape),
        "code_dtype": str(artifacts.code.dtype),
        "weight_device": artifacts.weight_device,
    }


def run_reference_layer(layer, decode_input: torch.Tensor) -> torch.Tensor:
    """Run the original bitsandbytes layer under no_grad for comparison."""
    with torch.no_grad():
        return layer(decode_input)
