"""Hard-coded model dimensions for the spikes. Replaced by config.json import (frontends/hf_llm/arch.py) in the compiler proper."""
from .frontends.hf_llm.builders.decoder import DecoderDims

PRESETS = {
    # Qwen3-4B: the vocabulary 151936 is already a multiple of 128
    "qwen3-4b": dict(dims=DecoderDims(hidden=2560, num_heads=32, num_kv_heads=8, head_dim=128, ffn=9728, vocab=151936, eps=1e-6, qk_norm=True), layers=36),
    # Aion-1.0-Instruct-2.3B: vocab 200029 padded to 200064 (a multiple of 128)
    "aion-2.3b": dict(dims=DecoderDims(hidden=2048, num_heads=24, num_kv_heads=8, head_dim=128, ffn=8192, vocab=200064, eps=1e-6, qk_norm=True), layers=28),
}
