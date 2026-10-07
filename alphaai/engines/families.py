"""Per-family AlphaAI engine adapters.

Each class here is a real engine: the same tested code path as
:class:`~alphaai.engines.hf_engine.TransformersEngine` (for ``safetensors``
checkpoints) or :class:`~alphaai.engines.llama_cpp.LlamaCppEngine` (for GGUF
builds), with family-specific defaults and, most importantly, honest attribution:
AlphaAI owns the engine, the model owner is whoever trained the weights.

``AlphaAIEngine`` is the engine for future AlphaAI-trained checkpoints. It works
exactly like the others; today no AlphaAI weights exist, so its model metadata
declares ``status = "not_trained"`` and the engine reports that truthfully.
"""

from __future__ import annotations

from .hf_engine import TransformersEngine
from .llama_cpp import LlamaCppEngine

# ---------------------------------------------------------------------------
# transformers-based family engines
# ---------------------------------------------------------------------------


class QwenEngine(TransformersEngine):
    engine_key = "qwen"
    engine_name = "AlphaAI Qwen Engine"
    family_kwargs = {"use_safetensors": True}


class KimiEngine(TransformersEngine):
    engine_key = "kimi"
    engine_name = "AlphaAI Kimi Engine"
    family_kwargs = {"use_safetensors": True}


class LlamaEngine(TransformersEngine):
    engine_key = "llama"
    engine_name = "AlphaAI Llama Engine"
    family_kwargs = {"use_safetensors": True}


class MistralEngine(TransformersEngine):
    engine_key = "mistral"
    engine_name = "AlphaAI Mistral Engine"
    family_kwargs = {"use_safetensors": True}


class GemmaEngine(TransformersEngine):
    engine_key = "gemma"
    engine_name = "AlphaAI Gemma Engine"
    family_kwargs = {"use_safetensors": True}


class AlphaAIEngine(TransformersEngine):
    """Engine for AlphaAI-owned checkpoints (none trained yet — see docs/TRAINING.md)."""

    engine_key = "alphaai"
    engine_name = "AlphaAI Core Engine"


# ---------------------------------------------------------------------------
# GGUF family engines (same code path as LlamaCppEngine)
# ---------------------------------------------------------------------------


class QwenGgufEngine(LlamaCppEngine):
    engine_key = "qwen_gguf"
    engine_name = "AlphaAI Qwen GGUF Engine"


class DeepSeekGgufEngine(LlamaCppEngine):
    engine_key = "deepseek_gguf"
    engine_name = "AlphaAI DeepSeek GGUF Engine"


__all__ = [
    "AlphaAIEngine",
    "DeepSeekGgufEngine",
    "GemmaEngine",
    "KimiEngine",
    "LlamaEngine",
    "MistralEngine",
    "QwenEngine",
    "QwenGgufEngine",
]
