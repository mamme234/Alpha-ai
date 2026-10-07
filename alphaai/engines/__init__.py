"""AlphaAI engines.

An *engine* is a real local runtime that can load weights and generate text. The
engines AlphaAI ships:

======================================  ==========================================
engine key                              runtime
======================================  ==========================================
``deepseek``                            DeepSeek-V3 reference implementation (torch)
``transformers``                        Hugging Face ``transformers``
``qwen`` ``kimi`` ``llama`` ``mistral`` ``gemma`` ``alphaai``
                                        same transformers path, family defaults
``llama_cpp``                           GGUF via llama-cpp-python
``qwen_gguf`` ``deepseek_gguf``         GGUF, family defaults
======================================  ==========================================

Importing this package is cheap: heavy runtimes (torch, transformers,
llama_cpp) are only imported when an engine actually loads.
"""

from __future__ import annotations

from typing import Any

from .hardware import (
    AvailabilityAssessment,
    HardwareReport,
    ModelSizeRecommendation,
    assess_model,
    describe_hardware,
    describe_hardware_block,
    detect_hardware,
    estimate_resources,
    hardware_block,
    recommend_model_size,
)

#: engine key -> "module:ClassName". Imported lazily by the model registry.
ENGINE_CLASSES: dict[str, str] = {
    "deepseek": "alphaai.engines.deepseek:DeepSeekEngine",
    "transformers": "alphaai.engines.hf_engine:TransformersEngine",
    "llama_cpp": "alphaai.engines.llama_cpp:LlamaCppEngine",
    "qwen": "alphaai.engines.families:QwenEngine",
    "kimi": "alphaai.engines.families:KimiEngine",
    "llama": "alphaai.engines.families:LlamaEngine",
    "mistral": "alphaai.engines.families:MistralEngine",
    "gemma": "alphaai.engines.families:GemmaEngine",
    "alphaai": "alphaai.engines.families:AlphaAIEngine",
    "qwen_gguf": "alphaai.engines.families:QwenGgufEngine",
    "deepseek_gguf": "alphaai.engines.families:DeepSeekGgufEngine",
}


def engine_class(key: str):
    """Resolve an engine key to its class without importing it at module load."""

    from importlib import import_module

    target = ENGINE_CLASSES.get(key)
    if target is None:
        raise KeyError(f"Unknown engine key '{key}'. Known: {', '.join(sorted(ENGINE_CLASSES))}.")
    module_name, _, class_name = target.partition(":")
    return getattr(import_module(module_name), class_name)


def engine_keys() -> list[str]:
    return sorted(ENGINE_CLASSES)


def runtime_summary() -> dict[str, Any]:
    """Which optional runtimes exist here (used by ``alphaai doctor`` and the API)."""

    from .helpers import module_present

    return {
        "torch": module_present("torch"),
        "transformers": module_present("transformers"),
        "llama_cpp": module_present("llama_cpp"),
        "triton": module_present("triton"),
        "safetensors": module_present("safetensors"),
    }


__all__ = [
    "AvailabilityAssessment",
    "ENGINE_CLASSES",
    "HardwareReport",
    "ModelSizeRecommendation",
    "assess_model",
    "describe_hardware",
    "describe_hardware_block",
    "detect_hardware",
    "engine_class",
    "engine_keys",
    "estimate_resources",
    "hardware_block",
    "recommend_model_size",
    "runtime_summary",
]
