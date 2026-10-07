"""AlphaAI Core.

The system layer: model engine interface, registry, router, context manager,
conversation engine, memory, tool engine, skills and the agent orchestrator.

AlphaAI Core has no dependency on any model vendor and no dependency on external
AI APIs. Engines are adapters that perform local inference.
"""

from __future__ import annotations

from .engine import ModelEngine, ModelSpec, load_model_specs, spec_from_dict
from .errors import (
    AlphaAIError,
    EngineLoadError,
    EngineUnavailableError,
    GenerationError,
    ModelIncompatibleError,
    NoSuitableModelError,
    UnknownModelError,
)
from .registry import ModelRegistry
from .router import ModelRouter, RouteDecision
from .types import (
    Capability,
    ChatMessage,
    EngineStatus,
    GenerationRequest,
    GenerationResult,
    SamplingParams,
    StreamChunk,
    TokenUsage,
    ToolCall,
    ToolResult,
)

__all__ = [
    "AlphaAIError",
    "Capability",
    "ChatMessage",
    "EngineLoadError",
    "EngineStatus",
    "EngineUnavailableError",
    "GenerationError",
    "GenerationRequest",
    "GenerationResult",
    "ModelEngine",
    "ModelIncompatibleError",
    "ModelRegistry",
    "ModelRouter",
    "ModelSpec",
    "NoSuitableModelError",
    "RouteDecision",
    "SamplingParams",
    "StreamChunk",
    "TokenUsage",
    "ToolCall",
    "ToolResult",
    "UnknownModelError",
    "load_model_specs",
    "spec_from_dict",
]
