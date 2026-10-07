"""AlphaAI configuration package.

``AlphaAIConfig`` (schema) + ``load_config`` (loader). AlphaAI configuration is
local-only: no AI provider keys are ever required or read.
"""

from __future__ import annotations

from .loader import (
    CONFIG_ENV_VAR,
    DEFAULT_CONFIG_PATHS,
    load_config,
    public_config_view,
    resolve_config_path,
)
from .schema import (
    AlphaAIConfig,
    ApiConfig,
    ConversationConfig,
    EnginesConfig,
    MemoryConfig,
    PathsConfig,
    RoutingConfig,
    RuntimeConfig,
    SamplingConfig,
    SkillsConfig,
    ToolPermissions,
    ToolsConfig,
    TrainingConfigPaths,
)

__all__ = [
    "AlphaAIConfig",
    "ApiConfig",
    "CONFIG_ENV_VAR",
    "ConversationConfig",
    "DEFAULT_CONFIG_PATHS",
    "EnginesConfig",
    "MemoryConfig",
    "PathsConfig",
    "RoutingConfig",
    "RuntimeConfig",
    "SamplingConfig",
    "SkillsConfig",
    "ToolPermissions",
    "ToolsConfig",
    "TrainingConfigPaths",
    "load_config",
    "public_config_view",
    "resolve_config_path",
]
