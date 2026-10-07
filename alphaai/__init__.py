"""AlphaAI — an open AI system layer.

AlphaAI Core provides the model router, conversation engine, context manager,
tool engine, skill manager, memory system, agent orchestrator, a unified API and
the training foundation for future AlphaAI-owned weights.

The DeepSeek engine inside AlphaAI runs the DeepSeek-V3 model. DeepSeek-V3 is
created by DeepSeek and is *not* an AlphaAI-trained model; see ATTRIBUTION.md.

Importing :mod:`alphaai` is deliberately cheap: no torch, no HTTP server and no
model runtime is imported here. Engines are discovered lazily and report their
own availability.
"""

from __future__ import annotations

from .branding import (
    ATTRIBUTION_ENGINE,
    ATTRIBUTION_LONG,
    ATTRIBUTION_SHORT,
    DISPLAY_NAME,
    NAME,
    SERVER_NAME,
    TAGLINE,
)
from .version import CORE_INTERFACE_VERSION, __version__

__all__ = [
    "ATTRIBUTION_ENGINE",
    "ATTRIBUTION_LONG",
    "ATTRIBUTION_SHORT",
    "CORE_INTERFACE_VERSION",
    "DISPLAY_NAME",
    "NAME",
    "SERVER_NAME",
    "TAGLINE",
    "__version__",
    "get_config",
    "build_runtime",
]


def get_config(*args, **kwargs):
    """Load an :class:`~alphaai.config.schema.AlphaAIConfig` (see config loader)."""

    from .config.loader import load_config

    return load_config(*args, **kwargs)


def build_runtime(*args, **kwargs):
    """Build the full AlphaAI runtime (registry, router, tools, skills, memory).

    Imported lazily so that ``import alphaai`` stays dependency-light.
    """

    from .core.runtime import AlphaRuntime

    return AlphaRuntime.create(*args, **kwargs)
