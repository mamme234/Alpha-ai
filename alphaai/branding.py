"""AlphaAI identity and attribution strings.

Everything that is *AlphaAI* (application layer branding) lives here, and so do
the *attribution* strings that must always accompany the underlying models.

Rules encoded in this module
----------------------------
* AlphaAI owns the **system layer**: router, conversation engine, context
  manager, tool engine, skills, memory, orchestrator, API and training
  foundation.
* DeepSeek owns the **DeepSeek-V3 model and its reference implementation**.
  AlphaAI must never claim that DeepSeek-V3 was created by AlphaAI.
* Any user-visible surface that talks about DeepSeek-V3 must present it as the
  underlying model of the AlphaAI DeepSeek engine.
"""

from __future__ import annotations

from .version import __version__

# --------------------------------------------------------------------------
# Core identity
# --------------------------------------------------------------------------
NAME = "AlphaAI"
DISPLAY_NAME = "ALPHA AI"
TAGLINE = "Intelligence, built from the ground up."
DESCRIPTION = (
    "AlphaAI is an open AI system layer: a model router over local inference "
    "engines, a real tool and skill runtime, persistent memory, an agent "
    "orchestrator and a training foundation for future AlphaAI-owned weights."
)
CODE_NAME = "AlphaAI Core"
VERSION = __version__

#: Machine-readable identifiers used for packages, config keys, server
#: metadata and container names.
SLUG = "alphaai"
SERVER_NAME = "alphaai-server"
PACKAGE_NAME = "alphaai"
ENV_PREFIX = "ALPHAI_"

BANNER = r"""
    ___    __    ____  __  __   _    ___
   /   |  / /   / __ \/ / / /  / \  |_ _|
  / /| | / /   / /_/ / /_/ /  / _ \  | |
 / ___ |/ /___/ ____/ __  /  / ___ \ | |
/_/  |_/_____/_/   /_/ /_/  /_/  \_\___|
""".strip("\n")

# --------------------------------------------------------------------------
# Underlying-model attribution (must remain intact — see NOTICE / ATTRIBUTION.md)
# --------------------------------------------------------------------------
DEEPSEEK_V3_MODEL = "DeepSeek-V3"
DEEPSEEK_OWNER = "DeepSeek"
DEEPSEEK_V3_ENGINE_NAME = "AlphaAI DeepSeek Engine"

#: AlphaAI always owns the *engine* layer, for every model it runs.
ENGINE_OWNER_DEFAULT = NAME
#: The owner recorded on AlphaAI-trained weights (none exist yet).
MODEL_OWNER_FUTURE = NAME

#: Short form used in UI headers, CLI output and README highlights.
ATTRIBUTION_SHORT = f"AlphaAI powered by {DEEPSEEK_V3_MODEL}"

#: Explicit form used wherever a model is described in detail.
ATTRIBUTION_ENGINE = (
    f"{DEEPSEEK_V3_ENGINE_NAME} — underlying model: {DEEPSEEK_V3_MODEL} "
    f"(created by {DEEPSEEK_OWNER}; not an AlphaAI-trained model)"
)

#: Long form used in the API, the dashboard footer and generated reports.
ATTRIBUTION_LONG = (
    f"AlphaAI is the system layer (router, tools, skills, memory, orchestrator, "
    f"API, training foundation) and is an independent project. The current "
    f"{DEEPSEEK_V3_ENGINE_NAME} runs {DEEPSEEK_V3_MODEL}, which is created and "
    f"released by {DEEPSEEK_OWNER}. {DEEPSEEK_V3_MODEL} weights and the vendored "
    f"reference implementation remain under {DEEPSEEK_OWNER} copyright: MIT for "
    f"code (LICENSE-CODE) and the {DEEPSEEK_OWNER} Model License for weights "
    f"(LICENSE-MODEL). No AlphaAI weights exist yet."
)

#: Copyright / license facts that must not be edited out of the repository.
PRESERVED_NOTICES = (
    "LICENSE-CODE",
    "LICENSE-MODEL",
    "NOTICE",
    "ATTRIBUTION.md",
)


def powered_by(model: str) -> str:
    """Transparent line for whichever model is actually serving requests.

    AlphaAI always names the real underlying model. There is no AlphaAI-trained
    model yet, so AlphaAI never says "AlphaAI model".
    """

    return f"{NAME} powered by {model}"


def engine_attribution(engine_name: str, model: str, owner: str) -> str:
    """Build the attribution line for an arbitrary engine/model pair."""

    if owner.lower() in {DEEPSEEK_OWNER.lower(), "alibaba", "meta", "mistral ai", "google", "moonshot ai"}:
        return f"{engine_name} — underlying model: {model} (created by {owner})"
    return f"{engine_name} — model: {model} (owned by {owner})"


def banner(subtitle: str | None = None) -> str:
    """Return the startup banner with the current version and attribution."""

    lines = [BANNER, ""]
    lines.append(f"  {DISPLAY_NAME} v{VERSION} — {TAGLINE}")
    if subtitle:
        lines.append(f"  {subtitle}")
    lines.append(f"  {ATTRIBUTION_SHORT}")
    return "\n".join(lines)


def attribution_lines() -> list[str]:
    """Return the attribution block used by CLI/server startup messages."""

    return [
        ATTRIBUTION_SHORT,
        ATTRIBUTION_ENGINE,
        "DeepSeek-V3 code: MIT (LICENSE-CODE) · weights: DeepSeek Model License (LICENSE-MODEL)",
    ]


__all__ = [
    "ATTRIBUTION_ENGINE",
    "ATTRIBUTION_LONG",
    "ATTRIBUTION_SHORT",
    "BANNER",
    "DESCRIPTION",
    "DEEPSEEK_V3_ENGINE_NAME",
    "DEEPSEEK_V3_MODEL",
    "DISPLAY_NAME",
    "ENGINE_OWNER_DEFAULT",
    "MODEL_OWNER_FUTURE",
    "NAME",
    "PRESERVED_NOTICES",
    "TAGLINE",
    "VERSION",
    "attribution_lines",
    "banner",
    "engine_attribution",
    "powered_by",
]
