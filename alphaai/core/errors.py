"""AlphaAI error taxonomy.

Every error carries a stable machine-readable ``code`` plus an optional
``remediation`` hint, so the CLI, the API and the dashboard can present a clear
failure instead of inventing a fallback answer.
"""

from __future__ import annotations


class AlphaAIError(RuntimeError):
    """Base class for all AlphaAI errors."""

    code = "alphaai_error"

    def __init__(self, message: str, *, remediation: str | None = None, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.remediation = remediation
        self.details = details or {}

    def to_dict(self) -> dict:
        payload = {"code": self.code, "message": self.message}
        if self.remediation:
            payload["remediation"] = self.remediation
        if self.details:
            payload["details"] = self.details
        return payload


# ---------------------------------------------------------------------------
# engines & routing
# ---------------------------------------------------------------------------
class EngineUnavailableError(AlphaAIError):
    """An engine cannot serve a request (missing runtime, weights or hardware)."""

    code = "engine_unavailable"


class EngineLoadError(AlphaAIError):
    """An engine failed while loading its model weights."""

    code = "engine_load_failed"


class GenerationError(AlphaAIError):
    """Inference itself failed."""

    code = "generation_failed"


class NoSuitableModelError(AlphaAIError):
    """The router found no usable engine for the requested task."""

    code = "no_suitable_model"


class UnknownModelError(AlphaAIError):
    """A model id was requested that is not registered."""

    code = "unknown_model"


class ModelIncompatibleError(AlphaAIError):
    """Hardware/format checks refuse to run a model."""

    code = "model_incompatible"


class InferenceUnreachableError(AlphaAIError):
    """A gateway deployment cannot reach the AlphaAI inference server.

    Raised by the remote-inference gateway (``api.inference_url``). The gateway
    never substitutes a local model or a canned answer for an unreachable
    inference host.
    """

    code = "inference_unreachable"


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------
class ToolError(AlphaAIError):
    code = "tool_error"


class ToolNotFoundError(ToolError):
    code = "tool_not_found"


class ToolPermissionError(ToolError):
    code = "tool_permission_denied"


class ToolTimeoutError(ToolError):
    code = "tool_timeout"


class ToolExecutionError(ToolError):
    code = "tool_execution_failed"


class ToolValidationError(ToolError):
    code = "tool_invalid_input"


# ---------------------------------------------------------------------------
# skills
# ---------------------------------------------------------------------------
class SkillError(AlphaAIError):
    code = "skill_error"


class SkillNotFoundError(SkillError):
    code = "skill_not_found"


class SkillPermissionError(SkillError):
    code = "skill_permission_denied"


class SkillValidationError(SkillError):
    code = "skill_invalid_input"


class SkillExecutionError(SkillError):
    code = "skill_execution_failed"


class SkillTimeoutError(SkillError):
    code = "skill_timeout"


# ---------------------------------------------------------------------------
# conversation / context / memory
# ---------------------------------------------------------------------------
class ContextOverflowError(AlphaAIError):
    code = "context_overflow"


class ConversationError(AlphaAIError):
    code = "conversation_error"


class MemoryError_(AlphaAIError):  # noqa: N801 - name mirrors the subsystem
    code = "memory_error"


# ---------------------------------------------------------------------------
# training foundation
# ---------------------------------------------------------------------------
class TrainingError(AlphaAIError):
    code = "training_error"


class DatasetError(TrainingError):
    code = "dataset_error"


class DatasetValidationError(DatasetError):
    code = "dataset_invalid"


class CheckpointError(TrainingError):
    code = "checkpoint_error"


class EvaluationError(TrainingError):
    code = "evaluation_error"


class MetadataError(TrainingError):
    code = "metadata_error"
