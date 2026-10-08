"""AlphaAI configuration loader.

Resolution order (first match wins):

1. explicit ``path`` argument
2. ``$ALPHAI_CONFIG``
3. ``<project_root>/configs/alphaai.toml`` (then ``.json``, then ``alphaai.toml``)
4. built-in defaults

Environment overrides are applied last, so a container can tune a deployment
without editing files. AlphaAI has **no** API-key environment variables: local
inference needs none.
"""

from __future__ import annotations

import json
import os
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

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

CONFIG_ENV_VAR = "ALPHAI_CONFIG"
DEFAULT_CONFIG_PATHS = (
    "configs/alphaai.toml",
    "configs/alphaai.json",
    "alphaai.toml",
    "alphaai.json",
)

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


class ConfigurationError(RuntimeError):
    """Raised when configuration cannot be read or parsed."""


# ---------------------------------------------------------------------------
# file parsing
# ---------------------------------------------------------------------------
def _read_structured(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if not path.exists():
        raise ConfigurationError(f"AlphaAI config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise ConfigurationError(f"Invalid JSON in {path}: {exc}") from exc
    elif suffix in {".toml", ".tml"}:
        try:
            import tomllib  # Python 3.11+
        except ModuleNotFoundError:  # pragma: no cover - Python 3.10 path
            try:
                import tomli as tomllib  # type: ignore[no-redef]
            except ModuleNotFoundError as exc:
                raise ConfigurationError(
                    "Reading TOML configs on Python < 3.11 requires the 'tomli' package "
                    "(`pip install tomli`), or use a .json config file."
                ) from exc
        try:
            data = tomllib.loads(text)
        except Exception as exc:  # tomllib raises TOMLDecodeError subclasses
            raise ConfigurationError(f"Invalid TOML in {path}: {exc}") from exc
    else:
        raise ConfigurationError(f"Unsupported config format '{suffix}' for {path} (use .toml or .json)")
    if not isinstance(data, dict):
        raise ConfigurationError(f"Config root must be a table/object in {path}")
    if "alphaai" in data and isinstance(data["alphaai"], dict):
        # Allow both flat configs and `[alphaai]`-namespaced configs.
        merged = dict(data["alphaai"])
        for key, value in data.items():
            if key != "alphaai":
                merged.setdefault(key, value)
        return merged
    return data


def resolve_config_path(
    path: str | os.PathLike[str] | None = None,
    project_root: str | os.PathLike[str] | None = None,
) -> Path | None:
    """Return the config file AlphaAI would use, or ``None`` for defaults."""

    if path:
        return Path(path).expanduser()
    env_path = os.environ.get(CONFIG_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser()
    root = Path(project_root or os.environ.get("ALPHAI_ROOT") or ".").expanduser().resolve()
    for candidate in DEFAULT_CONFIG_PATHS:
        full = root / candidate
        if full.exists():
            return full
    return None


# ---------------------------------------------------------------------------
# dataclass merging
# ---------------------------------------------------------------------------
def _coerce_dataclass(instance: Any, data: Mapping[str, Any], *, where: str) -> Any:
    if not is_dataclass(instance) or not isinstance(data, Mapping):
        return data
    known = {f.name: f for f in fields(instance)}
    if where in {"engines.options", "routing.task_preferences"}:  # free-form maps
        for key, value in data.items():
            setattr(instance, key, value)
        return instance
    for key, value in data.items():
        if key not in known:
            raise ConfigurationError(f"Unknown configuration key '{where}.{key}'")
        current = getattr(instance, key)
        if is_dataclass(current):
            setattr(instance, key, _coerce_dataclass(current, value, where=f"{where}.{key}"))
        elif key == "permissions" and isinstance(value, Mapping):
            parsed: dict[str, ToolPermissions] = {}
            for tool_id, perm_data in value.items():
                if not isinstance(perm_data, Mapping):
                    raise ConfigurationError(f"{where}.permissions.{tool_id} must be a table/object")
                parsed[str(tool_id)] = _coerce_dataclass(
                    ToolPermissions(), perm_data, where=f"{where}.permissions.{tool_id}"
                )
            setattr(instance, key, parsed)
        else:
            setattr(instance, key, value)
    return instance


# ---------------------------------------------------------------------------
# environment overrides
# ---------------------------------------------------------------------------
def _parse_bool(raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in _TRUTHY:
        return True
    if lowered in _FALSY:
        return False
    raise ConfigurationError(f"Expected a boolean value, got '{raw}'")


def _apply_env(config: AlphaAIConfig, env: Mapping[str, str]) -> list[str]:
    """Apply ``ALPHAI_*`` overrides. Returns the names of applied variables."""

    def get(name: str) -> str | None:
        value = env.get(name)
        return value if value is not None else None

    applied: list[str] = []

    def set_field(target: Any, attr: str, raw: str | None, cast=str) -> None:
        if raw is None:
            return
        try:
            value = _parse_bool(raw) if cast is bool else cast(raw)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"Invalid value for {attr}: {raw!r} ({exc})") from exc
        setattr(target, attr, value)
        applied.append(attr)

    def set_list(target: Any, attr: str, raw: str | None) -> None:
        if raw is None:
            return
        items = [part.strip() for part in raw.split(",") if part.strip()]
        setattr(target, attr, items)
        applied.append(attr)

    # paths
    set_field(config.paths, "project_root", get("ALPHAI_ROOT"))
    set_field(config.paths, "models_dir", get("ALPHAI_MODELS_DIR"))
    set_field(config.paths, "datasets_dir", get("ALPHAI_DATASETS_DIR"))
    set_field(config.paths, "tokenizer_dir", get("ALPHAI_TOKENIZER_DIR"))
    set_field(config.paths, "checkpoints_dir", get("ALPHAI_CHECKPOINTS_DIR"))
    set_field(config.paths, "training_dir", get("ALPHAI_TRAINING_DIR"))
    set_field(config.paths, "evaluation_dir", get("ALPHAI_EVALUATION_DIR"))
    set_field(config.paths, "state_dir", get("ALPHAI_STATE_DIR"))
    set_field(config.paths, "workspace_dir", get("ALPHAI_WORKSPACE_DIR"))
    set_field(config.paths, "log_dir", get("ALPHAI_LOG_DIR"))

    # runtime
    set_field(config.runtime, "device", get("ALPHAI_DEVICE"))
    set_field(config.runtime, "dtype", get("ALPHAI_DTYPE"))
    set_field(config.runtime, "cpu_threads", get("ALPHAI_CPU_THREADS"), int)
    set_field(config.runtime, "torch_num_threads", get("ALPHAI_TORCH_THREADS"), int)
    set_field(config.runtime, "gpu_layers", get("ALPHAI_GPU_LAYERS"), int)
    set_field(config.runtime, "worker_parallelism", get("ALPHAI_WORKER_PARALLELISM"), int)
    set_field(config.runtime, "allow_model_download", get("ALPHAI_ALLOW_MODEL_DOWNLOAD"), bool)

    # sampling / conversation
    set_field(config.sampling, "temperature", get("ALPHAI_TEMPERATURE"), float)
    set_field(config.sampling, "top_p", get("ALPHAI_TOP_P"), float)
    set_field(config.sampling, "max_tokens", get("ALPHAI_MAX_TOKENS"), int)
    set_field(config.conversation, "system_prompt", get("ALPHAI_SYSTEM_PROMPT"))
    set_field(config.conversation, "tool_loop_limit", get("ALPHAI_TOOL_LOOP_LIMIT"), int)

    # routing
    set_field(config.routing, "long_context_threshold_tokens", get("ALPHAI_LONG_CONTEXT_TOKENS"), int)

    # engines / skills
    set_list(config.engines, "enabled", get("ALPHAI_ENGINES"))
    set_list(config.engines, "disabled", get("ALPHAI_DISABLED_ENGINES"))
    set_list(config.skills, "enabled", get("ALPHAI_SKILLS"))
    set_list(config.tools, "enabled", get("ALPHAI_TOOLS"))

    # tools
    set_field(config.tools, "sandbox_root", get("ALPHAI_SANDBOX_ROOT"))
    set_field(config.tools, "allow_network", get("ALPHAI_TOOL_NETWORK"), bool)
    set_field(config.tools, "allow_code_execution", get("ALPHAI_TOOL_CODE_EXEC"), bool)
    set_field(config.tools, "allow_writes", get("ALPHAI_TOOL_WRITES"), bool)
    set_field(config.tools, "default_timeout_s", get("ALPHAI_TOOL_TIMEOUT"), float)

    # memory / api
    set_field(config.memory, "enabled", get("ALPHAI_MEMORY"), bool)
    set_field(config.memory, "path", get("ALPHAI_MEMORY_PATH"))
    set_field(config.api, "host", get("ALPHAI_HOST"))
    set_field(config.api, "port", get("ALPHAI_PORT"), int)
    set_field(config.api, "redact_paths", get("ALPHAI_REDACT_PATHS"), bool)
    set_field(config.api, "max_request_bytes", get("ALPHAI_MAX_REQUEST_BYTES"), int)
    # Production CORS is configuration, not a wildcard: set the deployed frontend
    # origin(s) here (comma-separated) instead of editing source.
    set_list(config.api, "cors_origins", get("ALPHAI_CORS_ORIGINS"))
    # Remote inference gateway. Setting an inference URL keeps this deployment
    # stateless (no local model, no writable disk) and forwards inference to a
    # real AlphaAI server. Both spellings are accepted; the ALPHAI_ one wins.
    set_field(
        config.api,
        "inference_url",
        get("ALPHAI_INFERENCE_URL") or get("ALPHA_INFERENCE_URL"),
    )
    set_field(
        config.api,
        "inference_token",
        get("ALPHAI_INFERENCE_TOKEN") or get("ALPHA_INFERENCE_TOKEN"),
    )
    set_field(config.api, "inference_timeout_s", get("ALPHAI_INFERENCE_TIMEOUT_S"), float)
    return applied


# ---------------------------------------------------------------------------
# path resolution
# ---------------------------------------------------------------------------
def _resolve_path(base: Path, value: str) -> str:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return str(candidate.resolve())


def resolve_paths(config: AlphaAIConfig, *, create: bool = False) -> AlphaAIConfig:
    """Make every path absolute (relative to ``paths.project_root``)."""

    root = Path(config.paths.project_root).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    root = root.resolve()
    config.paths.project_root = str(root)

    for attr in (
        "models_dir",
        "configs_dir",
        "datasets_dir",
        "tokenizer_dir",
        "checkpoints_dir",
        "training_dir",
        "evaluation_dir",
        "state_dir",
        "workspace_dir",
        "log_dir",
    ):
        setattr(config.paths, attr, _resolve_path(root, getattr(config.paths, attr)))

    config.memory.path = _resolve_path(root, config.memory.path)
    config.tools.sandbox_root = _resolve_path(root, config.tools.sandbox_root)
    for perm in config.tools.permissions.values():
        if perm.root:
            perm.root = _resolve_path(root, perm.root)

    if create:
        for attr in ("state_dir", "log_dir", "workspace_dir", "models_dir"):
            Path(getattr(config.paths, attr)).mkdir(parents=True, exist_ok=True)
        if config.memory.enabled:
            Path(config.memory.path).parent.mkdir(parents=True, exist_ok=True)
    return config


def load_config(
    path: str | os.PathLike[str] | None = None,
    *,
    project_root: str | os.PathLike[str] | None = None,
    overrides: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    create_dirs: bool = False,
    require_file: bool = False,
) -> AlphaAIConfig:
    """Load the AlphaAI configuration."""

    if project_root is not None:
        os.environ.setdefault("ALPHAI_ROOT", str(project_root))
    resolved = resolve_config_path(path, project_root)
    config = AlphaAIConfig()
    if resolved is not None:
        if require_file and not Path(resolved).exists():
            raise ConfigurationError(f"AlphaAI config file not found: {resolved}")
        config = _coerce_dataclass(config, _read_structured(Path(resolved)), where="")
        config.source = str(resolved)
    if overrides:
        config = _coerce_dataclass(config, overrides, where="(overrides)")
    applied = _apply_env(config, os.environ if env is None else env)
    if applied:
        config.source = f"{config.source} (+env: {', '.join(sorted(set(applied)))})"
    if project_root is not None:
        # An explicit project_root (CLI flag, API argument) is authoritative and
        # must not be overridden by a stale ALPHAI_ROOT in the environment.
        config.paths.project_root = str(project_root)
    resolve_paths(config, create=create_dirs)
    problems = config.validate()
    if problems:
        raise ConfigurationError("Invalid AlphaAI configuration: " + "; ".join(problems))
    return config


# ---------------------------------------------------------------------------
# client-safe view
# ---------------------------------------------------------------------------
_SENSITIVE_KEY_PARTS = ("key", "token", "secret", "password", "credential")


def public_config_view(config: AlphaAIConfig, *, redact_paths: bool | None = None) -> dict[str, Any]:
    """Return a config dict safe to hand to a client application.

    Local filesystem layout is redacted by default and anything that looks like a
    secret is dropped. Model *paths* are never required by a client: a client
    addresses models by id.
    """

    redact = config.api.redact_paths if redact_paths is None else redact_paths
    data = config.to_dict()
    root = config.paths.project_root

    def scrub(value: Any, key: str = "") -> Any:
        if any(part in key.lower() for part in _SENSITIVE_KEY_PARTS):
            return "<redacted>"
        if isinstance(value, dict):
            return {k: scrub(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [scrub(v, key) for v in value]
        if isinstance(value, str) and redact and (value.startswith("/") or (len(value) > 2 and value[1] == ":")):
            if root and value.startswith(root):
                return "<local>/" + os.path.relpath(value, root).replace(os.sep, "/")
            return "<local>/" + os.path.basename(value)
        return value

    return scrub(data)
