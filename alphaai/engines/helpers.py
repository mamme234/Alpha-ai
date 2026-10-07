"""Shared helpers for AlphaAI engine adapters.

Every heavy runtime import in AlphaAI happens inside a function in this module,
so that ``import alphaai`` and engine *discovery* never pull in torch,
transformers or llama.cpp.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from ..config.schema import AlphaAIConfig
from ..core.errors import EngineUnavailableError

#: torch dtypes AlphaAI maps its config values onto.
_TORCH_DTYPES = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "fp32": "float32",
}


def import_module(name: str) -> ModuleType:
    """Import a module or raise :class:`EngineUnavailableError` with a hint."""

    try:
        return importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - any import failure is "unavailable"
        raise EngineUnavailableError(
            f"The '{name}' runtime is not importable in this environment: {type(exc).__name__}: {exc}",
            remediation=_install_hint(name),
        ) from exc


def module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken installs
        return False


def _install_hint(name: str) -> str:
    if name == "torch":
        return "Install the torch extra: `pip install -e '.[torch]'`."
    if name == "transformers":
        return "Install the transformers extra: `pip install -e '.[llama]'` (includes transformers)."
    if name == "llama_cpp":
        return (
            "Install the llama extra: `pip install -e '.[llama]'`, or the official CPU wheel: "
            "`pip install llama-cpp-python --extra-index-url "
            "https://abetlen.github.io/llama-cpp-python/whl/cpu`."
        )
    if name in {"safetensors"}:
        return "Install the torch extra: `pip install -e '.[torch]'`."
    if name == "triton":
        return "FP8 kernels need triton on Linux + CUDA: `pip install -e '.[fp8]'`."
    if name in {"qwen", "kimi", "mistral", "gemma"}:
        return "Install the transformers extra: `pip install -e '.[llama]'`."
    return f"Install the '{name}' package (see docs/ENGINES.md)."


def import_from_directory(directory: Path, module_name: str) -> ModuleType:
    """Import ``module_name`` from ``directory``, resolving sibling imports.

    The vendored DeepSeek-V3 reference implementation (``inference/model.py``)
    does ``from kernel import ...``, so the directory itself must be importable.
    We prepend it to ``sys.path`` and load the module under its real name.
    """

    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise EngineUnavailableError(
            f"The reference implementation directory does not exist: {directory}",
            remediation="Keep the vendored `inference/` directory in the AlphaAI project root, "
            "or point engines.options.<model>.inference_dir at it.",
        )
    text = str(directory)
    if text not in sys.path:
        sys.path.insert(0, text)
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - surfaced as unavailable
        raise EngineUnavailableError(
            f"Could not import '{module_name}' from {directory.name}/: {type(exc).__name__}: {exc}",
            remediation="Install the reference implementation's dependencies "
            "(`pip install -r inference/requirements.txt`) and check the traceback.",
        ) from exc


def torch_dtype(config: AlphaAIConfig, torch_module):
    """Resolve ``runtime.dtype`` into a real ``torch.dtype``."""

    name = _TORCH_DTYPES.get(config.runtime.dtype)
    if name is None:
        # fp8/int*/q* are weight *formats*; they map onto bf16 compute or a
        # GGUF runtime and are handled before we get here.
        name = "bfloat16"
    return getattr(torch_module, name, torch_module.float32)


def resolve_device(config: AlphaAIConfig, torch_module: Any | None = None) -> str:
    """Resolve ``runtime.device`` to a device string usable by torch/llama.cpp."""

    requested = config.runtime.device
    if requested == "cpu":
        return "cpu"
    if torch_module is None:
        torch_module = import_module("torch")
    cuda = bool(getattr(torch_module.cuda, "is_available", lambda: False)())
    mps = bool(
        getattr(getattr(torch_module, "backends", None), "mps", None)
        and torch_module.backends.mps.is_available()
    )
    if requested == "cuda":
        if not cuda:
            raise EngineUnavailableError(
                "runtime.device is 'cuda' but no CUDA device is available.",
                remediation="Set runtime.device = 'auto' or 'cpu', or install a CUDA-capable torch build.",
            )
        return "cuda"
    if requested == "mps":
        if not mps:
            raise EngineUnavailableError(
                "runtime.device is 'mps' but Apple Metal is not available.",
                remediation="Set runtime.device = 'auto' or 'cpu'.",
            )
        return "mps"
    if cuda:
        return "cuda"
    if mps:
        return "mps"
    return "cpu"


def configure_threads(config: AlphaAIConfig, torch_module=None) -> int:
    """Apply the configured CPU thread counts. Returns the effective count."""

    threads = config.runtime.cpu_threads
    if torch_module is None:
        torch_module = import_module("torch")
    effective = threads or config.runtime.torch_num_threads
    if effective and effective > 0:
        torch_module.set_num_threads(int(effective))
        return int(effective)
    return int(torch_module.get_num_threads())


def resolve_model_path(config: AlphaAIConfig, spec_local_paths, candidate: Path | None) -> Path:
    """Return an absolute path for a model directory, or raise a precise error."""

    if candidate is not None and Path(candidate).exists():
        return Path(candidate)
    raise EngineUnavailableError(
        "Model path does not exist locally.",
        remediation="Place the weights under paths.models_dir (see docs/ENGINES.md).",
    )


def is_gguf(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".gguf"


def find_gguf(weights_path: Path) -> Path | None:
    """Locate a single GGUF file at or under ``weights_path``."""

    if is_gguf(weights_path):
        return weights_path
    if not weights_path.is_dir():
        return None
    direct = sorted(weights_path.glob("*.gguf"))
    if direct:
        return direct[0]
    nested = sorted(weights_path.glob("**/*.gguf"))
    return nested[0] if nested else None
