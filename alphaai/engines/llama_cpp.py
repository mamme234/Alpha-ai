"""GGUF engine for AlphaAI (llama.cpp).

Runs quantised GGUF checkpoints — the practical way to run a model on a machine
without a datacentre GPU. AlphaAI wraps ``llama-cpp-python``; it never shells out
to an external server and never downloads a model on its own.

Generation goes through ``llama_cpp.Llama.create_chat_completion`` so the GGUF's
*own* chat template (``tokenizer.chat_template`` in the file) is applied — the
prompt format the model was trained on, not a guess. AlphaAI's tool schemas are
injected as a system-message block (the same mechanism every AlphaAI engine uses,
see :mod:`alphaai.engines.hf_engine`), which keeps tool calling identical across
runtimes and independent of template features.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterator

from ..core.engine import ModelEngine
from ..core.errors import EngineUnavailableError, GenerationError
from ..core.types import (
    ChatMessage,
    GenerationRequest,
    GenerationResult,
    SamplingParams,
    StreamChunk,
)
from .helpers import find_gguf, import_module, module_present
from .hf_engine import TOOL_PROMPT_HEADER

#: Default chat format per model family: used for the deterministic plain-text
#: rendering and reported in metadata. Generation itself uses the template that
#: ships inside the GGUF file.
CHAT_FORMATS = {
    "deepseek": "chatml",
    "qwen": "chatml",
    "kimi": "chatml",
    "llama": "llama-3",
    "mistral": "mistral-instruct",
    "gemma": "gemma",
    "alphaai": "chatml",
}

#: Special tokens for the ChatML rendering (Qwen/DeepSeek/Kimi families).
_CHATML_OPEN = {"system": "<|im_start|>system", "user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}
_CHATML_CLOSE = "<|im_end|>"


class LlamaCppEngine(ModelEngine):
    """Real local inference through ``llama_cpp.Llama``."""

    engine_key = "llama_cpp"
    engine_name = "AlphaAI llama.cpp Engine"

    #: Populated by :meth:`load`.
    _llama: Any = None
    #: Real context size the model was loaded with (llama.cpp may round it up).
    _n_ctx: int = 0
    _n_threads: int = 0
    _gpu_layers: int = 0

    # -- runtime probing --------------------------------------------------
    def probe_runtime(self) -> tuple[bool, str, str | None]:
        if not module_present("llama_cpp"):
            return (
                False,
                "llama-cpp-python is not installed.",
                "Install the llama extra: `pip install -e '.[llama]'`, or the official CPU wheel "
                "(`pip install llama-cpp-python --extra-index-url "
                "https://abetlen.github.io/llama-cpp-python/whl/cpu`).",
            )
        return True, "llama-cpp-python is installed", None

    # -- paths ------------------------------------------------------------
    def gguf_path(self) -> Path:
        """Locate the GGUF file this engine will load."""

        found = self.find_weights()
        if found is None:
            raise EngineUnavailableError(
                self.missing_weights_detail(),
                remediation=(
                    f"Install a GGUF build of {self.model} with `alphaai models install {self.id}`, "
                    f"or place the .gguf file in "
                    f"{Path(self.config.paths.models_dir).name}/{self.id}/ and run "
                    f"`alphaai models check {self.id}`."
                ),
                details={"engine_id": self.id},
            )
        gguf = find_gguf(found)
        if gguf is None:
            raise EngineUnavailableError(
                f"No .gguf file found under {found}.",
                remediation="Point engines.options.<model>.model_path at the .gguf file itself.",
                details={"engine_id": self.id},
            )
        return gguf

    # -- loading ----------------------------------------------------------
    def n_ctx(self) -> int:
        """Context window AlphaAI will request from llama.cpp."""

        requested = self.options.get("n_ctx")
        if requested:
            return int(requested)
        return int(min(self.context_length, 4096))

    def load(self) -> None:
        llama_cpp = import_module("llama_cpp")
        path = self.gguf_path()
        gpu_layers = int(self.config.runtime.gpu_layers)
        if gpu_layers < 0:
            gpu_layers = 0 if not self.hardware.runtime.cuda_available else -1
        threads = int(self.options.get("cpu_threads") or self.config.runtime.cpu_threads or self.hardware.cpu_threads)
        kwargs: dict[str, Any] = {
            "model_path": str(path),
            "n_ctx": self.n_ctx(),
            "n_threads": threads,
            "n_gpu_layers": gpu_layers,
            "verbose": bool(self.options.get("verbose", False)),
        }
        for key in ("n_batch", "n_ubatch", "seed", "use_mmap", "use_mlock", "logits_all"):
            if key in self.options:
                kwargs[key] = self.options[key]
        self._llama = llama_cpp.Llama(**kwargs)
        reported = getattr(self._llama, "n_ctx", None)
        self._n_ctx = int(reported) if isinstance(reported, int) and reported > 0 else int(kwargs["n_ctx"])
        self._n_threads = threads
        self._gpu_layers = gpu_layers
        self._loaded = True
        self._last_error = None
        self.refresh_status()

    def unload(self) -> None:
        self._llama = None
        self._n_ctx = 0
        super().unload()

    # -- prompts ----------------------------------------------------------
    def build_messages(self, request: GenerationRequest) -> list[dict[str, Any]]:
        """Request messages plus the AlphaAI tool block when tools are offered."""

        messages: list[ChatMessage] = list(request.messages)
        if request.tools:
            block = TOOL_PROMPT_HEADER + "\n" + "\n".join(
                f"- {tool.tool_id}: {tool.description} (parameters: {tool.parameters})"
                for tool in request.tools
            )
            if messages and messages[0].role == "system":
                messages[0] = ChatMessage(role="system", content=messages[0].content + "\n\n" + block)
            else:
                messages.insert(0, ChatMessage(role="system", content=block))
        return [self._message_payload(message) for message in messages]

    @staticmethod
    def _message_payload(message: ChatMessage) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.role == "tool":
            # llama.cpp's chat templates expect tool output as user/tool text;
            # keeping the role and a name tag preserves tool transparency.
            payload["role"] = "user"
            payload["content"] = f"[tool result: {message.name or 'tool'}]\n{message.content}"
        return payload

    def render_chat_prompt(self, messages) -> str:
        """Deterministic ChatML rendering of the conversation (plain text).

        Generation uses ``create_chat_completion`` so the GGUF's embedded template
        is applied; this rendering is kept for diagnostics and for callers that
        need the prompt as text.
        """

        parts: list[str] = []
        for message in messages:
            body = message.content
            if message.role == "tool":
                body = f"[tool result: {message.name or 'tool'}]\n{body}"
                parts.append(f"{_CHATML_OPEN['user']}\n{body}{_CHATML_CLOSE}")
                continue
            opener = _CHATML_OPEN.get(message.role, f"<|im_start|>{message.role}")
            parts.append(f"{opener}\n{body}{_CHATML_CLOSE}")
        parts.append(f"{_CHATML_OPEN['assistant']}\n")
        return "\n".join(parts)

    def count_tokens(self, text: str) -> int | None:
        if self._llama is None:
            return None
        try:
            return len(self._llama.tokenize(text.encode("utf-8"), add_bos=False))
        except Exception:  # noqa: BLE001 - tokenizer variance
            return None

    # -- generation -------------------------------------------------------
    def _completion_kwargs(self, sampling: SamplingParams) -> dict[str, Any]:
        return {
            "max_tokens": int(sampling.max_tokens),
            "temperature": float(sampling.temperature),
            "top_p": float(sampling.top_p),
            "top_k": int(sampling.top_k) if sampling.top_k else 40,
            "repeat_penalty": float(sampling.repetition_penalty or 1.0),
            "stop": list(sampling.stop) or None,
            "seed": int(sampling.seed) if sampling.seed is not None else -1,
        }

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.ensure_ready()
        messages = self.build_messages(request)
        sampling = request.sampling
        started = time.perf_counter()
        try:
            response = self._llama.create_chat_completion(
                messages=messages, **self._completion_kwargs(sampling)
            )
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"{type(exc).__name__}: {exc}"
            self.refresh_status()
            raise GenerationError(
                f"{self.name} failed during inference: {self._last_error}",
                details={"engine_id": self.id},
            ) from exc

        choice = (response.get("choices") or [{}])[0]
        text = str((choice.get("message") or {}).get("content") or "")
        usage_payload = response.get("usage") or {}
        prompt = self.render_chat_prompt(request.messages)
        usage = self.build_usage(
            prompt,
            text,
            exact_prompt=usage_payload.get("prompt_tokens"),
            exact_completion=usage_payload.get("completion_tokens"),
        )
        return GenerationResult(
            text=text,
            engine_id=self.id,
            model=self.model,
            provider=self.provider,
            finish_reason=str(choice.get("finish_reason") or "stop"),
            usage=usage,
            latency_ms=(time.perf_counter() - started) * 1000,
            runtime=self.runtime,
            attribution=self.attribution,
            extras={
                "quantization": self.spec.quantization,
                "chat_template": CHAT_FORMATS.get(self.family, "chatml"),
                "n_ctx": self._n_ctx or self.n_ctx(),
                "n_threads": self._n_threads or self.config.runtime.cpu_threads,
                "gpu_layers": self._gpu_layers,
            },
        )

    def stream(self, request: GenerationRequest) -> Iterator[StreamChunk]:
        self.ensure_ready()
        messages = self.build_messages(request)
        sampling = request.sampling
        index = 0
        pieces: list[str] = []
        finish_reason = "stop"
        try:
            for piece in self._llama.create_chat_completion(
                messages=messages, stream=True, **self._completion_kwargs(sampling)
            ):
                choice = (piece.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                text = str(delta.get("content") or "")
                if choice.get("finish_reason"):
                    finish_reason = str(choice["finish_reason"])
                if not text:
                    continue
                pieces.append(text)
                yield StreamChunk(
                    text=text,
                    index=index,
                    engine_id=self.id,
                    model=self.model,
                    finish_reason=None,
                )
                index += 1
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"{type(exc).__name__}: {exc}"
            self.refresh_status()
            raise GenerationError(
                f"{self.name} failed while streaming: {self._last_error}",
                details={"engine_id": self.id},
            ) from exc

        full = "".join(pieces)
        usage = self.build_usage(
            self.render_chat_prompt(request.messages), full, exact_prompt=None, exact_completion=None
        )
        yield StreamChunk(
            text="",
            index=index,
            done=True,
            finish_reason=finish_reason,
            usage=usage,
            engine_id=self.id,
            model=self.model,
        )

    # -- description ------------------------------------------------------
    def info(self, *, redact_paths: bool = False) -> dict[str, Any]:
        payload = super().info(redact_paths=redact_paths)
        payload["weight_format"] = "gguf"
        payload["chat_format"] = CHAT_FORMATS.get(self.family, "chatml")
        payload["quantization"] = self.spec.quantization
        payload["n_ctx"] = self._n_ctx or self.n_ctx()
        try:
            gguf = self.gguf_path()
        except EngineUnavailableError:
            gguf = None
        if gguf is None:
            payload["gguf_path"] = None
        else:
            payload["gguf_path"] = f"<local>/{Path(gguf).name}" if redact_paths else str(gguf)
        payload["load_test"] = self.spec.provenance.get("load_test")
        payload["capability_detail"] = self.capabilities_dict()
        return payload


__all__ = ["LlamaCppEngine", "CHAT_FORMATS"]
