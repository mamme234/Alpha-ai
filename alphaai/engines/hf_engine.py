"""Hugging Face ``transformers`` engine for AlphaAI.

This engine runs any causal-LM checkpoint that ``transformers`` can load
(``safetensors`` / ``bin``) on CPU, CUDA or Apple Metal. The AlphaAI family
engines (Qwen, Kimi, Llama, Mistral, Gemma, AlphaAI-X) are thin subclasses of
this class: identical real code path, different declared model metadata.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Iterator

from ..core.engine import ModelEngine, ModelSpec
from ..core.errors import EngineUnavailableError, GenerationError
from ..core.types import (
    Capability,
    ChatMessage,
    GenerationRequest,
    GenerationResult,
    SamplingParams,
    StreamChunk,
)
from .helpers import (
    configure_threads,
    import_module,
    module_present,
    resolve_device,
    torch_dtype,
)

#: Appended to the system message when a request supplies tools, so that models
#: without a native tool template still see the available tool schemas.
TOOL_PROMPT_HEADER = (
    "You can call tools. Reply with a single JSON object of the form "
    '{"tool": "<tool_id>", "arguments": {...}} when a tool is required, '
    "or answer normally otherwise. Available tools:"
)


class TransformersEngine(ModelEngine):
    """Real local inference through ``transformers.AutoModelForCausalLM``."""

    engine_key = "transformers"
    engine_name = "AlphaAI Transformers Engine"
    #: Extra ``from_pretrained`` kwargs a family subclass may pin.
    family_kwargs: dict[str, Any] = {}
    #: Subclasses may require a specific model class.
    model_class_name = "AutoModelForCausalLM"

    #: Populated by :meth:`load`; declared here so health checks stay safe
    #: before the model is ever loaded.
    _model: Any = None
    _tokenizer: Any = None
    _device: str = "cpu"

    # -- runtime probing --------------------------------------------------
    def probe_runtime(self) -> tuple[bool, str, str | None]:
        missing = [name for name in ("torch", "transformers") if not module_present(name)]
        if missing:
            return (
                False,
                f"Missing Python runtime(s): {', '.join(missing)}.",
                "Install them with `pip install -e '.[llama]'` (includes torch + transformers).",
            )
        return True, "torch + transformers are installed", None

    # -- paths ------------------------------------------------------------
    def weights_path(self) -> Path:
        """The local checkpoint directory this engine will load."""

        found = self.find_weights()
        if found is None:
            raise EngineUnavailableError(
                self.missing_weights_detail(),
                remediation=self.missing_weights_remediation(),
                details={"engine_id": self.id, "model": self.model},
            )
        return found

    # -- loading ----------------------------------------------------------
    def load(self) -> None:
        torch = import_module("torch")
        transformers = import_module("transformers")
        configure_threads(self.config, torch)

        path = self.weights_path()
        device = resolve_device(self.config, torch)
        dtype = torch_dtype(self.config, torch)
        trust = bool(self.config.engines.trust_remote_code)
        self._tokenizer = self._load_tokenizer(transformers, path, trust)
        self._model = self._load_model(transformers, torch, path, trust, dtype)
        if device != "cpu":
            self._model.to(device)
        self._model.eval()
        self._device = device
        self._loaded = True
        self._last_error = None
        self.refresh_status()

    def _load_tokenizer(self, transformers, path: Path, trust: bool):
        try:
            return transformers.AutoTokenizer.from_pretrained(
                str(path), trust_remote_code=trust
            )
        except Exception as exc:  # noqa: BLE001
            raise EngineUnavailableError(
                f"The tokenizer for {self.model} could not be loaded from {Path(path).name}/: "
                f"{type(exc).__name__}: {exc}",
                remediation="Download the full checkpoint including tokenizer files "
                "(tokenizer.json / tokenizer_config.json).",
            ) from exc

    def _load_model(self, transformers, torch, path: Path, trust: bool, dtype):
        model_cls = getattr(transformers, self.model_class_name)
        kwargs: dict[str, Any] = {
            "trust_remote_code": trust,
            "low_cpu_mem_usage": module_present("accelerate"),
            **self.family_kwargs,
        }
        override = self.options.get("model_class")
        if override:
            model_cls = getattr(transformers, str(override))
        # ``torch_dtype`` is the long-standing name; recent transformers versions
        # renamed it to ``dtype``. Try both so AlphaAI works on either.
        for key in ("torch_dtype", "dtype"):
            try:
                return model_cls.from_pretrained(str(path), **{**kwargs, key: dtype})
            except TypeError as exc:
                if key == "dtype":
                    raise EngineUnavailableError(
                        f"{self.model} could not be loaded: {exc}",
                        remediation="Check the transformers version and the checkpoint files.",
                    ) from exc
                continue
        raise EngineUnavailableError(
            f"{self.model} could not be loaded from {Path(path).name}/.",
            remediation="Run `alphaai models check %s` for details." % self.id,
        )

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        super().unload()

    # -- prompt rendering -------------------------------------------------
    def render_chat_prompt(self, messages) -> str:
        tokenizer = getattr(self, "_tokenizer", None)
        if tokenizer is None or not getattr(tokenizer, "chat_template", None):
            return super().render_chat_prompt(messages)
        rendered = [self._message_payload(message) for message in messages]
        try:
            return tokenizer.apply_chat_template(
                rendered, tokenize=False, add_generation_prompt=True
            )
        except Exception:  # noqa: BLE001 - fall back to the deterministic rendering
            return super().render_chat_prompt(messages)

    @staticmethod
    def _message_payload(message: ChatMessage) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            payload["tool_calls"] = [call.to_dict() for call in message.tool_calls]
        return payload

    def build_prompt(self, request: GenerationRequest) -> str:
        """Render the request, including tool schemas when tools are supplied."""

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
        return self.render_chat_prompt(messages)

    # -- tokenisation -----------------------------------------------------
    def count_tokens(self, text: str) -> int | None:
        tokenizer = getattr(self, "_tokenizer", None)
        if tokenizer is None:
            return None
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:  # noqa: BLE001 - tokenizer oddities are not fatal
            return None

    # -- generation -------------------------------------------------------
    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.ensure_ready()
        torch = import_module("torch")
        prompt = self.build_prompt(request)
        sampling = request.sampling
        started = time.perf_counter()
        try:
            encoded = self._tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.to(self._device) for key, value in encoded.items()}
            with torch.inference_mode():
                output = self._model.generate(**inputs, **self._generation_kwargs(sampling, torch))
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"{type(exc).__name__}: {exc}"
            self.refresh_status()
            raise GenerationError(
                f"{self.name} failed during inference: {self._last_error}",
                details={"engine_id": self.id, "model": self.model},
            ) from exc

        prompt_len = int(inputs["input_ids"].shape[-1])
        generated = output[0][prompt_len:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        text, finish_reason = self._apply_stops(text, sampling, generated)
        completion_tokens = int(generated.shape[-1])
        usage = self.build_usage(
            prompt, text, exact_prompt=prompt_len, exact_completion=completion_tokens
        )
        return GenerationResult(
            text=text,
            engine_id=self.id,
            model=self.model,
            provider=self.provider,
            finish_reason=finish_reason,
            usage=usage,
            latency_ms=(time.perf_counter() - started) * 1000,
            runtime=f"{self.engine_key}/{self._device}",
            attribution=self.attribution,
        )

    def stream(self, request: GenerationRequest) -> Iterator[StreamChunk]:
        self.ensure_ready()
        torch = import_module("torch")
        transformers = import_module("transformers")
        prompt = self.build_prompt(request)
        sampling = request.sampling
        streamer = transformers.TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        encoded = self._tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in encoded.items()}
        prompt_len = int(inputs["input_ids"].shape[-1])
        error: list[BaseException] = []

        def _run() -> None:
            try:
                with torch.inference_mode():
                    self._model.generate(
                        **inputs,
                        streamer=streamer,
                        **self._generation_kwargs(sampling, torch),
                    )
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller thread
                error.append(exc)
                streamer.end()

        thread = threading.Thread(target=_run, name=f"alphaai-stream-{self.id}", daemon=True)
        thread.start()

        index = 0
        collected: list[str] = []
        finish_reason = "stop"
        for piece in streamer:
            if not piece:
                continue
            text, stopped = self._split_stop(piece, collected, sampling)
            if text:
                collected.append(text)
                yield StreamChunk(
                    text=text,
                    index=index,
                    engine_id=self.id,
                    model=self.model,
                )
                index += 1
            if stopped:
                finish_reason = "stop"
                break
        thread.join(timeout=1.0)
        if error:
            self._last_error = f"{type(error[0]).__name__}: {error[0]}"
            self.refresh_status()
            raise GenerationError(
                f"{self.name} failed while streaming: {self._last_error}",
                details={"engine_id": self.id},
            )
        full = "".join(collected)
        usage = self.build_usage(
            prompt, full, exact_prompt=prompt_len, exact_completion=None
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

    # -- internals --------------------------------------------------------
    def _generation_kwargs(self, sampling: SamplingParams, torch) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": int(sampling.max_tokens),
            "do_sample": sampling.temperature > 0,
            "pad_token_id": getattr(self._tokenizer, "pad_token_id", None)
            or getattr(self._tokenizer, "eos_token_id", None),
        }
        if sampling.temperature > 0:
            kwargs["temperature"] = float(sampling.temperature)
            kwargs["top_p"] = float(sampling.top_p)
            if sampling.top_k and sampling.top_k > 0:
                kwargs["top_k"] = int(sampling.top_k)
            if sampling.repetition_penalty and sampling.repetition_penalty != 1.0:
                kwargs["repetition_penalty"] = float(sampling.repetition_penalty)
        if sampling.seed is not None:
            torch.manual_seed(int(sampling.seed))
        return kwargs

    def _apply_stops(
        self, text: str, sampling: SamplingParams, generated
    ) -> tuple[str, str]:
        finish_reason = "length" if int(generated.shape[-1]) >= int(sampling.max_tokens) else "stop"
        for stop in sampling.stop:
            position = text.find(stop)
            if position >= 0:
                return text[:position], "stop"
        return text, finish_reason

    def _split_stop(
        self, piece: str, collected: list[str], sampling: SamplingParams
    ) -> tuple[str, bool]:
        if not sampling.stop:
            return piece, False
        combined = "".join(collected) + piece
        for stop in sampling.stop:
            position = combined.find(stop)
            if position >= 0:
                keep = combined[:position]
                already = sum(len(item) for item in collected)
                return keep[already:], True
        return piece, False


__all__ = ["TransformersEngine", "TOOL_PROMPT_HEADER"]
