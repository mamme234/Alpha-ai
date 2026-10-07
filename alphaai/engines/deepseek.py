"""The AlphaAI DeepSeek engine.

This engine runs the **DeepSeek-V3** model using DeepSeek's own reference
implementation, vendored unchanged in ``inference/`` (``model.py``, ``kernel.py``,
``convert.py``, ``fp8_cast_bf16.py``), which remains under DeepSeek copyright
(MIT — see ``LICENSE-CODE``). AlphaAI owns the engine layer around it: model
metadata, device/format selection, resource assessment, prompt rendering,
streaming, usage accounting and precise error reporting.

DeepSeek-V3 is **not** an AlphaAI-trained model. AlphaAI never claims otherwise;
see ``ATTRIBUTION.md``.

Real code paths
---------------
* FP8 path   – original DeepSeek-V3 FP8 weights + triton kernels (``kernel.py``).
* BF16 path  – weights converted with ``inference/fp8_cast_bf16.py``. When triton
  is absent the FP8 kernels are replaced by a *guard* module that raises if the
  FP8 path is ever actually executed, so a bf16 run can never silently use
  different numerics.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

from ..branding import DEEPSEEK_V3_ENGINE_NAME
from ..core.engine import ModelEngine
from ..core.errors import EngineLoadError, EngineUnavailableError, GenerationError
from ..core.types import (
    GenerationRequest,
    GenerationResult,
    SamplingParams,
    StreamChunk,
)
from .helpers import (
    configure_threads,
    import_from_directory,
    import_module,
    module_present,
    resolve_device,
)

#: safetensors dtype strings that mean "these weights need the triton FP8 kernels".
_FP8_DTYPES = {"F8_E4M3", "F8_E5M2"}
_BF16_DTYPES = {"BF16"}


class _KernelGuard(ModuleType):
    """Stand-in for ``kernel.py`` when triton is unavailable.

    The bf16 code path never calls these functions. If anything *does* call them
    the run must fail loudly rather than compute with a substitute kernel.
    """

    def __init__(self) -> None:
        super().__init__("kernel")
        self.__doc__ = "AlphaAI guard module: FP8 kernels are unavailable without triton."

    def _refuse(self, *_args: Any, **_kwargs: Any):
        raise EngineUnavailableError(
            "This model requested the FP8 kernel path, but triton is not installed. "
            "AlphaAI refuses to substitute a different kernel.",
            remediation="Install triton on Linux + CUDA (`pip install -e '.[fp8]'`) "
            "or convert the weights to BF16 with inference/fp8_cast_bf16.py.",
        )

    act_quant = _refuse  # type: ignore[assignment]
    weight_dequant = _refuse  # type: ignore[assignment]
    fp8_gemm = _refuse  # type: ignore[assignment]


class DeepSeekEngine(ModelEngine):
    """Real DeepSeek-V3 inference over the vendored reference implementation."""

    engine_key = "deepseek"
    engine_name = DEEPSEEK_V3_ENGINE_NAME

    _model: Any = None
    _tokenizer: Any = None
    _device: str = "cpu"
    _model_args: Any = None
    _weight_dtype: str | None = None

    # -- runtime probing --------------------------------------------------
    def probe_runtime(self) -> tuple[bool, str, str | None]:
        missing = [name for name in ("torch", "transformers", "safetensors") if not module_present(name)]
        if missing:
            return (
                False,
                f"Missing Python runtime(s): {', '.join(missing)}.",
                "Install them with `pip install -r inference/requirements.txt` "
                "(torch, triton, transformers, safetensors) or `pip install -e '.[torch]'`.",
            )
        has_triton = module_present("triton")
        detail = "torch + transformers + safetensors are installed"
        detail += "; triton present (FP8 path available)" if has_triton else "; triton absent (BF16 path only)"
        return True, detail, None

    # -- paths ------------------------------------------------------------
    def reference_dir(self) -> Path:
        override = self.options.get("inference_dir")
        if override:
            return Path(str(override))
        return Path(self.config.paths.project_root) / "inference"

    def inference_config_path(self) -> Path:
        """The vendored architecture config used to build ``ModelArgs``."""

        declared = self.spec.raw.get("inference_config") if self.spec.raw else None
        candidate = Path(str(declared)) if declared else Path("inference/configs/config_v3.1.json")
        if not candidate.is_absolute():
            candidate = Path(self.config.paths.project_root) / candidate
        if not candidate.exists():
            fallback = self.reference_dir() / "configs" / "config_v3.1.json"
            if fallback.exists():
                return fallback
            raise EngineUnavailableError(
                f"The DeepSeek-V3 architecture config was not found: {candidate}",
                remediation="Keep the vendored `inference/configs/` directory, or set "
                "engines.options.<model>.inference_config.",
            )
        return candidate

    def weights_path(self) -> Path:
        found = self.find_weights()
        if found is None:
            raise EngineUnavailableError(
                self.missing_weights_detail(),
                remediation=(
                    "Download DeepSeek-V3 from the official release "
                    f"({self.spec.weights_url or 'https://huggingface.co/deepseek-ai/DeepSeek-V3'}), "
                    "then follow docs/ENGINES.md: place the checkpoint in "
                    f"{Path(self.config.paths.models_dir).name}/{self.id}/ and convert it with "
                    "`python inference/convert.py`. Accept the DeepSeek Model License first "
                    "(LICENSE-MODEL)."
                ),
                details={"engine_id": self.id},
            )
        return found

    def weight_dtype(self) -> str | None:
        """Read the stored dtype of the first tensor in the first shard."""

        if self._weight_dtype is not None:
            return self._weight_dtype
        try:
            from safetensors import safe_open
        except Exception:  # noqa: BLE001 - safetensors missing is reported elsewhere
            return None
        path = self.find_weights()
        if path is None:
            return None
        for shard in sorted(path.glob("*.safetensors")):
            try:
                with safe_open(str(shard), framework="pt", device="cpu") as handle:
                    for key in handle.keys():
                        self._weight_dtype = str(handle.get_slice(key).get_dtype())
                        return self._weight_dtype
            except Exception:  # noqa: BLE001 - unreadable shard: try the next one
                continue
        return None

    # -- loading ----------------------------------------------------------
    def load(self) -> None:
        torch = import_module("torch")
        transformers = import_module("transformers")
        configure_threads(self.config, torch)

        path = self.weights_path()
        device = resolve_device(self.config, torch)
        if device != "cuda":
            raise EngineUnavailableError(
                f"The vendored DeepSeek-V3 reference implementation requires CUDA for real generation "
                f"(resolved device: {device}).",
                remediation=(
                    "Run DeepSeek-V3 on a CUDA machine with enough VRAM/RAM, or use a GGUF build of a "
                    "smaller model through the llama.cpp engine. AlphaAI will not emulate the model on CPU."
                ),
                details={"engine_id": self.id, "device": device},
            )

        weight_dtype = self.weight_dtype()
        stored = str(self.options.get("weight_dtype") or "").upper() or weight_dtype or ""
        is_fp8 = stored in _FP8_DTYPES
        if is_fp8 and not module_present("triton"):
            raise EngineUnavailableError(
                "These are the original FP8 DeepSeek-V3 weights, but triton is not installed, so the "
                "FP8 kernels cannot run.",
                remediation="Install triton on Linux + CUDA (`pip install -e '.[fp8]'`), or convert the "
                "weights with `python inference/fp8_cast_bf16.py --input-fp8-hf-path ... "
                "--output-bf16-hf-path ...` and run the BF16 path.",
                details={"engine_id": self.id, "weight_dtype": stored},
            )

        payload = json.loads(self.inference_config_path().read_text(encoding="utf-8"))
        if is_fp8:
            payload["dtype"] = "fp8"
        else:
            payload.setdefault("dtype", "bf16")
            payload["dtype"] = "bf16" if not is_fp8 else payload["dtype"]

        module = self._load_reference_module(triton_required=is_fp8)
        args = module.ModelArgs(**payload)
        if not is_fp8:
            torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device(device):
                model = module.Transformer(args)
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"{type(exc).__name__}: {exc}"
            self.refresh_status()
            raise EngineLoadError(
                f"{self.name} could not instantiate the DeepSeek-V3 architecture: {self._last_error}",
                remediation="Check that the vendored inference/model.py is intact and torch is CUDA-capable.",
                details={"engine_id": self.id},
            ) from exc

        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(path), trust_remote_code=bool(self.config.engines.trust_remote_code)
            )
            self._load_shards(model, path, torch)
        except EngineUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"{type(exc).__name__}: {exc}"
            self.refresh_status()
            raise EngineLoadError(
                f"{self.name} failed to load the DeepSeek-V3 weights: {self._last_error}",
                remediation="Convert the checkpoint with inference/convert.py (see docs/ENGINES.md).",
                details={"engine_id": self.id},
            ) from exc

        model.eval()
        self._model = model
        self._tokenizer = tokenizer
        self._model_args = args
        self._device = device
        self._loaded = True
        self._last_error = None
        self.refresh_status()

    def _load_reference_module(self, *, triton_required: bool) -> ModuleType:
        directory = self.reference_dir()
        if triton_required or module_present("triton"):
            return import_from_directory(directory, "model")
        guard = _KernelGuard()
        previous = sys.modules.get("kernel")
        sys.modules["kernel"] = guard
        try:
            return import_from_directory(directory, "model")
        finally:
            if previous is not None:
                sys.modules["kernel"] = previous

    def _load_shards(self, model, path: Path, torch) -> None:
        module = import_module("safetensors.torch")
        world_size = int(self.config.runtime.worker_parallelism or 1)
        rank = int(self.options.get("rank") or 0)
        candidates = [
            path / f"model{rank}-mp{world_size}.safetensors",
            path / "model.safetensors",
        ]
        for shard in candidates:
            if shard.exists():
                module.load_model(model, str(shard))
                return
        shards = sorted(path.glob("*.safetensors"))
        if not shards:
            raise EngineUnavailableError(
                f"No safetensors shards found in {Path(path).name}/.",
                remediation="Convert the Hugging Face checkpoint with `python inference/convert.py`.",
            )
        if len(shards) == 1:
            module.load_model(model, str(shards[0]))
            return
        raise EngineUnavailableError(
            f"Found {len(shards)} shards in {Path(path).name}/ but none named "
            f"model{rank}-mp{world_size}.safetensors.",
            remediation="Re-run inference/convert.py with the intended --model-parallel value.",
        )

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        super().unload()

    # -- prompt rendering -------------------------------------------------
    def chat_template(self) -> str | None:
        return self.spec.chat_template

    def count_tokens(self, text: str) -> int | None:
        if self._tokenizer is None:
            return None
        try:
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        except Exception:  # noqa: BLE001
            return None

    def prompt_tokens(self, request: GenerationRequest) -> list[int]:
        """Tokenise the request with DeepSeek's own chat template."""

        tokenizer = self._tokenizer
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        if request.tools:
            schemas = [
                {"type": "function", "function": {"name": t.tool_id, "description": t.description,
                                                  "parameters": t.parameters}}
                for t in request.tools
            ]
            try:
                return list(
                    tokenizer.apply_chat_template(
                        messages, tools=schemas, add_generation_prompt=True
                    )
                )
            except Exception:  # noqa: BLE001 - older tokenizers lack tool templates
                pass
        try:
            return list(tokenizer.apply_chat_template(messages, add_generation_prompt=True))
        except Exception:  # noqa: BLE001
            return list(tokenizer.encode(self.render_chat_prompt(request.messages)))

    def render_chat_prompt(self, messages) -> str:
        if self._tokenizer is None:
            return super().render_chat_prompt(messages)
        payload = [{"role": m.role, "content": m.content} for m in messages]
        try:
            return self._tokenizer.apply_chat_template(
                payload, tokenize=False, add_generation_prompt=True
            )
        except Exception:  # noqa: BLE001
            return super().render_chat_prompt(messages)

    # -- generation -------------------------------------------------------
    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.ensure_ready()
        started = time.perf_counter()
        prompt_tokens = self.prompt_tokens(request)
        generated, finish_reason = self._decode(prompt_tokens, request.sampling, on_token=None)
        text = generated
        usage = self.build_usage(
            request.prompt_text,
            text,
            exact_prompt=len(prompt_tokens),
            exact_completion=None,
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
            extras={"weight_dtype": self.weight_dtype() or "unknown"},
        )

    def stream(self, request: GenerationRequest) -> Iterator[StreamChunk]:
        self.ensure_ready()
        prompt_tokens = self.prompt_tokens(request)
        pending: list[StreamChunk] = []

        def emit(chunk: StreamChunk) -> None:
            pending.append(chunk)

        generated, finish_reason = self._decode(prompt_tokens, request.sampling, on_token=emit)
        for chunk in pending:
            yield chunk
        usage = self.build_usage(
            request.prompt_text, generated, exact_prompt=len(prompt_tokens), exact_completion=None
        )
        yield StreamChunk(
            text="",
            index=len(pending),
            done=True,
            finish_reason=finish_reason,
            usage=usage,
            engine_id=self.id,
            model=self.model,
        )

    def _decode(
        self,
        prompt_tokens: list[int],
        sampling: SamplingParams,
        on_token,
    ) -> tuple[str, str]:
        """Run the real autoregressive loop of the reference implementation."""

        torch = import_module("torch")
        model = self._model
        tokenizer = self._tokenizer
        if model is None or tokenizer is None:
            raise EngineUnavailableError("DeepSeek-V3 is not loaded.", details={"engine_id": self.id})

        max_seq_len = int(getattr(model, "max_seq_len", self.context_length))
        prompt = prompt_tokens
        if len(prompt) >= max_seq_len:
            prompt = prompt[-max_seq_len + 1 :]
        budget = min(int(sampling.max_tokens), max_seq_len - len(prompt))
        eos_id = int(tokenizer.eos_token_id or -1)
        device = self._device

        tokens = torch.full((1, max_seq_len), -1, dtype=torch.long, device=device)
        tokens[0, : len(prompt)] = torch.tensor(prompt, dtype=torch.long, device=device)
        generated_ids: list[int] = []
        decoded = ""
        finish_reason = "length"
        prev_pos = 0
        temperature = float(sampling.temperature)

        try:
            with torch.inference_mode():
                for cur_pos in range(len(prompt), len(prompt) + budget):
                    logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
                    if temperature > 0:
                        next_token = self._sample(logits, temperature, sampling, torch)
                    else:
                        next_token = logits[:, -1, :].argmax(dim=-1)
                    token_id = int(next_token[0])
                    tokens[0, cur_pos] = token_id
                    prev_pos = cur_pos
                    generated_ids.append(token_id)
                    if on_token is not None:
                        piece = tokenizer.decode(generated_ids, skip_special_tokens=True)
                        if len(piece) > len(decoded):
                            on_token(
                                StreamChunk(
                                    text=piece[len(decoded):],
                                    index=len(generated_ids) - 1,
                                    engine_id=self.id,
                                    model=self.model,
                                )
                            )
                        decoded = piece
                    if token_id == eos_id:
                        finish_reason = "stop"
                        break
                    if sampling.stop:
                        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                        if any(stop in text for stop in sampling.stop):
                            finish_reason = "stop"
                            break
        except EngineUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"{type(exc).__name__}: {exc}"
            self.refresh_status()
            raise GenerationError(
                f"{self.name} failed during inference: {self._last_error}",
                details={"engine_id": self.id},
            ) from exc

        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        for stop in sampling.stop:
            position = text.find(stop)
            if position >= 0:
                text = text[:position]
                finish_reason = "stop"
                break
        return text, finish_reason

    @staticmethod
    def _sample(logits, temperature: float, sampling: SamplingParams, torch):
        """Reference-style sampling: temperature, top-p filtering, then Gumbel trick."""

        scores = logits[:, -1, :] / max(temperature, 1e-5)
        if sampling.seed is not None:
            torch.manual_seed(int(sampling.seed))
        if sampling.top_k and sampling.top_k > 0:
            k = min(int(sampling.top_k), scores.shape[-1])
            values, _ = torch.topk(scores, k, dim=-1)
            scores = scores.masked_fill(scores < values[:, [-1]], float("-inf"))
        if 0.0 < sampling.top_p < 1.0:
            sorted_scores, sorted_idx = torch.sort(scores, descending=True, dim=-1)
            cumulative = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1)
            mask = cumulative - torch.softmax(sorted_scores, dim=-1) > sampling.top_p
            sorted_scores = sorted_scores.masked_fill(mask, float("-inf"))
            scores = sorted_scores.scatter(1, sorted_idx, sorted_scores)
        if sampling.repetition_penalty and sampling.repetition_penalty != 1.0:
            scores = scores / float(sampling.repetition_penalty)
        probs = torch.softmax(scores, dim=-1)
        return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1, keepdim=True)

    # -- description ------------------------------------------------------
    def info(self, *, redact_paths: bool = False) -> dict[str, Any]:
        payload = super().info(redact_paths=redact_paths)
        payload.update(
            {
                "reference_implementation": "inference/model.py (DeepSeek, MIT)",
                "weight_dtype": self.weight_dtype() or "unknown",
                "fp8_kernels": module_present("triton"),
            }
        )
        return payload


__all__ = ["DeepSeekEngine"]
