"""Real model installation for AlphaAI.

``alphaai models install <model-id>`` runs this module end to end:

1. **verify the model** — the spec must declare an authoritative source, a file
   name, a size and (when the publisher provides one) a checksum.
2. **verify the hardware** — required vs available RAM, VRAM and storage, using
   the same :mod:`alphaai.engines.hardware` assessment the engines use. If the
   model does not fit, AlphaAI stops and recommends a smaller one instead of
   downloading it.
3. **download** the file from the declared repository (plain HTTPS, no
   third-party client library) into ``paths.models_dir/<model-id>/``.
4. **verify the files** — size, SHA-256 and GGUF magic bytes.
5. **record the metadata** — provenance (source, revision, format, quantisation,
   size, checksum, runtime) is written back into ``configs/models/<id>.json``.
6. **detect the runtime** and make sure the engine for this format is present.
7. **load the model for real** and generate a few tokens, so a model is only ever
   reported as usable after it has actually produced a token.

Nothing here fabricates success: every failure is returned with the exact reason
and the command that fixes it.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from ..config.schema import AlphaAIConfig
from ..engines.hardware import HardwareReport, assess_model, detect_hardware
from .engine import ModelEngine, ModelSpec
from .errors import AlphaAIError
from .types import ChatMessage, GenerationRequest, SamplingParams

#: GGUF files start with these four bytes.
GGUF_MAGIC = b"GGUF"
#: Minimum free space AlphaAI keeps after a download, in GB.
MIN_FREE_DISK_AFTER_GB = 0.5
#: Prompt used for the post-install load test.
LOAD_TEST_PROMPT = "Say hello in one short sentence."

ProgressFn = Callable[[str], None]


@dataclass(slots=True)
class InstallPlan:
    """What installing this model would require, before anything is downloaded."""

    model_id: str
    display_name: str
    source: str
    filename: str
    format: str
    quantization: str
    params_total_b: float
    runtime: str
    runtime_present: bool
    license: str
    license_url: str | None
    weights_url: str | None
    required_ram_gb: float
    available_ram_gb: float
    total_ram_gb: float
    required_storage_gb: float
    available_storage_gb: float
    download_gb: float
    checksum: str | None
    target_path: str
    already_installed: bool
    verdict: str  # ok | already_installed | insufficient
    ok: bool
    reasons: list[str] = field(default_factory=list)
    remediation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "source": self.source,
            "filename": self.filename,
            "format": self.format,
            "quantization": self.quantization,
            "parameters": f"{self.params_total_b:g}B",
            "runtime": self.runtime,
            "runtime_present": self.runtime_present,
            "license": self.license,
            "license_url": self.license_url,
            "weights_url": self.weights_url,
            "download_gb": round(self.download_gb, 3),
            "required_ram_gb": round(self.required_ram_gb, 3),
            "available_ram_gb": round(self.available_ram_gb, 3),
            "total_ram_gb": round(self.total_ram_gb, 3),
            "required_storage_gb": round(self.required_storage_gb, 3),
            "available_storage_gb": round(self.available_storage_gb, 3),
            "checksum": self.checksum,
            "target_path": self.target_path,
            "already_installed": self.already_installed,
            "verdict": self.verdict,
            "ok": self.ok,
            "reasons": list(self.reasons),
            "remediation": self.remediation,
        }

    def report_lines(self) -> list[str]:
        """The pre-download safety report (never download silently)."""

        return [
            f"Model: {self.model_id} ({self.display_name})",
            f"Source: {self.source}",
            f"File: {self.filename}",
            f"Format: {self.format}",
            f"Quantization: {self.quantization}",
            f"Parameters: {self.params_total_b:g}B",
            f"Size: {self.download_gb:.2f} GB",
            f"Runtime: {self.runtime} ({'installed' if self.runtime_present else 'MISSING'})",
            f"Required RAM: {self.required_ram_gb:.2f} GB",
            f"Available RAM: {self.available_ram_gb:.2f} GB of {self.total_ram_gb:.2f} GB",
            f"Required storage: {self.required_storage_gb:.2f} GB",
            f"Available storage: {self.available_storage_gb:.2f} GB",
            f"License: {self.license}",
            f"Target: {self.target_path}",
            f"Verdict: {self.verdict}",
        ]


class ModelInstaller:
    """Installs one registered model's weights and proves they run."""

    def __init__(
        self,
        config: AlphaAIConfig,
        *,
        hardware: HardwareReport | None = None,
        engine_factory: Callable[..., ModelEngine] | None = None,
        progress: ProgressFn | None = None,
    ) -> None:
        self.config = config
        self.hardware = hardware or detect_hardware([config.paths.models_dir, config.paths.state_dir])
        self._engine_factory = engine_factory
        self._progress = progress or (lambda _message: None)

    # -- helpers ----------------------------------------------------------
    def _say(self, message: str) -> None:
        self._progress(message)

    def target_dir(self, spec: ModelSpec) -> Path:
        for raw in spec.local_paths:
            path = Path(raw)
            if not path.is_absolute():
                path = Path(self.config.paths.project_root) / path
            return path
        return Path(self.config.paths.models_dir) / spec.model_id

    def runtime_key(self, spec: ModelSpec) -> str:
        """Which runtime this model's format needs."""

        return "llama_cpp" if spec.engine in {"llama_cpp", "qwen_gguf", "deepseek_gguf"} else spec.engine

    def runtime_present(self, spec: ModelSpec) -> bool:
        from ..engines.helpers import module_present

        key = self.runtime_key(spec)
        if key == "llama_cpp":
            return module_present("llama_cpp")
        if key in {"transformers", "qwen", "kimi", "llama", "mistral", "gemma", "alphaai"}:
            return module_present("torch") and module_present("transformers")
        if key == "deepseek":
            return module_present("torch")
        return True

    def download_target(self, spec: ModelSpec) -> tuple[str, str | None, int | None]:
        """``(filename, url, size)`` declared in the model's provenance."""

        provenance = spec.provenance
        filename = str(provenance.get("filename") or (spec.weight_files[0] if spec.weight_files else ""))
        url = provenance.get("download_url")
        if not url and spec.weights_url and filename:
            url = f"{str(spec.weights_url).rstrip('/')}/resolve/main/{filename}"
        size = provenance.get("file_size_bytes")
        return filename, (str(url) if url else None), (int(size) if size else None)

    # -- planning ---------------------------------------------------------
    def plan(self, spec: ModelSpec) -> InstallPlan:
        filename, url, declared_size = self.download_target(spec)
        provenance = spec.provenance
        runtime_key = self.runtime_key(spec)
        runtime_present = self.runtime_present(spec)
        target = self.target_dir(spec) / filename if filename else self.target_dir(spec)

        download_gb = (declared_size or 0) / 1024**3
        estimate = assess_model(
            params_total_b=spec.params_total_b,
            runtime_requirements=spec.runtime_requirements,
            context_length=spec.context_length,
            hardware=self.hardware,
            kv_bytes_per_token=spec.kv_bytes_per_token,
            quant=spec.quantization,
            disk_path=None,
        )
        required_ram = float((estimate.estimate or {}).get("runtime_ram_gb") or 0.0)
        required_storage = download_gb * 1.05 + MIN_FREE_DISK_AFTER_GB
        free_storage = self.hardware.disk_free_gb.get(str(self.config.paths.models_dir))
        if free_storage is None:
            import shutil

            try:
                free_storage = shutil.disk_usage(str(self.config.paths.models_dir)).free / 1024**3
            except OSError:  # pragma: no cover - unreadable volume
                free_storage = 0.0

        reasons: list[str] = []
        remediation: str | None = None
        already = target.exists() and target.stat().st_size > 0

        if not url:
            reasons.append(
                "This model declares no download URL. AlphaAI only downloads weights from the "
                "authoritative repository recorded in its metadata."
            )
            remediation = (
                "Add provenance.download_url to configs/models/%s.json (or place the weights in %s "
                "yourself and run `alphaai models check %s`)." % (spec.model_id, target.parent, spec.model_id)
            )
        if not spec.weights_published:
            reasons.append(
                f"Model status is '{spec.status}': no weights exist yet to install."
            )
            remediation = remediation or "This model becomes installable once its weights are released."
        if not runtime_present:
            reasons.append(
                f"The runtime for this model ({runtime_key}) is not installed here."
            )
            remediation = remediation or (
                "pip install -e '.[llama]' (or the official CPU wheel: pip install llama-cpp-python "
                "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu)."
            )
        if estimate.verdict == "insufficient" or not estimate.ok:
            reasons.append(
                f"Not enough memory: needs about {required_ram:.2f} GB, "
                f"this machine has {self.hardware.total_ram_gb:.2f} GB total / "
                f"{self.hardware.available_ram_gb:.2f} GB available. "
                + " ".join(estimate.reasons)
            )
            remediation = remediation or estimate.remediation or (
                "Install a smaller quantised model, or move to a machine with more RAM."
            )
        if free_storage and free_storage < required_storage:
            reasons.append(
                f"Not enough disk: needs {required_storage:.2f} GB, {free_storage:.2f} GB free at "
                f"{self.config.paths.models_dir}."
            )
            remediation = remediation or "Free disk space or point paths.models_dir at a larger volume."

        if already and not reasons:
            verdict, ok = "already_installed", True
        elif reasons:
            verdict, ok = "insufficient", False
        else:
            verdict, ok = "ok", True

        return InstallPlan(
            model_id=spec.model_id,
            display_name=spec.display_name,
            source=str(provenance.get("source") or spec.weights_url or "unknown"),
            filename=filename,
            format=str(provenance.get("format") or "gguf"),
            quantization=str(spec.quantization or "unknown"),
            params_total_b=spec.params_total_b,
            runtime=runtime_key,
            runtime_present=runtime_present,
            license=spec.license,
            license_url=spec.license_url,
            weights_url=spec.weights_url,
            required_ram_gb=required_ram,
            available_ram_gb=self.hardware.available_ram_gb,
            total_ram_gb=self.hardware.total_ram_gb,
            required_storage_gb=required_storage,
            available_storage_gb=float(free_storage or 0.0),
            download_gb=download_gb,
            checksum=(str(provenance.get("sha256")) if provenance.get("sha256") else None),
            target_path=str(target),
            already_installed=already,
            verdict=verdict,
            ok=ok,
            reasons=reasons,
            remediation=None if ok else remediation,
        )

    # -- install ----------------------------------------------------------
    def install(
        self,
        spec: ModelSpec,
        *,
        dry_run: bool = False,
        force: bool = False,
        load_test: bool = True,
    ) -> dict[str, Any]:
        plan = self.plan(spec)
        result: dict[str, Any] = {
            "model_id": spec.model_id,
            "ok": False,
            "dry_run": dry_run,
            "plan": plan.to_dict(),
            "plan_lines": plan.report_lines(),
            "downloaded": False,
            "verified": None,
            "runtime": plan.runtime,
            "load_test": None,
            "attribution": spec.attribution,
        }

        if dry_run:
            result["ok"] = plan.ok
            result["status"] = "planned" if plan.ok else "blocked"
            return result

        if not plan.ok and not force:
            result["status"] = "blocked"
            result["error"] = {
                "code": "model_incompatible",
                "message": " ".join(plan.reasons) or "This model cannot be installed here.",
                "remediation": plan.remediation,
            }
            return result

        target = Path(plan.target_path)
        if not plan.already_installed:
            try:
                self._download(
                    plan.source if plan.source.startswith("http") else str(plan.weights_url or ""),
                    plan.filename,
                    target,
                    expected_size=(spec.provenance.get("file_size_bytes") or None),
                )
            except AlphaAIError as exc:
                result["status"] = "download_failed"
                result["error"] = exc.to_dict()
                return result
            result["downloaded"] = True
        else:
            self._say(f"{plan.filename} already present at {target}; verifying it.")

        verification = self.verify(target, expected_sha256=plan.checksum, expected_size=None)
        result["verified"] = verification
        if not verification["ok"]:
            result["status"] = "verification_failed"
            result["error"] = {
                "code": "model_incompatible",
                "message": verification["detail"],
                "remediation": "Delete the file and retry, or download it manually from the source URL.",
            }
            return result

        self._record_provenance(spec, target, verification, plan)
        result["registered"] = spec.model_id

        if load_test:
            test = self._load_test(spec)
            result["load_test"] = test
            if not test["ok"]:
                result["status"] = "load_failed"
                result["error"] = {
                    "code": "engine_load_failed",
                    "message": test["detail"],
                    "remediation": (
                        "The files are present but the model did not load, so AlphaAI does not mark it "
                        "available. Check the runtime version and the file checksum."
                    ),
                }
                return result
            self._record_load_test(spec, test)
            result["attribution"] = spec.attribution

        result["ok"] = True
        result["status"] = "installed"
        return result

    # -- steps ------------------------------------------------------------
    def _download(
        self,
        source: str,
        filename: str,
        target: Path,
        *,
        expected_size: int | None,
    ) -> None:
        url = source if source.endswith(filename) else f"{source.rstrip('/')}/resolve/main/{filename}"
        if not url.startswith("http"):
            raise AlphaAIError(
                f"No downloadable URL for {filename}.",
                remediation="Record provenance.download_url (or weights_url) in the model metadata.",
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        self._say(f"downloading {url}")
        try:
            with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as handle:
                total = int(response.headers.get("content-length") or expected_size or 0)
                written = 0
                last_report = 0.0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    written += len(chunk)
                    now = time.time()
                    if now - last_report >= 5.0 and total:
                        self._say(f"  {written / 1024**3:.2f}/{total / 1024**3:.2f} GB")
                        last_report = now
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            partial.unlink(missing_ok=True)
            raise AlphaAIError(
                f"Download failed from {url}: {type(exc).__name__}: {exc}",
                remediation="Check network access to the model repository and retry.",
            ) from exc
        partial.replace(target)
        self._say(f"downloaded {target}")

    def verify(
        self,
        path: Path,
        *,
        expected_sha256: str | None = None,
        expected_size: int | None = None,
    ) -> dict[str, Any]:
        """Check the downloaded file is really the model it claims to be."""

        if not path.exists():
            return {"ok": False, "path": str(path), "detail": "file is missing"}
        size = path.stat().st_size
        if size <= 0:
            return {"ok": False, "path": str(path), "detail": "file is empty"}
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        checksum = digest.hexdigest()
        result: dict[str, Any] = {
            "ok": True,
            "path": str(path),
            "size_bytes": size,
            "size_gb": round(size / 1024**3, 3),
            "sha256": checksum,
            "detail": "verified",
        }
        if expected_size is not None and size != int(expected_size):
            result.update(ok=False, detail=f"size mismatch: expected {expected_size} bytes, got {size}")
            return result
        if expected_sha256 and checksum.lower() != str(expected_sha256).lower():
            result.update(
                ok=False,
                detail=f"sha256 mismatch: expected {expected_sha256}, got {checksum}",
            )
            return result
        if path.suffix.lower() == ".gguf":
            with open(path, "rb") as handle:
                magic = handle.read(4)
            if magic != GGUF_MAGIC:
                result.update(ok=False, detail=f"not a GGUF file (magic bytes {magic!r})")
                return result
            result["format"] = "gguf"
        return result

    def _record_provenance(
        self,
        spec: ModelSpec,
        path: Path,
        verification: Mapping[str, Any],
        plan: InstallPlan,
    ) -> None:
        """Write the exact install facts back into ``configs/models/<id>.json``."""

        if not spec.source_path:
            return
        source = Path(spec.source_path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):  # pragma: no cover - metadata was valid at load
            return
        provenance = dict(payload.get("provenance") or {})
        provenance.update(
            {
                "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "installed_by": "alphaai models install",
                "local_path": str(path.relative_to(Path(self.config.paths.project_root)))
                if str(path).startswith(str(self.config.paths.project_root))
                else str(path),
                "file_size_bytes": verification.get("size_bytes"),
                "sha256": verification.get("sha256"),
                "runtime": plan.runtime,
                "format": plan.format,
                "quantization": plan.quantization,
            }
        )
        payload["provenance"] = provenance
        source.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self._say(f"recorded provenance in {source}")

    def _record_load_test(self, spec: ModelSpec, test: Mapping[str, Any]) -> None:
        if not spec.source_path:
            return
        source = Path(spec.source_path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):  # pragma: no cover
            return
        provenance = dict(payload.get("provenance") or {})
        provenance["load_test"] = dict(test)
        payload["provenance"] = provenance
        source.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # -- runtime + load ---------------------------------------------------
    def _engine(self, spec: ModelSpec) -> ModelEngine:
        if self._engine_factory is not None:
            return self._engine_factory(spec, self.config)
        from ..engines import engine_class

        options = dict(self.config.engines.options.get(spec.model_id) or {})
        return engine_class(spec.engine)(spec, self.config, options=options, hardware=self.hardware)

    def _load_test(self, spec: ModelSpec, *, max_tokens: int = 12) -> dict[str, Any]:
        """Load the model and generate tokens for real (never a file check)."""

        self._say(f"load test: loading {spec.model_id} through {self.runtime_key(spec)}")
        engine = self._engine(spec)
        started = time.perf_counter()
        try:
            engine.load()
            result = engine.generate(
                GenerationRequest(
                    messages=[ChatMessage(role="user", content=LOAD_TEST_PROMPT)],
                    sampling=SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens),
                    engine_id=spec.model_id,
                    model_id=spec.model,
                )
            )
        except AlphaAIError as exc:
            return {
                "ok": False,
                "detail": exc.message,
                "remediation": exc.remediation,
                "code": exc.code,
            }
        except Exception as exc:  # noqa: BLE001 - any load failure is "not available"
            return {"ok": False, "detail": f"{type(exc).__name__}: {exc}", "code": "engine_load_failed"}
        finally:
            try:
                engine.unload()
            except Exception:  # noqa: BLE001 - unloading must not mask the result
                pass
        text = (result.text or "").strip()
        return {
            "ok": bool(text),
            "detail": "the model loaded and generated text" if text else "the model loaded but produced no text",
            "prompt": LOAD_TEST_PROMPT,
            "text": text,
            "completion_tokens": result.usage.completion_tokens,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "runtime": result.runtime,
            "engine_id": result.engine_id,
            "model": result.model,
            "attribution": result.attribution,
        }


def install_summary(result: Mapping[str, Any]) -> list[str]:
    """Human-readable lines for the CLI."""

    plan = result.get("plan") or {}
    lines = [
        "AlphaAI model install",
        f"Model: {plan.get('model_id')}",
        f"Size: {plan.get('download_gb')} GB",
        f"Parameters: {plan.get('parameters')}",
        f"Quantization: {plan.get('quantization')}",
        f"Runtime: {plan.get('runtime')} ({'installed' if plan.get('runtime_present') else 'MISSING'})",
        f"Required RAM: {plan.get('required_ram_gb')} GB",
        f"Available RAM: {plan.get('available_ram_gb')} GB",
        f"Required storage: {plan.get('required_storage_gb')} GB",
        f"Available storage: {plan.get('available_storage_gb')} GB",
    ]
    if result.get("downloaded"):
        lines.append("Downloaded: yes")
    verification = result.get("verified") or {}
    if verification:
        lines.append(f"Verified: {verification.get('detail')} (sha256 {str(verification.get('sha256'))[:16]}…)")
    test = result.get("load_test") or {}
    if test:
        lines.append(f"Load test: {'passed' if test.get('ok') else 'FAILED'} — {test.get('detail')}")
        if test.get("text"):
            lines.append(f"  sample output: {str(test['text'])[:120]!r}")
    if result.get("error"):
        lines.append(f"Error: {result['error'].get('code')} — {result['error'].get('message')}")
        if result["error"].get("remediation"):
            lines.append(f"Fix: {result['error']['remediation']}")
    lines.append(f"Status: {result.get('status')}")
    return lines


__all__ = ["InstallPlan", "ModelInstaller", "install_summary", "GGUF_MAGIC", "LOAD_TEST_PROMPT"]
