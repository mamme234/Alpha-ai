"""Real hardware detection and model resource estimation.

AlphaAI refuses to load (and never downloads) a model when the machine cannot
hold it. All numbers here are derived from measured machine properties and from
publisher-declared model metadata — nothing is guessed at runtime about results.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

# Bytes per parameter for the weight formats AlphaAI knows about. GGUF quant
# sizes follow the llama.cpp block layouts (approximate, incl. scale overhead).
BYTES_PER_PARAM: dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "q8_0": 1.0625,
    "q6_k": 0.82,
    "q5_k_m": 0.71,
    "q4_k_m": 0.60,
    "q4_0": 0.58,
    "int4": 0.55,
}

#: Non-weight memory overhead (activations, CUDA/torch context, fragmentation).
OVERHEAD_FACTOR = 1.12
OVERHEAD_MIN_GB = 0.6


#: Known accelerator vendors, matched against the device name / backend. Used
#: only to label the hardware report — nothing in AlphaAI depends on the vendor.
_GPU_VENDORS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("NVIDIA", ("nvidia", "geforce", "quadro", "tesla", "rtx ", "gtx ", "a100", "h100", "v100")),
    ("AMD", ("amd", "radeon", "instinct", "mi300", "mi250")),
    ("Intel", ("intel", "arc a", "arc b", "ponte vecchio", "gaudi")),
    ("Apple", ("apple", "metal", "m1", "m2", "m3", "m4")),
    ("Google", ("tpu", "tensor processing")),
    ("Huawei", ("ascend", "cann")),
)


def guess_gpu_vendor(name: str, backend: str = "") -> str:
    """Best-effort vendor label for a detected accelerator ("unknown" if none)."""

    haystack = f"{name} {backend}".lower()
    for vendor, needles in _GPU_VENDORS:
        if any(needle in haystack for needle in needles):
            return vendor
    return "unknown"


@dataclass(slots=True)
class GpuInfo:
    index: int
    name: str
    vram_gb: float
    backend: str = "cuda"
    vendor: str = ""

    def __post_init__(self) -> None:
        if not self.vendor:
            self.vendor = guess_gpu_vendor(self.name, self.backend)

    def display(self) -> str:
        """One-line description used by the hardware report."""

        memory = f"{self.vram_gb:.2f} GB VRAM" if self.vram_gb else "unified memory"
        return f"{self.name} (vendor: {self.vendor}, {memory}, {self.backend})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "vendor": self.vendor,
            "vram_gb": round(self.vram_gb, 2),
            "backend": self.backend,
        }


@dataclass(slots=True)
class RuntimePresence:
    """Which optional runtimes are importable in this environment."""

    torch: bool = False
    torch_version: str | None = None
    transformers: bool = False
    llama_cpp: bool = False
    triton: bool = False
    numpy: bool = False
    cuda_available: bool = False
    mps_available: bool = False
    device_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "torch": self.torch,
            "torch_version": self.torch_version,
            "transformers": self.transformers,
            "llama_cpp": self.llama_cpp,
            "triton": self.triton,
            "numpy": self.numpy,
            "cuda_available": self.cuda_available,
            "mps_available": self.mps_available,
            "device_count": self.device_count,
        }


@dataclass(slots=True)
class HardwareReport:
    cpu_count: int
    cpu_threads: int
    cpu_name: str
    total_ram_gb: float
    available_ram_gb: float
    gpus: list[GpuInfo] = field(default_factory=list)
    disk_free_gb: dict[str, float] = field(default_factory=dict)
    runtime: RuntimePresence = field(default_factory=RuntimePresence)
    platform: str = ""
    python: str = ""

    @property
    def total_vram_gb(self) -> float:
        return round(sum(gpu.vram_gb for gpu in self.gpus), 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "python": self.python,
            "cpu_name": self.cpu_name,
            "cpu_count": self.cpu_count,
            "cpu_threads": self.cpu_threads,
            "total_ram_gb": round(self.total_ram_gb, 2),
            "available_ram_gb": round(self.available_ram_gb, 2),
            "gpus": [gpu.to_dict() for gpu in self.gpus],
            "total_vram_gb": self.total_vram_gb,
            "disk_free_gb": {k: round(v, 2) for k, v in self.disk_free_gb.items()},
            "runtime": self.runtime.to_dict(),
        }


def _module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken installs
        return False


def _cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def _meminfo() -> tuple[float, float]:
    """Return (total_gb, available_gb) using sysconf, with /proc fallback."""

    try:
        page = os.sysconf("SC_PAGE_SIZE")
        total = os.sysconf("SC_PHYS_PAGES") * page / 1024**3
        available = os.sysconf("SC_AVPHYS_PAGES") * page / 1024**3
        return float(total), float(available)
    except (ValueError, OSError, AttributeError):  # pragma: no cover - non-Linux
        pass
    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split(":")
                if len(parts) == 2:
                    info[parts[0].strip()] = int(parts[1].strip().split()[0])
        total = info.get("MemTotal", 0) / 1024**2
        available = info.get("MemAvailable", info.get("MemFree", 0)) / 1024**2
        return float(total), float(available)
    except OSError:  # pragma: no cover
        return 0.0, 0.0


def _nvidia_gpus() -> list[GpuInfo]:
    if not shutil.which("nvidia-smi"):  # pragma: no cover - depends on host
        return []
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return []
    gpus: list[GpuInfo] = []
    for line in out.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            gpus.append(GpuInfo(index=int(parts[0]), name=parts[1], vram_gb=float(parts[2]) / 1024))
        except ValueError:
            continue
    return gpus


def _torch_devices() -> tuple[list[GpuInfo], bool, bool, int, str | None]:
    """Query torch if it is installed. Returns (gpus, cuda, mps, count, version)."""

    if not _module_present("torch"):
        return [], False, False, 0, None
    try:
        import torch  # imported lazily: AlphaAI core must not require torch
    except Exception:  # pragma: no cover - broken torch install
        return [], False, False, 0, None
    version = getattr(torch, "__version__", None)
    gpus: list[GpuInfo] = []
    cuda = bool(getattr(torch.cuda, "is_available", lambda: False)())
    if cuda:
        for idx in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(idx)
            except Exception:  # pragma: no cover
                continue
            gpus.append(
                GpuInfo(index=idx, name=props.name, vram_gb=props.total_memory / 1024**3, backend="cuda")
            )
    mps = bool(getattr(getattr(torch, "backends", None), "mps", None) and torch.backends.mps.is_available())
    if mps and not gpus:
        gpus.append(GpuInfo(index=0, name="Apple Metal (MPS)", vram_gb=0.0, backend="mps"))
    return gpus, cuda, mps, len(gpus), version


def detect_hardware(extra_paths: Iterable[str | os.PathLike[str]] = ()) -> HardwareReport:
    """Measure the machine AlphaAI is running on."""

    gpus, cuda, mps, count, torch_version = _torch_devices()
    if not gpus:
        gpus = _nvidia_gpus()
        cuda = cuda or bool(gpus)
        count = count or len(gpus)

    total_ram, available_ram = _meminfo()
    cpu_count = os.cpu_count() or 1
    threads = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else cpu_count

    disk: dict[str, float] = {}
    paths = list(extra_paths) or [os.getcwd()]
    for path in paths:
        try:
            target = Path(path)
            while not target.exists() and target.parent != target:
                target = target.parent
            usage = shutil.disk_usage(str(target))
            disk[str(path)] = usage.free / 1024**3
        except OSError:  # pragma: no cover
            continue

    presence = RuntimePresence(
        torch=torch_version is not None,
        torch_version=torch_version,
        transformers=_module_present("transformers"),
        llama_cpp=_module_present("llama_cpp"),
        triton=_module_present("triton"),
        numpy=_module_present("numpy"),
        cuda_available=cuda,
        mps_available=mps,
        device_count=count,
    )
    return HardwareReport(
        cpu_count=cpu_count,
        cpu_threads=threads,
        cpu_name=_cpu_name(),
        total_ram_gb=total_ram,
        available_ram_gb=available_ram,
        gpus=gpus,
        disk_free_gb=disk,
        runtime=presence,
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        python=platform.python_version(),
    )


# ---------------------------------------------------------------------------
# resource estimation
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ResourceEstimate:
    """Estimated memory needed to run one model in one weight format."""

    dtype: str
    params_b: float
    weights_gb: float
    kv_cache_gb: float
    overhead_gb: float
    total_gb: float
    runtime_ram_gb: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "params_b": round(self.params_b, 3),
            "weights_gb": round(self.weights_gb, 3),
            "kv_cache_gb": round(self.kv_cache_gb, 3),
            "overhead_gb": round(self.overhead_gb, 3),
            "total_gb": round(self.total_gb, 3),
            "runtime_ram_gb": round(self.runtime_ram_gb, 3),
        }


def bytes_per_param(dtype: str) -> float:
    key = (dtype or "").lower()
    if key not in BYTES_PER_PARAM:
        raise ValueError(f"Unknown weight format '{dtype}'. Known: {', '.join(sorted(BYTES_PER_PARAM))}")
    return BYTES_PER_PARAM[key]


def estimate_resources(
    *,
    params_total_b: float,
    dtype: str,
    context_tokens: int,
    kv_bytes_per_token: float | None = None,
) -> ResourceEstimate:
    """Estimate weights + KV cache + runtime overhead for a model.

    ``kv_bytes_per_token`` is taken from model metadata when the publisher
    documents the attention layout (e.g. Multi-head Latent Attention); otherwise
    AlphaAI derives it from the declared hidden size and layer count and labels
    the estimate accordingly in its reports.
    """

    bpp = bytes_per_param(dtype)
    weights_gb = params_total_b * bpp
    kv_gb = 0.0
    if kv_bytes_per_token:
        kv_gb = (kv_bytes_per_token * context_tokens) / 1024**3
    overhead_gb = max(OVERHEAD_MIN_GB, weights_gb * (OVERHEAD_FACTOR - 1.0))
    total = weights_gb + kv_gb + overhead_gb
    return ResourceEstimate(
        dtype=dtype,
        params_b=params_total_b,
        weights_gb=weights_gb,
        kv_cache_gb=kv_gb,
        overhead_gb=overhead_gb,
        total_gb=total,
        # CPU/offload paths need the whole model resident in RAM.
        runtime_ram_gb=weights_gb + overhead_gb,
    )


@dataclass(slots=True)
class AvailabilityAssessment:
    """Verdict from comparing a model's requirements against real hardware."""

    ok: bool
    verdict: str  # ok | tight | insufficient | unknown
    reasons: list[str] = field(default_factory=list)
    remediation: str | None = None
    estimate: dict[str, Any] | None = None
    device: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "remediation": self.remediation,
            "estimate": self.estimate,
            "device": self.device,
        }


def assess_model(
    *,
    params_total_b: float,
    runtime_requirements: Mapping[str, Any],
    context_length: int,
    hardware: HardwareReport,
    kv_bytes_per_token: float | None = None,
    requested_device: str = "auto",
    quant: str | None = None,
    disk_path: str | os.PathLike[str] | None = None,
) -> AvailabilityAssessment:
    """Decide whether this machine can really run this model.

    ``runtime_requirements`` is publisher metadata, e.g.
    ``{"bf16": {"min_ram_gb": 1400, "min_vram_gb": 1300}, "q4_k_m": {...}}``.
    """

    reasons: list[str] = []
    remediation: str | None = None

    formats = [key for key in runtime_requirements.keys() if key in BYTES_PER_PARAM]
    if quant:
        chosen = quant
    elif requested_device in {"cuda", "mps"} and "fp8" in formats:
        chosen = "fp8"
    elif "bf16" in formats:
        chosen = "bf16"
    elif formats:
        chosen = formats[0]
    else:
        chosen = "bf16"
        reasons.append("No runtime requirement metadata; assuming bf16 weights.")

    estimate = estimate_resources(
        params_total_b=params_total_b,
        dtype=chosen,
        context_tokens=context_length,
        kv_bytes_per_token=kv_bytes_per_token,
    )
    requirements = dict(runtime_requirements.get(chosen, {}) or {})

    # --- runtime presence -------------------------------------------------
    primary_runtime = requirements.get("runtime")
    if primary_runtime == "torch" and not hardware.runtime.torch:
        reasons.append("PyTorch is not installed (required for this runtime).")
        remediation = "Install the torch extra: `pip install -e '.[torch]'`."
    if primary_runtime == "triton" and not hardware.runtime.triton:
        reasons.append("The FP8 kernel path requires triton, which is not installed.")
        remediation = "Install the fp8 extra on Linux + CUDA: `pip install -e '.[fp8]'`, or use bf16/q4_k_m weights."
    if primary_runtime == "llama_cpp" and not hardware.runtime.llama_cpp:
        reasons.append("llama-cpp-python is not installed (required for GGUF runtimes).")
        remediation = "Install the llama extra: `pip install -e '.[llama]'`."

    # --- device selection -------------------------------------------------
    devices = _usable_devices(hardware, requested_device)
    if not devices:
        reasons.append(f"Requested device '{requested_device}' is not available on this machine.")
        remediation = "Use --device cpu, or install a working GPU runtime."

    device, headroom = _pick_device(devices, estimate, requirements)
    if device is None:
        gpu_note = (
            f"largest device memory is {max(d.memory_gb for d in devices):.1f} GB"
            if devices
            else "no accelerator present"
        )
        reasons.append(
            f"Model needs about {estimate.total_gb:.0f} GB ({chosen}) but {gpu_note}; "
            f"system RAM is {hardware.total_ram_gb:.1f} GB."
        )
        remediation = (
            "Run a smaller model, use a lower-precision GGUF build, shard across more devices "
            "(runtime.worker_parallelism) or move to a machine with enough memory. "
            "AlphaAI will not run a model that does not fit."
        )

    # --- disk -------------------------------------------------------------
    if disk_path is not None:
        free = hardware.disk_free_gb.get(str(disk_path))
        if free is not None:
            needed = estimate.weights_gb * 1.05
            if free < needed:
                reasons.append(f"Only {free:.1f} GB free at {disk_path}; {needed:.1f} GB needed for weights.")
                remediation = remediation or "Free disk space or point paths.models_dir at a larger volume."

    ok = device is not None and not any("not installed" in r or "not available" in r for r in reasons)
    verdict = "ok" if ok and headroom >= 0.25 else ("tight" if ok else "insufficient")
    if not reasons:
        reasons.append(
            f"Estimated {estimate.total_gb:.1f} GB ({chosen}) fits on {device} "
            f"with {max(headroom, 0.0):.1f} GB headroom."
        )
    return AvailabilityAssessment(
        ok=ok,
        verdict=verdict,
        reasons=reasons,
        remediation=None if ok else remediation,
        estimate=estimate.to_dict() | {"requirements": requirements},
        device=device,
    )


@dataclass(slots=True)
class _Device:
    name: str
    memory_gb: float


def _usable_devices(hardware: HardwareReport, requested: str) -> list[_Device]:
    devices: list[_Device] = []
    if requested in {"auto", "cuda"} and hardware.runtime.cuda_available:
        for gpu in hardware.gpus:
            if gpu.backend == "cuda" and gpu.vram_gb > 0:
                devices.append(_Device(f"cuda:{gpu.index}", gpu.vram_gb))
    if requested in {"auto", "mps"} and hardware.runtime.mps_available:
        # Unified memory: Metal can address a large share of system RAM.
        devices.append(_Device("mps", hardware.total_ram_gb * 0.7))
    if requested in {"auto", "cpu", "cuda"}:  # CPU is always a fallback candidate
        devices.append(_Device("cpu", hardware.total_ram_gb))
    return devices


def _pick_device(
    devices: list[_Device],
    estimate: ResourceEstimate,
    requirements: Mapping[str, Any],
) -> tuple[str | None, float]:
    if not devices:
        return None, 0.0
    # Prefer accelerators, then CPU.
    ordered = sorted(devices, key=lambda d: (0 if d.name != "cpu" else 1, -d.memory_gb))
    for device in ordered:
        min_vram = float(requirements.get("min_vram_gb", 0) or 0)
        min_ram = float(requirements.get("min_ram_gb", 0) or 0)
        declared = max(min_vram if device.name != "cpu" else min_ram, 0.0)
        need = max(estimate.total_gb, declared) if device.name != "cpu" else max(estimate.runtime_ram_gb, declared)
        headroom = device.memory_gb - need
        if headroom >= 0:
            return device.name, headroom
    return None, 0.0


# ---------------------------------------------------------------------------
# model-size recommendation
# ---------------------------------------------------------------------------
#: Share of a CPU-only machine's RAM AlphaAI plans to give a model. The rest is
#: left for the OS, the Python runtime and page cache.
CPU_RAM_BUDGET_FRACTION = 0.70
#: Share of an accelerator's VRAM AlphaAI plans to give a model.
GPU_VRAM_BUDGET_FRACTION = 0.90
#: Parameter sizes AlphaAI recommends, in billions (a ladder of practical sizes).
PARAM_LADDER_B = (0.5, 1.5, 3.0, 7.0, 14.0, 32.0, 70.0)


@dataclass(slots=True)
class ModelSizeRecommendation:
    """The largest model AlphaAI considers practical on the measured machine."""

    backend: str
    quantization: str
    budget_gb: float
    overhead_gb: float
    max_params_b: float
    recommended_params_b: float
    total_ram_gb: float
    available_ram_gb: float
    total_vram_gb: float
    cpu_only: bool
    families: tuple[str, ...] = ()
    note: str = ""

    @property
    def fits_nothing(self) -> bool:
        return self.recommended_params_b <= 0.0

    def summary(self) -> str:
        if self.fits_nothing:
            return (
                f"no practical local model ({self.quantization}): "
                f"only {self.budget_gb:.1f} GB usable on {self.backend}"
            )
        return (
            f"up to {self.max_params_b:.1f}B parameters at {self.quantization} "
            f"(recommended: {self.recommended_params_b:g}B on {self.backend})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "quantization": self.quantization,
            "budget_gb": round(self.budget_gb, 2),
            "overhead_gb": round(self.overhead_gb, 2),
            "max_params_b": round(self.max_params_b, 2),
            "recommended_params_b": round(self.recommended_params_b, 2),
            "total_ram_gb": round(self.total_ram_gb, 2),
            "available_ram_gb": round(self.available_ram_gb, 2),
            "total_vram_gb": round(self.total_vram_gb, 2),
            "cpu_only": self.cpu_only,
            "families": list(self.families),
            "note": self.note,
            "summary": self.summary(),
        }


def recommend_model_size(
    hardware: HardwareReport,
    *,
    quant: str = "q4_k_m",
    context_tokens: int = 4096,
    kv_bytes_per_token: float | None = None,
) -> ModelSizeRecommendation:
    """Work out the largest model this machine can actually hold.

    The number is derived from measured RAM/VRAM and the quantisation's bytes per
    parameter, minus runtime overhead and a KV cache of ``context_tokens``. It is
    a *recommendation* used to pick a model to install — never a substitute for
    the real load test AlphaAI runs after installing.
    """

    bpp = bytes_per_param(quant)
    if hardware.gpus and hardware.total_vram_gb > 0:
        backend = hardware.gpus[0].backend
        budget = hardware.total_vram_gb * GPU_VRAM_BUDGET_FRACTION
        cpu_only = False
    else:
        backend = "cpu"
        budget = hardware.total_ram_gb * CPU_RAM_BUDGET_FRACTION
        cpu_only = True

    overhead = max(OVERHEAD_MIN_GB, budget * (OVERHEAD_FACTOR - 1.0))
    kv_gb = (kv_bytes_per_token * context_tokens) / 1024**3 if kv_bytes_per_token else 0.0
    usable = budget - overhead - kv_gb
    max_params = max(0.0, usable / bpp)

    recommended = 0.0
    for size in PARAM_LADDER_B:
        if size <= max_params:
            recommended = size

    families = (
        ("Qwen2.5", "SmolLM2", "Llama-3.2", "Gemma-2", "Phi-3.5")
        if recommended <= 4.0
        else ("Qwen2.5", "Llama-3.1", "Mistral", "Gemma-2", "DeepSeek-V3")
    )
    return ModelSizeRecommendation(
        backend=backend,
        quantization=quant,
        budget_gb=budget,
        overhead_gb=overhead,
        max_params_b=max_params,
        recommended_params_b=recommended,
        total_ram_gb=hardware.total_ram_gb,
        available_ram_gb=hardware.available_ram_gb,
        total_vram_gb=hardware.total_vram_gb,
        cpu_only=cpu_only,
        families=families,
        note=(
            "CPU-only inference: prefer a quantised GGUF model through the llama.cpp engine."
            if cpu_only
            else "Accelerator present: keep the weights resident in VRAM."
        ),
    )


def hardware_block(
    hardware: HardwareReport,
    *,
    storage_paths: Iterable[str | os.PathLike[str]] = (),
    context_tokens: int = 4096,
    kv_bytes_per_token: float | None = None,
) -> dict[str, Any]:
    """The labelled ``AlphaAI Hardware`` summary printed by ``alphaai doctor``.

    Every value is measured on this machine (or read from ``/proc``); the
    recommendation is computed from those measurements.
    """

    recommendation = recommend_model_size(
        hardware, context_tokens=context_tokens, kv_bytes_per_token=kv_bytes_per_token
    )
    storage = dict(hardware.disk_free_gb)
    for path in storage_paths:
        key = str(path)
        if key not in storage:
            try:
                target = Path(path)
                while not target.exists() and target.parent != target:
                    target = target.parent
                storage[key] = shutil.disk_usage(str(target)).free / 1024**3
            except OSError:  # pragma: no cover - unreadable volume
                continue
    os_name = hardware.platform.split(" (")[0].strip() or "unknown"
    machine = platform.machine() or "unknown"
    gpu = ", ".join(gpu.display() for gpu in hardware.gpus) or "none detected (CPU inference only)"
    vendors = sorted({gpu.vendor for gpu in hardware.gpus})
    block = {
        "cpu": (
            f"{hardware.cpu_name} · {hardware.cpu_count} cores ({hardware.cpu_threads} usable) · {machine}"
        ),
        "ram": f"{hardware.total_ram_gb:.2f} GB total · {hardware.available_ram_gb:.2f} GB available",
        "gpu": gpu,
        "gpu_vendor": ", ".join(vendors) if vendors else "none (CPU inference only)",
        "gpu_details": [gpu.to_dict() for gpu in hardware.gpus],
        "vram": f"{hardware.total_vram_gb:.2f} GB" if hardware.gpus else "none (CPU inference only)",
        "storage": {path: f"{free:.2f} GB free" for path, free in storage.items()},
        "os": os_name,
        "architecture": f"{machine}",
        "python": hardware.python,
        "cpu_architecture": machine,
        "runtime": hardware.runtime.to_dict(),
        "runtimes": hardware.runtime.to_dict(),
        "supported_runtimes": [
            name
            for name, present in (
                ("torch", hardware.runtime.torch),
                ("transformers", hardware.runtime.transformers),
                ("llama.cpp", hardware.runtime.llama_cpp),
                ("triton", hardware.runtime.triton),
                ("cuda", hardware.runtime.cuda_available),
                ("mps", hardware.runtime.mps_available),
            )
            if present
        ],
        "recommended_model_size": recommendation.summary(),
        "recommendation": recommendation.to_dict(),
    }
    return block


def describe_hardware_block(hardware: HardwareReport, **kwargs: Any) -> list[str]:
    """Render :func:`hardware_block` as the CLI's labelled hardware report."""

    block = hardware_block(hardware, **kwargs)
    lines = [
        "AlphaAI Hardware",
        f"OS: {block['os']} · Python {block['python']}",
        f"Architecture: {block['architecture']}",
        f"CPU: {block['cpu']}",
        f"RAM: {block['ram']}",
    ]
    if hardware.gpus:
        for gpu in hardware.gpus:
            lines.append(f"GPU: {gpu.display()}")
        lines.append(f"VRAM: {block['vram']} total ({len(hardware.gpus)} device(s))")
    else:
        lines.append(f"GPU: {block['gpu']}")
        lines.append(f"VRAM: {block['vram']}")
    for path, free in block["storage"].items():
        lines.append(f"Storage: {path} — {free}")
    runtime = block["runtimes"]
    lines.append(
        "Runtimes: "
        + ", ".join(
            [
                f"torch {runtime['torch_version'] or 'missing'}",
                f"transformers {'yes' if runtime['transformers'] else 'no'}",
                f"llama.cpp {'yes' if runtime['llama_cpp'] else 'no'}",
                f"triton {'yes' if runtime['triton'] else 'no'}",
                f"cuda {'yes' if runtime['cuda_available'] else 'no'}",
                f"mps {'yes' if runtime['mps_available'] else 'no'}",
            ]
        )
    )
    supported = block["supported_runtimes"]
    lines.append("Supported inference runtimes: " + (", ".join(supported) if supported else "none detected"))
    lines.append(f"Recommended model size: {block['recommended_model_size']}")
    return lines


def describe_hardware(hardware: HardwareReport) -> list[str]:
    """Human-readable hardware summary used by ``alphaai doctor`` and the API."""

    lines = [
        f"{hardware.platform} · Python {hardware.python}",
        f"{hardware.cpu_name} · {hardware.cpu_count} cores ({hardware.cpu_threads} usable)",
        f"RAM {hardware.total_ram_gb:.1f} GB total · {hardware.available_ram_gb:.1f} GB available",
    ]
    if hardware.gpus:
        for gpu in hardware.gpus:
            vram = f"{gpu.vram_gb:.1f} GB VRAM" if gpu.vram_gb else "unified memory"
            lines.append(f"accelerator: {gpu.name} ({vram}, {gpu.backend})")
    else:
        lines.append("accelerator: none detected (CPU inference only)")
    rt = hardware.runtime
    lines.append(
        "runtime: "
        + ", ".join(
            [
                f"torch {rt.torch_version or 'missing'}",
                f"transformers {'yes' if rt.transformers else 'no'}",
                f"llama.cpp {'yes' if rt.llama_cpp else 'no'}",
                f"triton {'yes' if rt.triton else 'no'}",
                f"cuda {'yes' if rt.cuda_available else 'no'}",
            ]
        )
    )
    for path, free in hardware.disk_free_gb.items():
        lines.append(f"disk free at {path}: {free:.1f} GB")
    return lines
