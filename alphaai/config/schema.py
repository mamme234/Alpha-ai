"""AlphaAI configuration schema.

The configuration describes *paths, runtime knobs and policies*. It never
contains AI provider API keys, because AlphaAI performs local inference only.

Design rules
------------
* Every field has a safe default, so AlphaAI runs with an empty config file.
* Policy fields (tools, skills, engines) are explicit allow-lists.
* ``api.redact_paths`` keeps local filesystem layout out of client responses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Device = Literal["auto", "cpu", "cuda", "mps"]
DType = Literal["auto", "bf16", "fp16", "fp32", "fp8", "q8_0", "q4_k_m", "int8", "int4"]

DEFAULT_SYSTEM_PROMPT = (
    "You are AlphaAI, an AI system built on local inference engines. "
    "You answer accurately, state uncertainty plainly, and use tools when a "
    "question needs computation or file/network access you do not have."
)


@dataclass(slots=True)
class PathsConfig:
    """Where AlphaAI reads configuration, models and data from."""

    project_root: str = "."
    models_dir: str = "models"
    configs_dir: str = "configs"
    datasets_dir: str = "datasets"
    tokenizer_dir: str = "tokenizer"
    checkpoints_dir: str = "checkpoints"
    training_dir: str = "training"
    evaluation_dir: str = "evaluation"
    state_dir: str = ".alphaai"
    workspace_dir: str = "workspace"
    log_dir: str = ".alphaai/logs"


@dataclass(slots=True)
class RuntimeConfig:
    """Hardware/runtime knobs shared by every engine."""

    device: Device = "auto"
    dtype: DType = "auto"
    cpu_threads: int = 0  # 0 => let the runtime decide
    gpu_layers: int = -1  # llama.cpp style offload; -1 => all layers if a GPU exists
    max_loaded_engines: int = 1
    allow_model_download: bool = False  # AlphaAI never downloads weights implicitly
    torch_num_threads: int = 0  # 0 => follow cpu_threads
    worker_parallelism: int = 1  # torchrun / process-parallel engines


@dataclass(slots=True)
class SamplingConfig:
    """Real sampling parameters handed to the engines."""

    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 0  # 0 => engine default
    max_tokens: int = 512
    repetition_penalty: float = 1.0
    seed: int | None = None
    stop: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ConversationConfig:
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_history_turns: int = 24
    context_reserve_tokens: int = 1024
    tool_loop_limit: int = 6
    stream: bool = True


@dataclass(slots=True)
class ToolPermissions:
    """Permission boundary for a single tool."""

    allow: bool = False
    network: bool = False
    write: bool = False
    root: str | None = None
    timeout_s: float = 10.0
    max_calls_per_run: int = 32


@dataclass(slots=True)
class ToolsConfig:
    """Tool engine configuration: allow-lists plus per-tool permissions."""

    enabled: list[str] = field(default_factory=lambda: ["*"])
    disabled: list[str] = field(default_factory=list)
    sandbox_root: str = "workspace"
    allow_network: bool = False
    allow_code_execution: bool = False
    allow_writes: bool = False
    default_timeout_s: float = 10.0
    max_output_bytes: int = 65536
    permissions: dict[str, ToolPermissions] = field(default_factory=dict)


@dataclass(slots=True)
class SkillsConfig:
    """Skill manager configuration."""

    enabled: list[str] = field(default_factory=lambda: ["*"])
    disabled: list[str] = field(default_factory=list)
    allow_engine_backed: bool = True  # skills that require a live model engine
    default_timeout_s: float = 30.0
    max_output_bytes: int = 262144


@dataclass(slots=True)
class EnginesConfig:
    """Which engines AlphaAI is allowed to load."""

    enabled: list[str] = field(
        default_factory=lambda: [
            # every engine key AlphaAI ships (see alphaai.engines.ENGINE_CLASSES);
            # add "*" to allow custom engine keys too.
            "deepseek",
            "transformers",
            "llama_cpp",
            "llama",
            "qwen",
            "kimi",
            "mistral",
            "gemma",
            "alphaai",
            "qwen_gguf",
            "deepseek_gguf",
        ]
    )
    disabled: list[str] = field(default_factory=list)
    model_parallel: int = 1
    trust_remote_code: bool = False
    #: Extra args forwarded verbatim to each engine constructor.
    options: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(slots=True)
class RoutingConfig:
    """Model router behaviour.

    ``task_preferences`` maps a task kind to engine/model ids that should win
    when several engines can serve the request. It can only *order* engines that
    already exist in the registry — the router never invents a model.
    """

    task_preferences: dict[str, list[str]] = field(default_factory=dict)
    long_context_threshold_tokens: int = 16_000
    prefer_loaded_engines: bool = True


@dataclass(slots=True)
class MemoryConfig:
    """Persistent memory system configuration."""

    enabled: bool = True
    path: str = ".alphaai/memory/memory.sqlite3"
    max_entries_per_namespace: int = 5000
    default_namespace: str = "default"
    retrieval_limit: int = 5


@dataclass(slots=True)
class ApiConfig:
    """Unified AlphaAI HTTP API configuration.

    When ``inference_url`` is set the API stops loading models locally and
    forwards every ``/api/*`` request to that AlphaAI inference server instead.
    That is how a stateless front-end deployment (for example Vercel, which has
    no persistent disk and cannot keep a model resident) reaches a real local
    inference host. It is never a client-side or provider API URL: it must point
    at another ``alphaai serve`` instance.
    """

    host: str = "0.0.0.0"
    port: int = 8090
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    max_request_bytes: int = 1_048_576
    redact_paths: bool = True
    require_tool_permissions: bool = True
    enable_dashboard: bool = True
    #: Base URL of the AlphaAI inference server (env: ALPHAI_INFERENCE_URL).
    inference_url: str = ""
    #: Shared secret the inference server requires (env: ALPHAI_INFERENCE_TOKEN).
    inference_token: str = ""
    #: Seconds to wait for the inference server before giving up.
    inference_timeout_s: float = 300.0


@dataclass(slots=True)
class TrainingConfigPaths:
    """Defaults for the training foundation."""

    default_tokenizer: str = "tokenizer"
    default_dataset: str = "alphaai-sample"
    default_experiment: str = "alphaai-alpha-1"
    checkpoint_keep: int = 3
    device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    max_steps: int = 100
    seed: int = 1337


@dataclass(slots=True)
class AlphaAIConfig:
    """Root AlphaAI configuration object."""

    name: str = "AlphaAI"
    tagline: str = "Intelligence, built from the ground up."
    version: str = "1.0.0"
    source: str = "<defaults>"
    paths: PathsConfig = field(default_factory=PathsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    engines: EnginesConfig = field(default_factory=EnginesConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    training: TrainingConfigPaths = field(default_factory=TrainingConfigPaths)

    # -- helpers ---------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> list[str]:
        """Return a list of human-readable configuration problems."""

        problems: list[str] = []
        if self.runtime.cpu_threads < 0:
            problems.append("runtime.cpu_threads must be >= 0")
        if self.runtime.max_loaded_engines < 1:
            problems.append("runtime.max_loaded_engines must be >= 1")
        if not 0.0 <= self.sampling.temperature <= 4.0:
            problems.append("sampling.temperature must be between 0.0 and 4.0")
        if not 0.0 < self.sampling.top_p <= 1.0:
            problems.append("sampling.top_p must be within (0.0, 1.0]")
        if self.sampling.max_tokens < 1:
            problems.append("sampling.max_tokens must be >= 1")
        if self.conversation.tool_loop_limit < 1:
            problems.append("conversation.tool_loop_limit must be >= 1")
        if self.api.port < 1 or self.api.port > 65535:
            problems.append("api.port must be a valid TCP port")
        if self.api.inference_url and not self.api.inference_url.startswith(("http://", "https://")):
            problems.append("api.inference_url must be an http(s) URL of an AlphaAI inference server")
        if self.api.inference_timeout_s <= 0:
            problems.append("api.inference_timeout_s must be > 0")
        if self.tools.allow_code_execution and not self.tools.sandbox_root:
            problems.append("tools.sandbox_root is required when code execution is enabled")
        for tool_id, perm in self.tools.permissions.items():
            if perm.timeout_s <= 0:
                problems.append(f"tools.permissions.{tool_id}.timeout_s must be > 0")
        if self.memory.max_entries_per_namespace < 1:
            problems.append("memory.max_entries_per_namespace must be >= 1")
        if self.routing.long_context_threshold_tokens < 1:
            problems.append("routing.long_context_threshold_tokens must be >= 1")
        if self.training.checkpoint_keep < 0:
            problems.append("training.checkpoint_keep must be >= 0")
        return problems
