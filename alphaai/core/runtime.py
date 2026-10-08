"""The AlphaAI runtime.

``AlphaRuntime`` is the single object that wires AlphaAI Core together:

    config → hardware → model registry (real engines) → model router
           → tool executor → skill manager → memory → context manager
           → conversation engine → agent orchestrator

Everything the CLI, the API and the dashboard do goes through a runtime, so
availability, permissions and attribution are identical on every surface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from ..branding import ATTRIBUTION_LONG, ATTRIBUTION_SHORT
from ..config.loader import load_config
from ..config.schema import AlphaAIConfig
from ..engines import ENGINE_CLASSES, engine_keys
from ..engines.hardware import (
    HardwareReport,
    describe_hardware,
    describe_hardware_block,
    detect_hardware,
    hardware_block,
)
from .context import ContextManager
from .conversation import ChatOutcome, ConversationEngine, ConversationSession
from .engine import ModelEngine
from .errors import AlphaAIError
from .memory import MemoryStore, memory_from_config
from .orchestrator import AgentOrchestrator
from .registry import ModelRegistry
from .router import ModelRouter
from .skills.base import SkillContext, SkillResult
from .skills.manager import SkillManager, build_default_manager
from .tools.executor import ToolExecutor, build_executor
from .types import ChatMessage, GenerationRequest, GenerationResult, SamplingParams, StreamChunk, ToolResult

logger = logging.getLogger("alphaai.runtime")


@dataclass
class AlphaRuntime:
    """A fully wired AlphaAI instance."""

    config: AlphaAIConfig
    hardware: HardwareReport
    registry: ModelRegistry
    router: ModelRouter
    tools: ToolExecutor
    skills: SkillManager
    memory: MemoryStore | None
    context: ContextManager
    conversation: ConversationEngine
    orchestrator: AgentOrchestrator
    discovery: dict[str, Any] = field(default_factory=dict)
    skill_context: SkillContext | None = None

    # -- construction -----------------------------------------------------
    @classmethod
    def create(
        cls,
        config: AlphaAIConfig | str | None = None,
        *,
        project_root: str | None = None,
        discover: bool = True,
        create_dirs: bool = False,
        engine_factories: Mapping[str, Any] | None = None,
    ) -> "AlphaRuntime":
        if isinstance(config, AlphaAIConfig):
            active = config
        else:
            active = load_config(config, project_root=project_root, create_dirs=create_dirs)

        hardware = detect_hardware([active.paths.models_dir, active.paths.state_dir])
        registry = ModelRegistry(active, hardware=hardware)
        discovery: dict[str, Any] = {}
        if discover:
            report = registry.discover(engine_factories=engine_factories)
            discovery = report.to_dict()
            logger.info(
                "AlphaAI discovered %d engine(s): %s",
                len(report.registered),
                ", ".join(report.registered) or "(none)",
            )

        tools = build_executor(active)
        memory = memory_from_config(active)
        context = ContextManager(active)

        runtime = cls(
            config=active,
            hardware=hardware,
            registry=registry,
            router=ModelRouter(registry, active),
            tools=tools,
            skills=None,  # type: ignore[arg-type] - assigned below
            memory=memory,
            context=context,
            conversation=None,  # type: ignore[arg-type] - assigned below
            orchestrator=None,  # type: ignore[arg-type] - assigned below
            discovery=discovery,
        )

        skill_context = SkillContext(
            config=active,
            tools=tools,
            router=runtime.router,
            registry=registry,
            memory=memory,
            run_id="runtime",
            timeout_s=active.skills.default_timeout_s,
        )
        skill_context.engine_generate = runtime.engine_generate
        runtime.skill_context = skill_context
        runtime.skills = build_default_manager(active, skill_context)
        skill_context.metadata["skill_manager"] = runtime.skills

        runtime.conversation = ConversationEngine(
            active,
            registry=registry,
            router=runtime.router,
            tools=tools,
            skills=runtime.skills,
            memory=memory,
            context=context,
        )
        runtime.orchestrator = AgentOrchestrator(
            active,
            registry=registry,
            conversation=runtime.conversation,
            skills=runtime.skills,
            skill_context_factory=lambda: runtime._fresh_skill_context(),
        )
        return runtime

    # -- skill context ----------------------------------------------------
    def _fresh_skill_context(self) -> SkillContext:
        """A per-run skill context (fresh ``run_id``, same wiring)."""

        base = self.skill_context
        context = SkillContext(
            config=self.config,
            tools=self.tools,
            router=self.router,
            registry=self.registry,
            memory=self.memory,
            engine_generate=self.engine_generate,
            timeout_s=self.config.skills.default_timeout_s,
            metadata={"skill_manager": self.skills},
        )
        if base is not None:
            context.metadata.update({k: v for k, v in base.metadata.items() if k != "skill_manager"})
        return context

    # -- engine bridge ----------------------------------------------------
    def engine_generate(self, prompt: str, options: Mapping[str, Any] | None = None) -> GenerationResult:
        """Route a single prompt to a real engine (used by engine-backed skills)."""

        options = dict(options or {})
        task = options.pop("task", None)
        engine_id = options.pop("engine_id", None)
        system_prompt = options.pop("system_prompt", None)
        messages: list[ChatMessage] = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=str(system_prompt)))
        messages.append(ChatMessage(role="user", content=prompt))

        decision = self.router.choose(messages, task=task, engine_id=engine_id)
        engine = decision.engine(self.registry)
        sampling = self._sampling_from_options(options)
        return engine.generate(
            GenerationRequest(messages=messages, sampling=sampling, engine_id=engine.id)
        )

    def _sampling_from_options(self, options: Mapping[str, Any]) -> SamplingParams:
        base = self.config.sampling
        params = SamplingParams(
            temperature=base.temperature,
            top_p=base.top_p,
            top_k=base.top_k,
            max_tokens=base.max_tokens,
            repetition_penalty=base.repetition_penalty,
            seed=base.seed,
            stop=tuple(base.stop),
        )
        aliases = {"max_new_tokens": "max_tokens"}
        for key, value in options.items():
            target = aliases.get(key, key)
            if value is None or not hasattr(params, target):
                continue
            setattr(params, target, tuple(value) if target == "stop" else value)
        return params

    # -- introspection ----------------------------------------------------
    def health(self) -> dict[str, Any]:
        states: dict[str, int] = {}
        engines = []
        for engine in self.registry.engines():
            payload = engine.info(redact_paths=self.config.api.redact_paths)
            engines.append(payload)
            state = payload["status"]["state"]
            states[state] = states.get(state, 0) + 1
        usable = [engine for engine in engines if engine["status"]["usable"]]
        # ``ok`` means the API server itself is up. Inference is a *separate*
        # question: a running server with no loadable model must never be
        # reported as inference-ready.
        inference = {
            "ready": bool(usable),
            "model_loaded": any(engine["status"]["loaded"] for engine in engines),
            "model_unavailable": bool(engines) and not usable,
            "models_registered": len(engines),
            "usable_models": len(usable),
            "detail": (
                f"{len(usable)} model(s) can serve local inference."
                if usable
                else "No usable model: AlphaAI cannot generate responses yet."
            ),
        }
        return {
            "ok": True,
            "name": self.config.name,
            "inference": inference,
            "tagline": self.config.tagline,
            "version": self.config.version,
            "config_source": self.config.source,
            "attribution": {"short": ATTRIBUTION_SHORT, "full": ATTRIBUTION_LONG},
            "hardware": self.hardware.to_dict(),
            "hardware_lines": describe_hardware(self.hardware),
            "engines": {"registered": len(engines), "usable": len(usable), "states": states, "details": engines},
            "skills": self.skills.summary(),
            "tools": {
                "registered": len(self.tools.registry),
                "permitted": len(self.tools.available_tools()),
            },
            "memory": self.memory.stats() if self.memory else {"enabled": False},
            "orchestrator": self.orchestrator.capabilities(),
            "discovery": self.discovery,
        }

    def models(self) -> list[dict[str, Any]]:
        return [engine.info(redact_paths=self.config.api.redact_paths) for engine in self.registry.engines()]

    def refresh(self) -> dict[str, Any]:
        statuses = self.registry.refresh()
        return {key: status.to_dict() for key, status in statuses.items()}

    def skills_view(self) -> list[dict[str, Any]]:
        return self.skills.describe_all(self._fresh_skill_context())

    def tools_view(self) -> list[dict[str, Any]]:
        return [
            tool.to_dict(self.tools.permissions(tool.tool_id).to_dict())
            for tool in self.tools.registry.list()
        ]

    def capabilities(self) -> dict[str, Any]:
        from .types import TASK_REQUIREMENTS

        return {
            "engine_capabilities": self.registry.capability_matrix(),
            "task_requirements": {
                task: [cap.value for cap in caps] for task, caps in sorted(TASK_REQUIREMENTS.items())
            },
            "engine_keys": {key: ENGINE_CLASSES[key] for key in engine_keys()},
        }

    # -- execution --------------------------------------------------------
    def chat(self, message: str, **kwargs: Any) -> ChatOutcome:
        return self.conversation.chat(message, **kwargs)

    def stream(self, message: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
        return self.conversation.stream(message, **kwargs)

    def create_session(self, **kwargs: Any) -> ConversationSession:
        return self.conversation.create_session(**kwargs)

    def generate(self, prompt: str, **options: Any) -> GenerationResult:
        return self.engine_generate(prompt, options)

    def execute_tool(self, tool_id: str, arguments: Any = None, *, run_id: str = "api") -> ToolResult:
        return self.tools.run(tool_id, arguments, run_id=run_id)

    def execute_skill(self, skill_id: str, inputs: Any = None, *, run_id: str = "") -> SkillResult:
        return self.skills.run_result(
            skill_id, inputs, context=self._fresh_skill_context(), run_id=run_id
        )

    def route(self, message: str, **kwargs: Any) -> dict[str, Any]:
        return self.router.explain(message, **kwargs)

    def doctor(self) -> dict[str, Any]:
        """Diagnostics: what is missing and the exact command that fixes it."""

        runtime_presence = self.hardware.runtime
        findings: list[dict[str, Any]] = []

        if not runtime_presence.torch:
            findings.append(
                {
                    "level": "warning",
                    "message": "PyTorch is not installed: the deepseek and transformers engines cannot run.",
                    "remediation": "pip install -e '.[torch]'",
                }
            )
        if not runtime_presence.llama_cpp:
            findings.append(
                {
                    "level": "info",
                    "message": "llama-cpp-python is not installed: GGUF models cannot run, so the "
                    "lightweight local model cannot be installed or served.",
                    "remediation": (
                        "pip install -e '.[llama]', or on a CPU-only machine use the official "
                        "wheel: pip install llama-cpp-python --extra-index-url "
                        "https://abetlen.github.io/llama-cpp-python/whl/cpu"
                    ),
                }
            )
        if not runtime_presence.cuda_available:
            findings.append(
                {
                    "level": "info",
                    "message": "No CUDA device detected: DeepSeek-V3 (FP8/BF16) cannot run here.",
                    "remediation": "Use a GGUF build through the llama.cpp engine or a CUDA host.",
                }
            )
        for engine in self.registry.engines():
            status = engine.health()
            if not status.usable:
                findings.append(
                    {
                        "level": "warning",
                        "model": engine.id,
                        "state": status.state,
                        "message": status.detail,
                        "remediation": status.remediation,
                        "attribution": engine.attribution,
                    }
                )
        for name in ("models", "workspace", "state", "log"):
            path = Path(getattr(self.config.paths, f"{name}_dir"))
            if not path.exists():
                findings.append(
                    {
                        "level": "info",
                        "message": f"{name} directory does not exist yet: {path.name}/",
                        "remediation": f"mkdir -p {path.name}",
                    }
                )
        problems = self.config.validate()
        for problem in problems:
            findings.append({"level": "error", "message": problem, "remediation": "Fix configs/alphaai.toml."})

        block = hardware_block(self.hardware, storage_paths=[self.config.paths.models_dir])
        return {
            "ok": not [item for item in findings if item["level"] == "error"],
            "config_source": self.config.source,
            "hardware": describe_hardware(self.hardware),
            "hardware_lines": describe_hardware_block(
                self.hardware, storage_paths=[self.config.paths.models_dir]
            ),
            "hardware_block": block,
            "recommended_model_size": block["recommended_model_size"],
            "runtime": runtime_presence.to_dict(),
            "engines": {
                engine.id: engine.health().to_dict() for engine in self.registry.engines()
            },
            "findings": findings,
        }

    def close(self) -> None:
        for engine in self.registry.engines():
            try:
                engine.unload()
            except Exception:  # noqa: BLE001 - shutdown must not raise
                logger.debug("engine %s failed to unload cleanly", engine.id, exc_info=True)
        if self.memory is not None:
            self.memory.close()

    def __enter__(self) -> "AlphaRuntime":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


__all__ = ["AlphaRuntime"]
