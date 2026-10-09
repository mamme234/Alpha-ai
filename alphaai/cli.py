"""AlphaAI command-line interface.

    alphaai doctor                     # what works here, and the fix for what does not
    alphaai models list                # every engine + live status
    alphaai models check qwen2.5-0.5b-instruct
    alphaai chat "explain MLA in one paragraph"
    alphaai chat --stream              # interactive, streaming
    alphaai tools list / run calculator.evaluate '{"expression": "2+2"}'
    alphaai skills list / run skill.code_analysis '{"code": "print(1)"}'
    alphaai route "translate this to french"
    alphaai orchestrate --goal "..."   # agent loop (needs a usable engine)
    alphaai train validate             # training foundation checks
    alphaai db status                  # PostgreSQL/Supabase connection + migrations
    alphaai db migrate                 # apply pending SQL migrations
    alphaai serve                      # FastAPI on api.host:api.port
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .branding import (
    ATTRIBUTION_SHORT,
    DISPLAY_NAME,
    NAME,
    TAGLINE,
    attribution_lines,
    banner,
    powered_by,
)
from .core.errors import AlphaAIError
from .core.runtime import AlphaRuntime
from .version import __version__


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _load_input(raw: str | None) -> dict[str, Any]:
    """Parse ``--input`` as inline JSON or ``@path/to/file.json``."""

    if not raw:
        return {}
    text = raw
    if raw.startswith("@"):
        text = Path(raw[1:]).read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON input: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("Input JSON must be an object.")
    return payload


def _dump(payload: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _runtime(args: argparse.Namespace) -> AlphaRuntime:
    return AlphaRuntime.create(
        getattr(args, "config", None),
        project_root=getattr(args, "project_root", None),
        create_dirs=True,
    )


def _print_attribution() -> None:
    for line in attribution_lines():
        print(f"  {line}")


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_version(_args: argparse.Namespace) -> int:
    print(f"{DISPLAY_NAME} {__version__}")
    print(f"{TAGLINE}")
    print(f"{ATTRIBUTION_SHORT}")
    return 0


def cmd_attribution(_args: argparse.Namespace) -> int:
    from .branding import ATTRIBUTION_ENGINE, ATTRIBUTION_LONG, DESCRIPTION

    print(f"{DISPLAY_NAME}: {DESCRIPTION}\n")
    print(ATTRIBUTION_ENGINE)
    print()
    print(ATTRIBUTION_LONG)
    print("\nPreserved in this repository: LICENSE-CODE (MIT), LICENSE-MODEL, NOTICE, ATTRIBUTION.md")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    runtime = _runtime(args)
    report = runtime.doctor()
    if not args.json:
        for line in report.get("hardware_lines") or []:
            print(line)
        print()
    _dump(report, True)
    return 0 if report["ok"] else 1


def cmd_config(args: argparse.Namespace) -> int:
    runtime = _runtime(args)
    if args.action == "paths":
        from dataclasses import asdict

        payload = asdict(runtime.config.paths)
    elif args.action == "validate":
        problems = runtime.config.validate()
        payload = {"ok": not problems, "source": runtime.config.source, "problems": problems}
        _dump(payload, args.json)
        return 0 if not problems else 1
    else:
        from .config.loader import public_config_view

        payload = public_config_view(runtime.config)
    _dump(payload, args.json)
    return 0


def cmd_db(args: argparse.Namespace) -> int:
    """Inspect and migrate AlphaAI's PostgreSQL (Supabase) schema.

    Loads configuration only — no model, no weights — so it is safe to run on a
    database host that has never run inference.
    """

    from .config.loader import load_config, public_config_view
    from .db import open_database
    from .db.migrate import discover_migrations

    config = load_config(
        getattr(args, "config", None), project_root=getattr(args, "project_root", None)
    )
    store = open_database(config)
    public = public_config_view(config)["database"]

    if args.action == "status":
        payload = {"ok": True, "database": store.status(), "config": public}
        if args.json:
            _dump(payload, True)
            return 0
        status = payload["database"]
        print(f"configured: {status.get('configured')}")
        if status.get("configured"):
            print(f"host: {status.get('host')}:{status.get('port')} ({status.get('mode')})")
            print(f"database: {status.get('database')} as {status.get('user')} (sslmode {status.get('sslmode')})")
        print(f"reachable: {status.get('reachable')}")
        if not status.get("reachable"):
            if status.get("configured"):
                error = status.get("error") or {}
                print(f"  [{error.get('code', 'unknown')}] {error.get('message')}")
                if error.get("remediation"):
                    print(f"  fix: {error['remediation']}")
            else:
                print(f"  {status.get('detail')}")
                if status.get("remediation"):
                    print(f"  fix: {status['remediation']}")
        migrations = status.get("migrations")
        if migrations:
            print(f"migrations applied: {migrations.get('applied')} (latest {migrations.get('latest')})")
            if migrations.get("pending"):
                print(f"  pending: {', '.join(migrations['pending'])}")
            if migrations.get("drift"):
                for item in migrations["drift"]:
                    print(f"  drift: {item['name']} changed after it was applied")
        return 0

    # plan / migrate
    directory = args.directory or config.paths.migrations_dir
    migrations = discover_migrations(directory)
    if not migrations:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": {
                        "code": "no_migrations_found",
                        "message": f"No migration files found in {directory}.",
                        "remediation": "Run from the AlphaAI repository, or pass --directory.",
                    },
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1

    if args.action == "plan":
        # Reading the ledger needs a reachable database; without one the plan is
        # simply "every file on disk is unapplied", which is the truth.
        state = store.migration_state() if store.configured else {}
        pending = state.get("pending") or [item.name for item in migrations]
        payload = {
            "ok": True,
            "directory": str(directory),
            "database_configured": store.configured,
            "applied": state.get("applied"),
            "latest": state.get("latest"),
            "drift": state.get("drift") or [],
            "migrations": [item.to_dict() for item in migrations],
            "pending": pending,
            "note": (
                "Plan only: nothing applied. Run `alphaai db migrate` to apply pending "
                "migrations (each file is one transaction)."
            ),
        }
        if not args.json:
            for item in migrations:
                marker = "applied" if item.name not in pending else "pending"
                print(f"{item.name:<56} {marker}")
            if payload["drift"]:
                for item in payload["drift"]:
                    print(f"drift: {item['name']} changed after it was applied")
            return 0
        _dump(payload, True)
        return 0

    try:
        report = store.migrate(directory=args.directory)
    except AlphaAIError as exc:
        _dump({"ok": False, "error": exc.to_dict()}, True)
        return 1
    _dump(report, True)
    return 0 if report.get("ok", True) else 1


def cmd_models(args: argparse.Namespace) -> int:
    runtime = _runtime(args)
    if args.action == "list":
        models = runtime.models()
        if args.json:
            _dump({"count": len(models), "models": models}, True)
            return 0
        for model in models:
            status = model["status"]
            print(
                f"{model['id']:<32} {status['state'].upper():<12} {model['engine_name']:<34} "
                f"{model['model_owner']}"
            )
            if status["detail"]:
                print(f"    {status['detail']}")
            if status["remediation"]:
                print(f"    fix: {status['remediation']}")
        return 0

    if args.action == "check":
        targets = [args.model_id] if args.model_id else [engine.id for engine in runtime.registry.engines()]
        statuses = {}
        for model_id in targets:
            engine = runtime.registry.get(model_id)
            status = engine.health(refresh=True)
            statuses[model_id] = status.to_dict()
        if args.json:
            _dump(statuses, True)
        else:
            for model_id, status in statuses.items():
                print(f"{model_id}: {status['state']} — {status['detail']}")
                if status["remediation"]:
                    print(f"  fix: {status['remediation']}")
                if status["extras"].get("hardware"):
                    estimate = status["extras"]["hardware"].get("estimate") or {}
                    if estimate:
                        print(
                            f"  estimate: {estimate.get('total_gb')} GB at {estimate.get('dtype')} "
                            f"→ {status['extras']['hardware'].get('verdict')}"
                        )
        # `models check` is a diagnostic: a model that cannot run here is reported with
        # its remediation instead of failing the command.
        return 0

    if args.action == "show":
        engine = runtime.registry.get(args.model_id)
        _dump(engine.info(redact_paths=False), True)
        return 0

    if args.action == "install":
        from .core.installer import ModelInstaller, install_summary

        if not args.model_id:
            print("models install needs a model id, e.g. `alphaai models install qwen2.5-0.5b-instruct-gguf`.")
            print("Run `alphaai models list` to see every registered model.")
            return 2
        spec = runtime.registry.spec(args.model_id)
        installer = ModelInstaller(
            runtime.config,
            hardware=runtime.hardware,
            progress=lambda message: print(f"  {message}", file=sys.stderr),
        )
        result = installer.install(
            spec,
            dry_run=args.dry_run,
            force=args.force,
            load_test=not args.no_load_test,
        )
        if args.json:
            _dump(result, True)
        else:
            for line in install_summary(result):
                print(line)
        # Re-probe so the reported availability comes from the live engine.
        runtime.registry.discover(replace=True)
        status = runtime.registry.get(spec.model_id).health(refresh=True)
        print(f"Availability: {status.state.upper()} — {status.detail}")
        if status.remediation and not status.usable:
            print(f"Fix: {status.remediation}")
        return 0 if result["ok"] else 1
    return 2


def cmd_skills(args: argparse.Namespace) -> int:
    runtime = _runtime(args)
    if args.action == "list":
        skills = runtime.skills_view()
        if args.json:
            _dump({"count": len(skills), "skills": skills}, True)
            return 0
        for skill in skills:
            flag = "available" if skill.get("available") else "unavailable"
            print(f"{skill['id']:<26} {skill['category']:<14} {flag:<12} {skill['name']}")
            if not skill.get("available"):
                print(f"    {skill.get('availability_reason')}")
        return 0
    if args.action == "run":
        result = runtime.execute_skill(args.skill_id, _load_input(args.input), run_id="cli")
        _dump(result.to_dict(), True)
        return 0 if result.ok else 1
    return 2


def cmd_tools(args: argparse.Namespace) -> int:
    runtime = _runtime(args)
    if args.action == "list":
        tools = runtime.tools_view()
        if args.json:
            _dump({"count": len(tools), "tools": tools}, True)
            return 0
        for tool in tools:
            decision = tool["permissions"]
            flag = "permitted" if decision.get("allowed") else "denied"
            print(f"{tool['id']:<22} {tool['category']:<12} {flag:<10} {tool['name']}")
        return 0
    if args.action == "run":
        result = runtime.execute_tool(args.tool_id, _load_input(args.input), run_id="cli")
        _dump(result.to_dict(), True)
        return 0 if result.ok else 1
    if args.action == "calls":
        _dump(runtime.tools.log.to_dict(), True)
        return 0
    return 2


def cmd_route(args: argparse.Namespace) -> int:
    runtime = _runtime(args)
    _dump(runtime.route(args.text, task=args.task), True)
    return 0


def inference_lines(*, model: str, runtime: str, engine_id: str, attribution: str) -> list[str]:
    """Transparent inference provenance: AlphaAI never hides the real model."""

    return [
        f"Model: {model}",
        f"Runtime: {runtime or 'local'}",
        "Mode: Local",
        powered_by(model),
        f"Engine: {engine_id} · {attribution}",
    ]


def cmd_infer(args: argparse.Namespace) -> int:
    runtime = _runtime(args)
    try:
        result = runtime.generate(
            args.prompt,
            engine_id=args.engine,
            task=args.task,
            **{"max_tokens": args.max_tokens, "temperature": args.temperature},
        )
    except AlphaAIError as exc:
        print(json.dumps({"ok": False, "error": exc.to_dict()}, indent=2))
        return 1
    print(result.text)
    if not args.quiet:
        for line in inference_lines(
            model=result.model,
            runtime=result.runtime,
            engine_id=result.engine_id,
            attribution=result.attribution,
        ):
            print(line, file=sys.stderr)
        print(
            f"[{result.usage.total_tokens} tokens ({result.usage.source}) · "
            f"{result.latency_ms:.0f} ms]",
            file=sys.stderr,
        )
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    runtime = _runtime(args)
    session = runtime.create_session(system_prompt=args.system, engine_id=args.engine)

    def one(message: str) -> int:
        try:
            if args.stream:
                for event in runtime.stream(
                    message,
                    session=session,
                    engine_id=args.engine,
                    task=args.task,
                    use_tools=not args.no_tools,
                    use_memory=not args.no_memory,
                    sampling={"temperature": args.temperature, "max_tokens": args.max_tokens},
                ):
                    if event["type"] == "delta":
                        print(event["text"], end="", flush=True)
                    elif event["type"] == "route":
                        print(f"[router] {event['routing']['reason']}", file=sys.stderr)
                    elif event["type"] == "tool_call":
                        names = ", ".join(call["tool_id"] for call in event["calls"])
                        print(f"\n[tools] {names}", file=sys.stderr)
                    elif event["type"] == "tool_result":
                        print(f"[tool] {json.dumps(event['result']['output'], default=str)[:300]}", file=sys.stderr)
                    elif event["type"] == "error":
                        error = event["error"]
                        print(f"\n[{error['code']}] {error['message']}", file=sys.stderr)
                        if error.get("remediation"):
                            print(f"fix: {error['remediation']}", file=sys.stderr)
                        return 1
                    elif event["type"] == "done":
                        # Keep the metadata on stderr, right after the streamed answer.
                        print(file=sys.stderr)
                        for line in inference_lines(
                            model=event.get("model", ""),
                            runtime=event.get("runtime", ""),
                            engine_id=event.get("engine_id", ""),
                            attribution=event.get("attribution", ""),
                        ):
                            print(line, file=sys.stderr)
                return 0
            outcome = runtime.chat(
                message,
                session=session,
                engine_id=args.engine,
                task=args.task,
                use_tools=not args.no_tools,
                use_memory=not args.no_memory,
                sampling={"temperature": args.temperature, "max_tokens": args.max_tokens},
            )
        except AlphaAIError as exc:
            print(json.dumps({"ok": False, "error": exc.to_dict()}, indent=2), file=sys.stderr)
            return 1
        print(outcome.text)
        for result in outcome.tool_results:
            print(f"[tool {result.tool_id}] {json.dumps(result.output, default=str)[:300]}", file=sys.stderr)
        for line in inference_lines(
            model=outcome.model,
            runtime=outcome.runtime,
            engine_id=outcome.engine_id,
            attribution=outcome.attribution,
        ):
            print(line, file=sys.stderr)
        print(
            f"[{outcome.usage.total_tokens} tokens · {outcome.latency_ms:.0f} ms · "
            f"route: {outcome.routing.get('task', 'general')}]",
            file=sys.stderr,
        )
        if args.verbose:
            _dump(outcome.to_dict(), True)
        return 0

    if args.prompt:
        return one(" ".join(args.prompt))

    print(banner("interactive chat — /exit to quit, /clear to reset"))
    _print_attribution()
    usable = runtime.registry.usable()
    if usable:
        model = usable[0].model
        print(f"  {powered_by(model)} · Runtime: {usable[0].runtime} · Mode: Local")
    while True:
        try:
            message = input("\n› ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not message:
            continue
        if message in {"/exit", "/quit"}:
            return 0
        if message == "/clear":
            session = runtime.create_session(system_prompt=args.system, engine_id=args.engine)
            continue
        one(message)


def cmd_orchestrate(args: argparse.Namespace) -> int:
    runtime = _runtime(args)
    try:
        if args.plan:
            steps = json.loads(Path(args.plan).read_text(encoding="utf-8"))
            report = runtime.orchestrator.run_plan(steps, goal=args.goal or "")
        else:
            if not args.goal:
                print("Provide --goal or --plan.", file=sys.stderr)
                return 2
            report = runtime.orchestrator.run_goal(
                args.goal, max_steps=args.max_steps, engine_id=args.engine, use_tools=not args.no_tools
            )
    except AlphaAIError as exc:
        _dump({"ok": False, "error": exc.to_dict()}, True)
        return 1
    _dump(report.to_dict(), True)
    return 0 if report.ok else 1


def cmd_serve(args: argparse.Namespace) -> int:
    from .api import resolve_bind, resolve_mode, serve

    # Show the bind this process will really use (explicit flag, then $PORT,
    # then api.port) instead of the raw flag values.
    active, gateway_mode = resolve_mode(getattr(args, "config", None))
    host, port, _level = resolve_bind(
        args.host,
        args.port,
        fallback_host=active.api.host,
        fallback_port=active.api.port,
        log_level=args.log_level,
    )
    subtitle = f"serving on {host}:{port}"
    if gateway_mode:
        subtitle += f" · gateway → {active.api.inference_url}"
    print(banner(subtitle))
    _print_attribution()
    serve(active, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from . import training as training_pkg

    return training_pkg.cli(getattr(args, "action", "status"), args)


# ---------------------------------------------------------------------------
# argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alphaai",
        description=f"{DISPLAY_NAME} — {TAGLINE}",
    )
    parser.add_argument("--version", action="version", version=f"{DISPLAY_NAME} {__version__}")
    parser.add_argument("--config", help="Path to an AlphaAI config file (.toml/.json).", default=None)
    parser.add_argument("--project-root", help="AlphaAI project root (default: cwd).", default=None)

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="Print version and attribution.").set_defaults(func=cmd_version)
    sub.add_parser("attribution", help="Print the full AlphaAI/DeepSeek attribution.").set_defaults(
        func=cmd_attribution
    )

    doctor = sub.add_parser("doctor", help="Diagnose what works on this machine.")
    doctor.add_argument("--json", action="store_true", help="Only print the JSON report.")
    doctor.set_defaults(func=cmd_doctor)

    config = sub.add_parser("config", help="Inspect the AlphaAI configuration.")
    config.add_argument("action", choices=["show", "paths", "validate"], nargs="?", default="show")
    config.add_argument("--json", action="store_true")
    config.set_defaults(func=cmd_config)

    db = sub.add_parser("db", help="Inspect and migrate the AlphaAI PostgreSQL schema.")
    db.add_argument("action", choices=["status", "plan", "migrate"], nargs="?", default="status")
    db.add_argument(
        "--directory",
        help="Migration directory (default: paths.migrations_dir, i.e. supabase/migrations).",
    )
    db.add_argument("--json", action="store_true")
    db.set_defaults(func=cmd_db)

    models = sub.add_parser("models", help="List, check and install model engines.")
    models.add_argument("action", choices=["list", "check", "show", "install"], nargs="?", default="list")
    models.add_argument("model_id", nargs="?")
    models.add_argument("--json", action="store_true")
    models.add_argument(
        "--dry-run",
        action="store_true",
        help="install: only print the hardware/source/size safety report.",
    )
    models.add_argument(
        "--force",
        action="store_true",
        help="install: continue even when the hardware check says the model will not fit.",
    )
    models.add_argument(
        "--no-load-test",
        action="store_true",
        help="install: skip the post-download load test (not recommended).",
    )
    models.set_defaults(func=cmd_models)

    skills = sub.add_parser("skills", help="List and run AlphaAI skills.")
    skills.add_argument("action", choices=["list", "run"], nargs="?", default="list")
    skills.add_argument("skill_id", nargs="?")
    skills.add_argument("--input", help="JSON input, or @path/to/file.json")
    skills.add_argument("--json", action="store_true")
    skills.set_defaults(func=cmd_skills)

    tools = sub.add_parser("tools", help="List and run AlphaAI tools.")
    tools.add_argument("action", choices=["list", "run", "calls"], nargs="?", default="list")
    tools.add_argument("tool_id", nargs="?")
    tools.add_argument("--input", help="JSON input, or @path/to/file.json")
    tools.add_argument("--json", action="store_true")
    tools.set_defaults(func=cmd_tools)

    route = sub.add_parser("route", help="Explain which engine would handle a request.")
    route.add_argument("text")
    route.add_argument("--task")
    route.set_defaults(func=cmd_route)

    infer = sub.add_parser("infer", help="One-shot generation (real inference).")
    infer.add_argument("prompt")
    infer.add_argument("--engine")
    infer.add_argument("--task")
    infer.add_argument("--max-tokens", type=int, default=256)
    # None => use the configured default (sampling.temperature), which is tuned
    # for the local model actually installed.
    infer.add_argument("--temperature", type=float, default=None)
    infer.add_argument("--quiet", action="store_true")
    infer.set_defaults(func=cmd_infer)

    chat = sub.add_parser("chat", help="Chat with AlphaAI (real inference + tools).")
    chat.add_argument("prompt", nargs="*")
    chat.add_argument("--engine")
    chat.add_argument("--task")
    chat.add_argument("--system", default=None)
    chat.add_argument("--no-tools", action="store_true")
    chat.add_argument("--no-memory", action="store_true")
    chat.add_argument("--stream", action="store_true", default=True)
    chat.add_argument("--no-stream", dest="stream", action="store_false")
    chat.add_argument("--max-tokens", type=int, default=512)
    chat.add_argument("--temperature", type=float, default=None)
    chat.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full generation result, including which model produced it.",
    )
    chat.set_defaults(func=cmd_chat)

    orchestrate = sub.add_parser("orchestrate", help="Run an agent goal or a declared plan.")
    orchestrate.add_argument("--goal")
    orchestrate.add_argument("--plan", help="JSON file with declared steps.")
    orchestrate.add_argument("--engine")
    orchestrate.add_argument("--max-steps", type=int, default=None)
    orchestrate.add_argument("--no-tools", action="store_true")
    orchestrate.set_defaults(func=cmd_orchestrate)

    serve = sub.add_parser("serve", help="Run the AlphaAI HTTP API.")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument(
        "--log-level",
        default=None,
        help="uvicorn log level (defaults to $ALPHAI_LOG_LEVEL, then info).",
    )
    serve.set_defaults(func=cmd_serve)

    train = sub.add_parser("train", help="Training foundation commands.")
    train.add_argument(
        "action",
        choices=["validate", "prepare", "tokenize", "finetune", "evaluate", "status"],
        nargs="?",
        default="status",
    )
    train.add_argument("--dataset")
    train.add_argument("--tokenizer")
    train.add_argument("--experiment")
    train.add_argument("--config")
    train.add_argument("--json", action="store_true")
    train.set_defaults(func=cmd_train)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except AlphaAIError as exc:
        print(
            json.dumps({"ok": False, "error": exc.to_dict()}, indent=2, ensure_ascii=False),
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print()
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main", "NAME"]
