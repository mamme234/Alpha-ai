"""System AlphaAI skills: code analysis, tool calling, web research, agent workflows."""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Sequence

from ..errors import SkillExecutionError
from ..types import ToolCall
from .base import Skill, SkillContext

# ===========================================================================
# 1. code analysis
# ===========================================================================
_DANGEROUS_PATTERNS: tuple[tuple[str, str, str], ...] = (
    (r"\beval\s*\(", "high", "eval() executes arbitrary expressions"),
    (r"\bexec\s*\(", "high", "exec() executes arbitrary code"),
    (r"\b__import__\s*\(", "medium", "dynamic import"),
    (r"\bos\.system\s*\(", "high", "shell execution via os.system"),
    (r"\bsubprocess\.\w+\([^)]*shell\s*=\s*True", "high", "subprocess with shell=True"),
    (r"\bpickle\.loads?\s*\(", "high", "unsafe pickle deserialisation"),
    (r"\bmarshal\.loads?\s*\(", "high", "unsafe marshal deserialisation"),
    (r"\byaml\.load\s*\((?![^)]*Safe)", "medium", "yaml.load without SafeLoader"),
    (r"\binput\s*\(", "low", "interactive input() call"),
    (r"assert\s+", "low", "assert is stripped under -O; avoid for validation"),
    (r"except\s*:", "medium", "bare except hides errors"),
    (r"(?i)\b(api[_-]?key|secret|password|token)\b\s*[:=]\s*['\"][^'\"]{8,}['\"]", "high", "hard-coded credential"),
    (r"http://", "low", "plaintext HTTP endpoint"),
    (r"\brequests\.(get|post)\(", "low", "third-party HTTP dependency (use urllib in AlphaAI tools)"),
)


class CodeAnalysisSkill(Skill):
    skill_id = "skill.code_analysis"
    name = "Code analysis"
    description = (
        "Real static analysis of source code: syntax validation, size metrics, cyclomatic complexity per "
        "function, imports, dangerous-pattern and credential scanning, plus optional execution in the "
        "AlphaAI Python sandbox."
    )
    category = "code"
    # Static analysis needs no tools, so the skill stays available everywhere.
    # The optional 'execute' action calls python.sandbox and therefore fails with
    # an explicit permission error until tools.allow_code_execution is enabled.
    required_tools = ()
    input_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "maxLength": 400000},
            "path": {"type": "string"},
            "language": {"type": "string", "enum": ["auto", "python", "javascript", "typescript", "other"]},
            "actions": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["syntax", "metrics", "complexity", "imports", "security", "describe", "execute"],
                },
            },
            "tests": {"type": "string", "description": "Extra code appended before sandboxed execution.", "maxLength": 100000},
            "timeout_s": {"type": "number", "minimum": 0.5, "maximum": 60},
            "max_functions": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
        },
    }
    output_schema = {"type": "object", "properties": {"language": {"type": "string"}, "actions": {"type": "array"}}}
    timeout_s = 60.0
    tags = ("code", "static-analysis")

    def run(self, inputs: dict[str, Any], context: SkillContext) -> dict[str, Any]:
        code = inputs.get("code")
        source = "inline"
        if code is None:
            path = inputs.get("path")
            if not path:
                raise SkillExecutionError("Provide 'code' or a 'path' to analyse.")
            source = path
            code = context.tool_output("file.read", {"path": path, "limit": 20000}).get("content", "")
        code = str(code)
        language = inputs.get("language") or "auto"
        if language == "auto":
            language = _guess_language(source, code)
        actions = inputs.get("actions") or ["syntax", "metrics", "complexity", "imports", "security"]

        result: dict[str, Any] = {"source": source, "language": language, "actions": actions, "lines": len(code.splitlines()), "characters": len(code)}

        if language != "python":
            result["metrics"] = _generic_metrics(code)
            result["security"] = _scan_patterns(code)
            result["note"] = (
                f"Deep AST analysis is implemented for Python; '{language}' was analysed with real "
                "lexical metrics and pattern scanning only."
            )
            if "execute" in actions:
                result["execution"] = _execute(context, code, inputs, language)
            return result

        try:
            tree = ast.parse(code)
            result["syntax"] = {"valid": True, "errors": []}
        except SyntaxError as exc:
            result["syntax"] = {
                "valid": False,
                "errors": [
                    {
                        "message": exc.msg,
                        "line": exc.lineno,
                        "offset": exc.offset,
                        "text": (exc.text or "").strip(),
                    }
                ],
            }
            result["metrics"] = _generic_metrics(code)
            result["security"] = _scan_patterns(code)
            if "execute" in actions:
                result["execution"] = {"skipped": "syntax errors prevent execution"}
            return result

        if "metrics" in actions:
            result["metrics"] = _python_metrics(code, tree)
        if "complexity" in actions:
            result["complexity"] = _complexity_report(tree, int(inputs.get("max_functions", 50)))
        if "imports" in actions:
            result["imports"] = _imports(tree)
        if "security" in actions:
            result["security"] = _scan_patterns(code)
        if "describe" in actions:
            result["describe"] = _describe(tree, code)
        if "execute" in actions:
            result["execution"] = _execute(context, code, inputs, language)
        return result


def _guess_language(path: str, code: str) -> str:
    if path.endswith(".py") or re.search(r"^\s*(def |class |import |from \w+ import)", code, re.MULTILINE):
        return "python"
    if re.search(r"^\s*(function |const |let |=>|import .* from)", code, re.MULTILINE):
        return "javascript"
    if re.search(r"^\s*(interface |type \w+ =|: \w+\[\])", code, re.MULTILINE):
        return "typescript"
    return "other"


def _generic_metrics(code: str) -> dict[str, Any]:
    lines = code.splitlines()
    blank = sum(1 for line in lines if not line.strip())
    comment = sum(1 for line in lines if line.strip().startswith(("#", "//", "/*", "*")))
    return {
        "lines": len(lines),
        "code_lines": len(lines) - blank - comment,
        "blank_lines": blank,
        "comment_lines": comment,
        "comment_ratio": round(comment / max(len(lines), 1), 4),
        "max_line_length": max((len(line) for line in lines), default=0),
        "long_lines": sum(1 for line in lines if len(line) > 120),
    }


def _python_metrics(code: str, tree: ast.AST) -> dict[str, Any]:
    metrics = _generic_metrics(code)
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    metrics.update(
        {
            "functions": len(functions),
            "async_functions": len([node for node in functions if isinstance(node, ast.AsyncFunctionDef)]),
            "classes": len(classes),
            "methods": sum(len([item for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))]) for node in classes),
            "docstrings": sum(1 for node in functions + classes if ast.get_docstring(node)),
            "type_annotated_functions": sum(
                1 for node in functions if node.returns is not None or any(arg.annotation for arg in node.args.args)
            ),
            "max_nesting_depth": _max_depth(tree),
            "imports": len([node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]),
            "try_blocks": len([node for node in ast.walk(tree) if isinstance(node, ast.Try)]),
        }
    )
    return metrics


def _max_depth(tree: ast.AST, depth: int = 0) -> int:
    nesting_nodes = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try, ast.Match)
    deepest = depth
    for node in ast.iter_child_nodes(tree):
        child_depth = depth + 1 if isinstance(node, nesting_nodes) else depth
        deepest = max(deepest, _max_depth(node, child_depth))
    return deepest


def _complexity(tree: ast.AST) -> int:
    decisions = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.BoolOp, ast.IfExp, ast.comprehension, ast.Match)
    return 1 + sum(1 for node in ast.walk(tree) if isinstance(node, decisions))


def _complexity_report(tree: ast.AST, limit: int) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_tree = ast.Module(body=node.body, type_ignores=[])
            complexity = _complexity(function_tree)
            body_lines = (node.end_lineno or node.lineno) - node.lineno + 1
            items.append(
                {
                    "name": node.name,
                    "line": node.lineno,
                    "lines": body_lines,
                    "cyclomatic_complexity": complexity,
                    "arguments": len(node.args.args) + len(node.args.kwonlyargs),
                    "rating": "simple" if complexity <= 5 else ("moderate" if complexity <= 10 else "complex"),
                }
            )
    items.sort(key=lambda item: -item["cyclomatic_complexity"])
    total = sum(item["cyclomatic_complexity"] for item in items)
    return {
        "functions": items[:limit],
        "function_count": len(items),
        "average_complexity": round(total / len(items), 3) if items else 0,
        "highest_complexity": items[0]["cyclomatic_complexity"] if items else 0,
        "hotspots": [item["name"] for item in items if item["cyclomatic_complexity"] > 10][:20],
    }


def _imports(tree: ast.AST) -> dict[str, Any]:
    modules: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.setdefault(alias.name, []).append("import")
        elif isinstance(node, ast.ImportFrom):
            modules.setdefault(node.module or "", []).append(f"from (level {node.level})")
    return {"count": len(modules), "modules": sorted(modules)}


def _scan_patterns(code: str) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    for pattern, severity, message in _DANGEROUS_PATTERNS:
        for match in re.finditer(pattern, code):
            line = code[: match.start()].count("\n") + 1
            findings.append(
                {
                    "rule": pattern,
                    "severity": severity,
                    "message": message,
                    "line": line,
                    "excerpt": code.splitlines()[line - 1][:160] if line - 1 < len(code.splitlines()) else "",
                }
            )
    findings.sort(key=lambda item: {"high": 0, "medium": 1, "low": 2}[item["severity"]])
    return {
        "findings": findings[:50],
        "count": len(findings),
        "high": len([item for item in findings if item["severity"] == "high"]),
        "medium": len([item for item in findings if item["severity"] == "medium"]),
        "low": len([item for item in findings if item["severity"] == "low"]),
    }


def _describe(tree: ast.AST, code: str) -> dict[str, Any]:
    module_doc = ast.get_docstring(tree)
    return {
        "module_docstring": (module_doc[:400] if module_doc else None),
        "top_level_names": sorted(
            node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ),
        "has_main_guard": "__main__" in code,
    }


def _execute(context: SkillContext, code: str, inputs: dict[str, Any], language: str) -> dict[str, Any]:
    if language != "python":
        return {"skipped": f"Sandboxed execution is implemented for Python; '{language}' was not executed."}
    script = code if not inputs.get("tests") else f"{code}\n\n{inputs['tests']}"
    try:
        output = context.tool_output(
            "python.sandbox",
            {"code": script, "timeout_s": float(inputs.get("timeout_s", 10))},
            )
    except Exception as exc:  # noqa: BLE001 - report the real reason (usually permissions)
        return {"executed": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "executed": True,
        "exit_code": output["exit_code"],
        "stdout": output["stdout"],
        "stderr": output["stderr"],
        "duration_ms": output["duration_ms"],
    }


# ===========================================================================
# 2. tool calling
# ===========================================================================
class ToolCallingSkill(Skill):
    skill_id = "skill.tool_calling"
    name = "Tool calling"
    description = (
        "Drive AlphaAI's tool layer: parse structured tool calls out of model text (JSON blocks, JSON "
        "arrays, name(args) syntax) and execute them through the permission-checked executor, with "
        "$<index>.output.<path> dependency substitution between calls."
    )
    category = "orchestration"
    required_tools = ("calculator.evaluate",)
    input_schema = {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["execute", "parse"]},
            "calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool_id": {"type": "string"},
                        "arguments": {"type": "object"},
                        "call_id": {"type": "string"},
                    },
                    "required": ["tool_id"],
                },
            },
            "text": {"type": "string", "maxLength": 200000, "description": "Model output to parse in 'parse' mode."},
            "fail_fast": {"type": "boolean", "default": False},
            "run_id": {"type": "string"},
        },
        "required": ["mode"],
    }
    output_schema = {"type": "object", "properties": {"mode": {"type": "string"}, "results": {"type": "array"}}}
    timeout_s = 60.0
    tags = ("orchestration", "tools")

    def run(self, inputs: dict[str, Any], context: SkillContext) -> dict[str, Any]:
        mode = inputs["mode"]
        if mode == "parse":
            return self._parse(inputs, context)
        if inputs.get("calls"):
            return self._execute(inputs, context)
        if inputs.get("text"):
            parsed = self._parse(inputs, context)
            return self._execute({**inputs, "calls": parsed["calls"]}, context)
        raise SkillExecutionError("Provide 'calls' (execute) or 'text' (parse).")

    def _parse(self, inputs: dict[str, Any], context: SkillContext) -> dict[str, Any]:
        text = inputs.get("text") or ""
        candidates = _extract_tool_payloads(text)
        calls: list[dict[str, Any]] = []
        invalid: list[dict[str, Any]] = []
        for payload in candidates:
            call = ToolCall.from_dict(payload)
            if not call.tool_id:
                invalid.append({"reason": "missing tool id", "payload": payload})
                continue
            entry = {"tool_id": call.tool_id, "arguments": call.arguments, "call_id": call.call_id or f"parsed_{len(calls)}"}
            if context.tools is not None and not context.tools.registry.has(call.tool_id):
                entry["known"] = False
                invalid.append({"reason": f"unknown tool '{call.tool_id}'", "payload": payload})
            else:
                entry["known"] = True
                calls.append(entry)
        return {
            "mode": "parse",
            "calls": calls,
            "count": len(calls),
            "invalid": invalid,
            "parser": "alphaai.tool_call_parser",
        }

    def _execute(self, inputs: dict[str, Any], context: SkillContext) -> dict[str, Any]:
        calls = inputs["calls"]
        fail_fast = bool(inputs.get("fail_fast"))
        run_id = inputs.get("run_id") or context.run_id
        results: list[dict[str, Any]] = []
        for index, raw in enumerate(calls):
            call = ToolCall.from_dict(raw)
            arguments = _resolve_references(call.arguments, results)
            result = context.tools.run(
                call.tool_id,
                arguments,
                call_id=call.call_id or f"call_{index}",
                run_id=run_id,
                context_metadata={"skill_manager": context.metadata.get("skill_manager")},
            )
            results.append(result.to_dict())
            if fail_fast and not result.ok:
                break
        return {
            "mode": "execute",
            "count": len(results),
            "succeeded": len([item for item in results if item["ok"]]),
            "failed": len([item for item in results if not item["ok"]]),
            "results": results,
        }


def _extract_tool_payloads(text: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    seen_spans: set[tuple[int, int]] = set()

    for match in re.finditer(r"```(?:json|tool_call|tools)?\s*(.+?)```", text, flags=re.DOTALL):
        block = match.group(1).strip()
        seen_spans.add(match.span(1))
        payloads.extend(_json_payloads(block))

    for match in re.finditer(r"(?<![`\w])([a-zA-Z][\w]*(?:\.[\w]+)+)\s*\(\s*(\{.*?\})\s*\)", text, flags=re.DOTALL):
        if any(start <= match.start() < end for start, end in seen_spans):
            continue
        try:
            arguments = json.loads(match.group(2))
        except json.JSONDecodeError:
            continue
        payloads.append({"tool_id": match.group(1), "arguments": arguments})

    if not payloads:
        payloads.extend(_json_payloads(text.strip()))
    return payloads


def _json_payloads(text: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        simplified = re.sub(r"^\s*(?:tool_calls|tools|calls)\s*[:=]\s*", "", text).strip()
        try:
            payload = json.loads(simplified)
        except json.JSONDecodeError:
            return []
    return _coerce_payloads(payload)


def _coerce_payloads(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if "tool_calls" in payload and isinstance(payload["tool_calls"], list):
            return [item for item in payload["tool_calls"] if isinstance(item, dict)]
        if any(key in payload for key in ("tool_id", "name", "tool", "function")):
            return [payload]
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _resolve_references(arguments: Any, results: Sequence[dict[str, Any]]) -> Any:
    if isinstance(arguments, dict):
        return {key: _resolve_references(value, results) for key, value in arguments.items()}
    if isinstance(arguments, list):
        return [_resolve_references(item, results) for item in arguments]
    if isinstance(arguments, str) and arguments.startswith("$"):
        return _dereference(arguments, results)
    return arguments


def _dereference(reference: str, results: Sequence[dict[str, Any]]) -> Any:
    body = reference[1:]
    index_text, _, path = body.partition(".")
    try:
        index = int(index_text)
    except ValueError:
        return reference
    if index < 0 or index >= len(results):
        return reference
    value: Any = results[index]
    path = path.removeprefix("output").lstrip(".")
    for token in [part for part in re.split(r"\.", path) if part]:
        if isinstance(value, dict) and token in value:
            value = value[token]
        elif isinstance(value, list) and token.isdigit() and int(token) < len(value):
            value = value[int(token)]
        else:
            return None
    return value


# ===========================================================================
# 3. web research
# ===========================================================================
class WebResearchSkill(Skill):
    skill_id = "skill.web_research"
    name = "Web research"
    description = (
        "Perform real web research: search a live engine (DuckDuckGo HTML, no API key), fetch the top "
        "sources over HTTP, extract their text and rank them by BM25 relevance to the query. Returns "
        "citations with the exact excerpts that matched."
    )
    category = "research"
    required_tools = ("web.search", "http.fetch")
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "maxLength": 500},
            "max_sources": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
            "max_chars_per_source": {"type": "integer", "minimum": 500, "maximum": 20000, "default": 6000},
            "include_domains": {"type": "array", "items": {"type": "string"}},
            "exclude_domains": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["query"],
    }
    output_schema = {"type": "object", "properties": {"query": {"type": "string"}, "sources": {"type": "array"}}}
    timeout_s = 120.0
    tags = ("research", "network", "engine-optional")

    def run(self, inputs: dict[str, Any], context: SkillContext) -> dict[str, Any]:
        query = inputs["query"]
        max_sources = int(inputs.get("max_sources", 5))
        search = context.tool_output("web.search", {"query": query, "limit": max(max_sources * 2, 8)})
        found = list(search.get("results") or [])
        include = [domain.lower() for domain in (inputs.get("include_domains") or [])]
        exclude = [domain.lower() for domain in (inputs.get("exclude_domains") or [])]

        def allowed(url: str) -> bool:
            host = url.split("/")[2].lower() if url.count("/") >= 2 else ""
            if include and not any(host.endswith(domain) for domain in include):
                return False
            return not any(host.endswith(domain) for domain in exclude)

        candidates = [item for item in found if allowed(item.get("url", ""))][: max(max_sources * 2, max_sources)]
        if not candidates:
            raise SkillExecutionError(
                "The web search returned no usable results for this query (or all were filtered out).",
                remediation="Broaden the query, relax include_domains/exclude_domains, or retry later.",
            )

        limit_chars = int(inputs.get("max_chars_per_source", 6000))
        documents: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for item in candidates:
            entry: dict[str, Any] = {"title": item.get("title", ""), "url": item.get("url", ""), "snippet": item.get("snippet", "")}
            try:
                fetched = context.tool_output("http.fetch", {"url": entry["url"], "max_bytes": min(limit_chars * 2, 800000)})
                body = fetch_text = fetched.get("text") or ""
                text = re.sub(r"\s+", " ", _strip_markup(fetch_text))[:limit_chars]
                entry.update({"status": fetched.get("status"), "content_type": fetched.get("content_type"), "text": text, "fetched": bool(text)})
                documents.append(entry)
            except Exception as exc:  # noqa: BLE001 - keep going, report honestly per source
                entry["fetched"] = False
                entry["error"] = f"{type(exc).__name__}: {exc}"
                errors.append({"url": entry["url"], "error": entry["error"]})
                documents.append(entry)

        ranked = _rank_documents(query, documents)
        return {
            "query": query,
            "provider": search.get("provider", "duckduckgo-html"),
            "candidates": len(candidates),
            "fetched": len([doc for doc in documents if doc.get("fetched")]),
            "sources": ranked[:max_sources],
            "errors": errors,
            "ranking": "bm25_lite",
            "citations": [
                {"index": index + 1, "title": doc.get("title"), "url": doc.get("url")}
                for index, doc in enumerate(ranked[:max_sources])
            ],
        }


def _strip_markup(html: str) -> str:
    import html as html_module

    without_scripts = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    return html_module.unescape(re.sub(r"<[^>]+>", " ", without_scripts))


def _rank_documents(query: str, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank documents with a real BM25-lite score over the fetched text."""

    terms = [term.lower() for term in re.findall(r"[^\W_]{3,}", query)]
    tokenised = [re.findall(r"[^\W_]{3,}", (doc.get("text") or "").lower()) for doc in documents]
    total_docs = max(len(documents), 1)
    average_length = sum(len(tokens) for tokens in tokenised) / total_docs or 1.0
    document_frequency: dict[str, int] = {}
    for tokens in tokenised:
        for term in set(tokens):
            document_frequency[term] = document_frequency.get(term, 0) + 1

    k1, b = 1.5, 0.75
    scored: list[dict[str, Any]] = []
    for doc, tokens in zip(documents, tokenised):
        score = 0.0
        for term in terms:
            frequency = tokens.count(term)
            if frequency == 0:
                continue
            idf = __import__("math").log(1 + (total_docs - document_frequency.get(term, 0) + 0.5) / (document_frequency.get(term, 0) + 0.5))
            denominator = frequency + k1 * (1 - b + b * len(tokens) / average_length)
            score += idf * frequency * (k1 + 1) / max(denominator, 1e-9)
        entry = dict(doc)
        entry["relevance"] = round(score, 6)
        entry["matched_terms"] = sorted({term for term in terms if term in entry.get("text", "").lower()})
        entry["excerpt"] = _excerpt(entry.get("text") or entry.get("snippet") or "", terms)
        scored.append(entry)
    scored.sort(key=lambda item: -item["relevance"])
    return scored


def _excerpt(text: str, terms: Sequence[str], window: int = 320) -> str:
    lowered = text.lower()
    for term in terms:
        position = lowered.find(term)
        if position >= 0:
            start = max(0, position - window // 3)
            return text[start : start + window].strip()
    return text[:window].strip()


# ===========================================================================
# 4. agent workflow execution
# ===========================================================================
class AgentWorkflowSkill(Skill):
    skill_id = "skill.agent_workflow"
    name = "Agent workflow"
    description = (
        "Execute a multi-step AlphaAI workflow: a real dependency graph of tool and skill steps with "
        "topological ordering, cycle detection, $<step>.output.<path> data passing, per-step errors and "
        "optional steps."
    )
    category = "orchestration"
    input_schema = {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "kind": {"type": "string", "enum": ["tool", "skill"]},
                        "target": {"type": "string"},
                        "input": {"type": "object"},
                        "depends_on": {"type": "array", "items": {"type": "string"}},
                        "optional": {"type": "boolean"},
                        "timeout_s": {"type": "number", "minimum": 0.5, "maximum": 600},
                    },
                    "required": ["id", "kind", "target"],
                },
            },
            "fail_fast": {"type": "boolean", "default": True},
            "max_steps": {"type": "integer", "minimum": 1, "maximum": 64, "default": 24},
        },
        "required": ["steps"],
    }
    output_schema = {"type": "object", "properties": {"order": {"type": "array"}, "completed": {"type": "integer"}}}
    timeout_s = 300.0
    tags = ("orchestration", "agent")

    def run(self, inputs: dict[str, Any], context: SkillContext) -> dict[str, Any]:
        steps = inputs["steps"]
        max_steps = int(inputs.get("max_steps", 24))
        if len(steps) > max_steps:
            raise SkillExecutionError(f"Workflow has {len(steps)} steps; the limit is {max_steps}.")
        fail_fast = inputs.get("fail_fast", True)

        by_id: dict[str, dict[str, Any]] = {}
        for step in steps:
            step_id = step["id"]
            if step_id in by_id:
                raise SkillExecutionError(f"Duplicate step id '{step_id}'.")
            by_id[step_id] = step
        for step in steps:
            for dependency in step.get("depends_on") or []:
                if dependency not in by_id:
                    raise SkillExecutionError(f"Step '{step['id']}' depends on unknown step '{dependency}'.")

        order = _topological_order(by_id)
        results: dict[str, dict[str, Any]] = {}
        statuses: dict[str, str] = {}
        executed: list[str] = []

        for step_id in order:
            step = by_id[step_id]
            dependencies = step.get("depends_on") or []
            failed_dependencies = [dep for dep in dependencies if statuses.get(dep) != "completed"]
            if failed_dependencies:
                statuses[step_id] = "skipped"
                results[step_id] = {
                    "id": step_id,
                    "status": "skipped",
                    "reason": f"dependency failed or was skipped: {', '.join(failed_dependencies)}",
                }
                continue

            resolved_input = _resolve_step_input(step.get("input") or {}, results)
            started = __import__("time").perf_counter()
            try:
                output = self._run_step(step, resolved_input, context)
                duration = (__import__("time").perf_counter() - started) * 1000
                statuses[step_id] = "completed"
                executed.append(step_id)
                results[step_id] = {
                    "id": step_id,
                    "kind": step["kind"],
                    "target": step["target"],
                    "status": "completed",
                    "output": output,
                    "duration_ms": round(duration, 3),
                    "depends_on": dependencies,
                }
            except Exception as exc:  # noqa: BLE001 - per-step failure reporting
                duration = (__import__("time").perf_counter() - started) * 1000
                message = f"{type(exc).__name__}: {exc}"
                if step.get("optional"):
                    statuses[step_id] = "completed"
                    results[step_id] = {
                        "id": step_id,
                        "kind": step["kind"],
                        "target": step["target"],
                        "status": "optional_failed",
                        "error": message,
                        "duration_ms": round(duration, 3),
                        "depends_on": dependencies,
                    }
                    continue
                statuses[step_id] = "failed"
                results[step_id] = {
                    "id": step_id,
                    "kind": step["kind"],
                    "target": step["target"],
                    "status": "failed",
                    "error": message,
                    "duration_ms": round(duration, 3),
                    "depends_on": dependencies,
                }
                if fail_fast:
                    break

        completed = len([key for key, value in statuses.items() if value == "completed"])
        failed = len([key for key, value in statuses.items() if value == "failed"])
        return {
            "order": order,
            "executed": executed,
            "completed": completed,
            "failed": failed,
            "skipped": len(order) - completed - failed,
            "success": failed == 0,
            "steps": [results[step_id] for step_id in order if step_id in results],
        }

    def _run_step(self, step: dict[str, Any], resolved_input: dict[str, Any], context: SkillContext) -> Any:
        timeouts = step.get("timeout_s")
        if step["kind"] == "tool":
            result = context.tools.execute(
                step["target"],
                resolved_input,
                run_id=context.run_id,
                context_metadata={"skill_manager": context.metadata.get("skill_manager")},
                timeout_s=float(timeouts) if timeouts else None,
            )
            if not result.ok:
                raise SkillExecutionError(
                    f"Tool step '{step['id']}' failed: {(result.error or {}).get('message', 'unknown error')}",
                    details={"tool_id": step["target"], "tool_error": result.error},
                )
            return result.output
        manager = context.metadata.get("skill_manager")
        if manager is None:
            raise SkillExecutionError(
                "Skill steps require the AlphaAI skill manager in the run context.",
                remediation="Run agent workflows through the AlphaAI runtime or API.",
            )
        output, _ = manager._execute(step["target"], resolved_input, context=context, run_id=context.run_id)
        return output


def _topological_order(steps: dict[str, dict[str, Any]]) -> list[str]:
    incoming: dict[str, int] = {step_id: 0 for step_id in steps}
    for step in steps.values():
        for dependency in step.get("depends_on") or []:
            incoming[step["id"]] += 1
    ready = sorted([step_id for step_id, count in incoming.items() if count == 0])
    order: list[str] = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        for step in steps.values():
            if current in (step.get("depends_on") or []):
                incoming[step["id"]] -= 1
                if incoming[step["id"]] == 0:
                    ready.append(step["id"])
        ready.sort()
    if len(order) != len(steps):
        cyclic = sorted(set(steps) - set(order))
        raise SkillExecutionError(f"Workflow has a dependency cycle involving: {', '.join(cyclic)}.")
    return order


def _resolve_step_input(value: Any, results: dict[str, dict[str, Any]]) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_step_input(item, results) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_step_input(item, results) for item in value]
    if isinstance(value, str) and value.startswith("$"):
        step_id, _, path = value[1:].partition(".")
        if step_id not in results:
            return value
        current: Any = results[step_id]
        for token in [part for part in path.split(".") if part]:
            if isinstance(current, dict) and token in current:
                current = current[token]
            elif isinstance(current, list) and token.isdigit() and int(token) < len(current):
                current = current[int(token)]
            else:
                return None
        return current
    return value


def build_skills() -> list[Skill]:
    return [CodeAnalysisSkill(), ToolCallingSkill(), WebResearchSkill(), AgentWorkflowSkill()]
