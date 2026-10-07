"""AlphaAI built-in tools.

Every tool performs a real operation with a typed input schema, declared
permission requirements and structured output. Nothing here fabricates results:
when an operation cannot be performed, the handler raises and the executor turns
that into a structured tool error.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import ipaddress
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from ...branding import ATTRIBUTION_SHORT, DISPLAY_NAME, VERSION
from ..calc import CalculationError, evaluate
from ..errors import ToolExecutionError
from .permissions import resolve_sandbox_path
from .registry import Tool, ToolContext

# ---------------------------------------------------------------------------
# calculator
# ---------------------------------------------------------------------------
CALCULATOR = Tool(
    tool_id="calculator.evaluate",
    name="Calculator",
    description=(
        "Evaluate an arithmetic/math expression exactly. Supports + - * / // % **, comparisons, "
        "constants (pi, e, tau) and math functions such as sqrt, log, exp, sin, factorial, comb, mean."
    ),
    category="math",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Expression to evaluate, e.g. '(3.5 ** 2 + 41) / 7'",
                "maxLength": 1000,
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
    output_schema={
        "type": "object",
        "properties": {
            "expression": {"type": "string"},
            "value": {},
            "text": {"type": "string"},
            "approximate": {"type": "number"},
        },
    },
    timeout_s=2.0,
    tags=("math", "deterministic"),
)


def _calculator(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    try:
        result = evaluate(arguments["expression"])
    except CalculationError as exc:
        raise ToolExecutionError(f"Calculator could not evaluate the expression: {exc}") from exc
    payload = result.to_dict()
    # Keep the payload lean: the function list is available through the tool
    # schema, so it is not repeated on every call.
    payload.pop("functions", None)
    return payload


# ---------------------------------------------------------------------------
# text tools
# ---------------------------------------------------------------------------
_TEXT_OPERATIONS = (
    "upper",
    "lower",
    "title",
    "strip",
    "collapse_whitespace",
    "slugify",
    "snake_case",
    "kebab_case",
    "camel_case",
    "reverse",
    "count",
    "lines",
    "words",
    "unique_words",
    "frequency",
)

TEXT_TRANSFORM = Tool(
    tool_id="text.transform",
    name="Text transform",
    description=(
        "Deterministic text operations: case conversion, whitespace normalisation, slug/case "
        "conversion, reversal, and real counting/statistics (characters, words, lines, word frequency)."
    ),
    category="text",
    parameters={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Input text.", "maxLength": 400000},
            "operation": {"type": "string", "enum": list(_TEXT_OPERATIONS)},
            "limit": {"type": "integer", "minimum": 1, "maximum": 200, "description": "Max items for frequency results."},
        },
        "required": ["text", "operation"],
        "additionalProperties": False,
    },
    output_schema={"type": "object", "properties": {"operation": {"type": "string"}, "result": {}}},
    timeout_s=5.0,
    tags=("text", "deterministic"),
)


def _text_transform(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    text = arguments["text"]
    operation = arguments["operation"]
    limit = int(arguments.get("limit", 25))
    words = re.findall(r"[^\W_]+(?:['’-][^\W_]+)*", text, flags=re.UNICODE)

    if operation == "upper":
        result: Any = text.upper()
    elif operation == "lower":
        result = text.lower()
    elif operation == "title":
        result = text.title()
    elif operation == "strip":
        result = text.strip()
    elif operation == "collapse_whitespace":
        result = re.sub(r"\s+", " ", text).strip()
    elif operation == "slugify":
        ascii_text = (
            text.encode("ascii", "ignore").decode("ascii") if not text.isascii() else text
        )
        result = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    elif operation == "snake_case":
        result = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    elif operation == "kebab_case":
        result = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    elif operation == "camel_case":
        parts = [part for part in re.split(r"[^A-Za-z0-9]+", text) if part]
        result = parts[0].lower() + "".join(part.capitalize() for part in parts[1:]) if parts else ""
    elif operation == "reverse":
        result = text[::-1]
    elif operation == "count":
        result = {
            "characters": len(text),
            "characters_no_spaces": len(re.sub(r"\s", "", text)),
            "words": len(words),
            "lines": text.count("\n") + (1 if text and not text.endswith("\n") else 0),
            "unique_words": len({word.lower() for word in words}),
        }
    elif operation == "lines":
        result = text.splitlines()
    elif operation == "words":
        result = words
    elif operation == "unique_words":
        seen: list[str] = []
        for word in words:
            lowered = word.lower()
            if lowered not in seen:
                seen.append(lowered)
        result = seen
    elif operation == "frequency":
        counts: dict[str, int] = {}
        for word in words:
            key = word.lower()
            counts[key] = counts.get(key, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
        result = [{"word": word, "count": count} for word, count in ranked]
    else:  # pragma: no cover - schema enum blocks this
        raise ToolExecutionError(f"Unsupported text operation '{operation}'.")

    return {"operation": operation, "result": result, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


TEXT_REGEX = Tool(
    tool_id="text.regex",
    name="Regex extract",
    description="Run a real regular-expression search over text and return all matches with groups and positions.",
    category="text",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Python regular expression.", "maxLength": 500},
            "text": {"type": "string", "maxLength": 400000},
            "ignore_case": {"type": "boolean"},
            "max_matches": {"type": "integer", "minimum": 1, "maximum": 1000},
        },
        "required": ["pattern", "text"],
        "additionalProperties": False,
    },
    timeout_s=5.0,
    tags=("text", "deterministic"),
)


def _text_regex(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    flags = re.MULTILINE
    if arguments.get("ignore_case"):
        flags |= re.IGNORECASE
    try:
        compiled = re.compile(arguments["pattern"], flags)
    except re.error as exc:
        raise ToolExecutionError(f"Invalid regular expression: {exc}") from exc
    limit = int(arguments.get("max_matches", 100))
    matches = []
    for match in compiled.finditer(arguments["text"]):
        if len(matches) >= limit:
            break
        matches.append(
            {
                "match": match.group(0),
                "groups": list(match.groups()),
                "named_groups": match.groupdict(),
                "start": match.start(),
                "end": match.end(),
            }
        )
    return {"pattern": arguments["pattern"], "count": len(matches), "matches": matches}


# ---------------------------------------------------------------------------
# structured data
# ---------------------------------------------------------------------------
JSON_PATH = Tool(
    tool_id="json.path",
    name="JSON path query",
    description=(
        "Query a JSON document with a real path expression such as 'items[0].name' or "
        "'results[*].score' (use [*] to expand every element of an array)."
    ),
    category="data",
    parameters={
        "type": "object",
        "properties": {
            "data": {"type": ["string", "object", "array"], "description": "JSON text or an object."},
            "path": {"type": "string", "maxLength": 500},
            "limit": {"type": "integer", "minimum": 1, "maximum": 5000},
        },
        "required": ["data", "path"],
        "additionalProperties": False,
    },
    timeout_s=5.0,
    tags=("data", "deterministic"),
)


def _json_path(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    data = arguments["data"]
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(f"'data' is not valid JSON: {exc}") from exc
    limit = int(arguments.get("limit", 1000))
    tokens = _parse_json_path(arguments["path"])
    current: list[Any] = [data]
    for token in tokens:
        next_values: list[Any] = []
        for value in current:
            if token == "*":
                if isinstance(value, dict):
                    next_values.extend(value.values())
                elif isinstance(value, list):
                    next_values.extend(value)
            elif isinstance(token, int):
                if isinstance(value, list) and -len(value) <= token < len(value):
                    next_values.append(value[token])
            elif isinstance(value, dict) and token in value:
                next_values.append(value[token])
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and token in item:
                        next_values.append(item[token])
        current = next_values
        if len(current) > limit:
            current = current[:limit]
    single = current[0] if len(current) == 1 else None
    return {
        "path": arguments["path"],
        "found": bool(current),
        "count": len(current),
        "value": single if len(current) == 1 else None,
        "values": current if len(current) != 1 else [single],
    }


def _parse_json_path(path: str) -> list[Any]:
    tokens: list[Any] = []
    for part in re.finditer(r"[^.\[\]]+|\[(?:\*|-?\d+|'[^']*'|\"[^\"]*\")\]", path):
        raw = part.group(0)
        if raw.startswith("["):
            inner = raw[1:-1]
            if inner == "*":
                tokens.append("*")
            elif inner.strip("'\"") != inner and not inner.strip("'\"").isdigit():
                tokens.append(inner.strip("'\""))
            else:
                try:
                    tokens.append(int(inner))
                except ValueError:
                    tokens.append(inner.strip("'\""))
        elif raw != "$":
            tokens.append(raw)
    return tokens


# ---------------------------------------------------------------------------
# filesystem (sandbox-rooted)
# ---------------------------------------------------------------------------
FILE_READ = Tool(
    tool_id="file.read",
    name="Read file",
    description="Read a UTF-8 text file (or base64 for binary) from inside the AlphaAI sandbox root.",
    category="filesystem",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the sandbox root."},
            "offset": {"type": "integer", "minimum": 0, "description": "1-based line offset."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20000, "description": "Max lines."},
            "encoding": {"type": "string", "maxLength": 32},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    timeout_s=5.0,
    tags=("filesystem", "read"),
)


def _file_read(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    path = resolve_sandbox_path(context.sandbox_root, arguments["path"])
    if not path.exists():
        raise ToolExecutionError(f"File not found: {path.name}")
    if path.is_dir():
        raise ToolExecutionError(f"'{path.name}' is a directory; use file.list instead.")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    encoding = arguments.get("encoding", "utf-8")
    try:
        text = raw.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        limit = context.max_output_bytes
        return {
            "path": str(path),
            "size_bytes": len(raw),
            "sha256": digest,
            "encoding": "binary",
            "base64": base64.b64encode(raw[:limit]).decode("ascii"),
            "truncated": len(raw) > limit,
        }
    lines = text.splitlines()
    offset = max(int(arguments.get("offset", 0) or 0), 0)
    limit = int(arguments.get("limit", 2000) or 2000)
    selected = lines[offset : offset + limit]
    return {
        "path": str(path),
        "size_bytes": len(raw),
        "sha256": digest,
        "encoding": encoding,
        "total_lines": len(lines),
        "offset": offset,
        "returned_lines": len(selected),
        "truncated": offset + limit < len(lines),
        "content": "\n".join(selected),
    }


FILE_LIST = Tool(
    tool_id="file.list",
    name="List directory",
    description="List files and directories inside the AlphaAI sandbox root with real sizes and types.",
    category="filesystem",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory relative to the sandbox root.", "maxLength": 1000},
            "pattern": {"type": "string", "description": "Optional glob pattern, e.g. '*.json'.", "maxLength": 100},
            "recursive": {"type": "boolean"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 2000},
        },
        "additionalProperties": False,
    },
    timeout_s=10.0,
    tags=("filesystem", "read"),
)


def _file_list(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    root = context.sandbox_root
    target = resolve_sandbox_path(root, arguments.get("path") or ".")
    if not target.exists():
        raise ToolExecutionError(f"Directory not found: {target.name}")
    if not target.is_dir():
        raise ToolExecutionError(f"'{target.name}' is not a directory.")
    limit = int(arguments.get("limit", 500))
    pattern = arguments.get("pattern")
    recursive = bool(arguments.get("recursive"))
    iterator: Iterable[Path] = target.rglob(pattern or "*") if recursive else target.glob(pattern or "*")
    entries: list[dict[str, Any]] = []
    for entry in sorted(iterator, key=lambda item: str(item)):
        if len(entries) >= limit:
            break
        try:
            stat = entry.stat()
        except OSError:
            continue
        try:
            relative = entry.relative_to(root).as_posix() if root else entry.name
        except ValueError:  # pragma: no cover - target outside the sandbox root
            relative = entry.name
        entries.append(
            {
                "path": relative,
                "name": entry.name,
                "type": "directory" if entry.is_dir() else "file",
                "size_bytes": stat.st_size if entry.is_file() else None,
                "modified": stat.st_mtime,
            }
        )
    return {"root": str(target), "count": len(entries), "entries": entries}


FILE_WRITE = Tool(
    tool_id="file.write",
    name="Write file",
    description="Write or append text inside the AlphaAI sandbox root. Requires the write permission.",
    category="filesystem",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string", "maxLength": 1000000},
            "mode": {"type": "string", "enum": ["overwrite", "append"]},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
    requires_write=True,
    timeout_s=10.0,
    tags=("filesystem", "write"),
)


def _file_write(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    if not context.allow_write:
        raise PermissionError("File writes are not permitted for this call.")
    path = resolve_sandbox_path(context.sandbox_root, arguments["path"])
    mode = arguments.get("mode", "overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = arguments["content"].encode("utf-8")
    if mode == "append":
        with open(path, "ab") as handle:
            handle.write(encoded)
    else:
        with open(path, "wb") as handle:
            handle.write(encoded)
    return {
        "path": str(path),
        "bytes_written": len(encoded),
        "mode": mode,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


FILE_HASH = Tool(
    tool_id="file.hash",
    name="Hash file",
    description="Compute a real cryptographic hash (sha256, sha1, md5, blake2b) of a file inside the sandbox.",
    category="filesystem",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "algorithm": {"type": "string", "enum": ["sha256", "sha1", "md5", "blake2b"]},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
    timeout_s=20.0,
    tags=("filesystem", "read", "integrity"),
)


def _file_hash(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    path = resolve_sandbox_path(context.sandbox_root, arguments["path"])
    if not path.is_file():
        raise ToolExecutionError(f"Not a file: {path.name}")
    algorithm = arguments.get("algorithm", "sha256")
    if algorithm not in {"sha256", "sha1", "md5", "blake2b"}:
        raise ToolExecutionError(f"Unsupported hash algorithm '{algorithm}'.")
    digest = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "algorithm": algorithm, "digest": digest.hexdigest(), "size_bytes": path.stat().st_size}


FILE_STAT = Tool(
    tool_id="file.stat",
    name="File metadata",
    description="Return real filesystem metadata for a path inside the sandbox root.",
    category="filesystem",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    },
    timeout_s=5.0,
    tags=("filesystem", "read"),
)


def _file_stat(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    path = resolve_sandbox_path(context.sandbox_root, arguments["path"])
    if not path.exists():
        raise ToolExecutionError(f"Path not found: {path.name}")
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "type": "directory" if path.is_dir() else "file",
        "size_bytes": stat.st_size,
        "modified": dt.datetime.fromtimestamp(stat.st_mtime, tz=dt.timezone.utc).isoformat(),
        "mode": oct(stat.st_mode & 0o777),
        "suffix": path.suffix,
    }


# ---------------------------------------------------------------------------
# code execution sandbox
# ---------------------------------------------------------------------------
PYTHON_SANDBOX = Tool(
    tool_id="python.sandbox",
    name="Python sandbox",
    description=(
        "Run a short Python program in an isolated subprocess with CPU/memory/time limits and no network "
        "credentials. Requires tools.allow_code_execution = true. Returns real stdout/stderr/exit code."
    ),
    category="code",
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "maxLength": 200000},
            "timeout_s": {"type": "number", "minimum": 0.5, "maximum": 60},
            "memory_mb": {"type": "integer", "minimum": 64, "maximum": 4096},
        },
        "required": ["code"],
        "additionalProperties": False,
    },
    requires_code_execution=True,
    timeout_s=30.0,
    output_schema={
        "type": "object",
        "properties": {
            "exit_code": {"type": "integer"},
            "stdout": {"type": "string"},
            "stderr": {"type": "string"},
            "duration_ms": {"type": "number"},
        },
    },
    tags=("code", "sandbox"),
)


def _python_sandbox(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    sandbox_root = context.sandbox_root or Path.cwd()
    workdir = sandbox_root / ".alphaai-sandbox" / (context.run_id or "adhoc")
    workdir.mkdir(parents=True, exist_ok=True)
    script = workdir / "main.py"
    script.write_text(arguments["code"], encoding="utf-8")

    timeout = float(arguments.get("timeout_s", min(context.timeout_s, 30.0)))
    memory_mb = int(arguments.get("memory_mb", 1024))
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(workdir),
        "TMPDIR": str(workdir),
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": "",
        "PYTHONIOENCODING": "utf-8",
    }

    def _limits() -> None:  # pragma: no cover - runs in the child process
        try:
            import resource

            cpu = int(timeout) + 1
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
            resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024 * 1024, memory_mb * 1024 * 1024))
            resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        except (ImportError, ValueError, OSError):
            pass

    started = time.time()
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-B", str(script)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(workdir),
            env=env,
            preexec_fn=_limits if os.name == "posix" else None,  # noqa: PLW1509 - intentional hardening
            start_new_session=True,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolExecutionError(
            f"Sandboxed Python exceeded {timeout:.0f}s and was terminated.",
            details={"stdout": (exc.stdout or b"").decode("utf-8", "ignore")[:2000] if isinstance(exc.stdout, bytes) else exc.stdout},
        ) from exc
    duration = (time.time() - started) * 1000
    return {
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "duration_ms": duration,
        "workdir": str(workdir),
        "python": sys.version.split()[0],
    }


# ---------------------------------------------------------------------------
# network tools
# ---------------------------------------------------------------------------
_HTTP_ALLOWED_SCHEMES = {"http", "https"}
_BLOCKED_HOST_PATTERNS = (
    re.compile(r"^(localhost|127\.|0\.0\.0\.0|::1)", re.IGNORECASE),
    re.compile(r"^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)", re.IGNORECASE),
    re.compile(r"^169\.254\.", re.IGNORECASE),
)


def _assert_public_host(host: str) -> None:
    """Block obvious SSRF targets (loopback, link-local, private ranges)."""

    if any(pattern.search(host) for pattern in _BLOCKED_HOST_PATTERNS):
        raise ToolExecutionError(
            f"Refusing to fetch a private/loopback address ({host}). AlphaAI tools only reach public hosts."
        )
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ToolExecutionError(f"Cannot resolve host '{host}': {exc}") from exc
    for info in infos:
        address = info[4][0]
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            continue
        if parsed.is_private or parsed.is_loopback or parsed.is_link_local or parsed.is_reserved:
            raise ToolExecutionError(f"Host '{host}' resolves to a private address ({address}); refusing.")


HTTP_FETCH = Tool(
    tool_id="http.fetch",
    name="HTTP fetch",
    description=(
        "Fetch a public http(s) URL and return the real status, headers and (for text types) the body. "
        "Requires network permission. Private and loopback addresses are refused."
    ),
    category="network",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "maxLength": 2000},
            "method": {"type": "string", "enum": ["GET", "HEAD", "POST"]},
            "body": {"type": "string", "maxLength": 100000},
            "content_type": {"type": "string", "maxLength": 100},
            "max_bytes": {"type": "integer", "minimum": 1024, "maximum": 5000000},
            "user_agent": {"type": "string", "maxLength": 200},
        },
        "required": ["url"],
        "additionalProperties": False,
    },
    requires_network=True,
    timeout_s=20.0,
    tags=("network", "web"),
)


def _http_fetch(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    if not context.allow_network:
        raise PermissionError("Network access is not permitted for this call.")
    url = arguments["url"]
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() not in _HTTP_ALLOWED_SCHEMES:
        raise ToolExecutionError(f"Only http/https URLs are allowed, got '{parsed.scheme or 'none'}'.")
    if not parsed.hostname:
        raise ToolExecutionError("URL has no host component.")
    _assert_public_host(parsed.hostname)

    method = arguments.get("method", "GET").upper()
    max_bytes = int(arguments.get("max_bytes", min(context.max_output_bytes, 1_000_000)))
    data = arguments.get("body")
    request = urllib.request.Request(url, method=method)
    request.add_header(
        "User-Agent",
        arguments.get("user_agent") or f"AlphaAI/{VERSION} (+local inference; no API keys)",
    )
    request.add_header("Accept", "text/html,application/json,text/plain;q=0.9,*/*;q=0.5")
    if data is not None:
        request.data = data.encode("utf-8")
        request.add_header("Content-Type", arguments.get("content_type") or "application/json")
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=min(context.timeout_s, 20.0)) as response:
            raw = response.read(max_bytes)
            headers = dict(response.headers.items())
            status = response.status
            content_type = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        raw = exc.read(max_bytes) if hasattr(exc, "read") else b""
        headers = dict(exc.headers.items()) if exc.headers else {}
        status = exc.code
        content_type = headers.get("Content-Type", "")
    except urllib.error.URLError as exc:
        raise ToolExecutionError(f"Network request failed: {exc.reason}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise ToolExecutionError("Network request timed out.") from exc

    text: str | None = None
    if content_type and any(token in content_type for token in ("text/", "json", "xml", "javascript", "html")):
        charset = "utf-8"
        match = re.search(r"charset=([\w\-]+)", content_type)
        if match:
            charset = match.group(1)
        try:
            text = raw.decode(charset, errors="replace")
        except LookupError:
            text = raw.decode("utf-8", errors="replace")
    return {
        "url": url,
        "status": status,
        "content_type": content_type,
        "headers": {key: value for key, value in headers.items() if key.lower() in {"content-type", "content-length", "location", "server", "date", "cache-control"}},
        "bytes": len(raw),
        "text": text,
        "duration_ms": (time.time() - started) * 1000,
        "truncated": len(raw) >= max_bytes,
    }


WEB_SEARCH = Tool(
    tool_id="web.search",
    name="Web search",
    description=(
        "Run a real web search through DuckDuckGo's HTML endpoint (no API key required) and return "
        "the result titles, URLs and snippets. Requires network permission."
    ),
    category="network",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "maxLength": 500},
            "limit": {"type": "integer", "minimum": 1, "maximum": 25},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    requires_network=True,
    timeout_s=20.0,
    tags=("network", "research"),
)


def _web_search(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    if not context.allow_network:
        raise PermissionError("Network access is not permitted for this call.")
    query = arguments["query"]
    limit = int(arguments.get("limit", 8))
    payload = urllib.parse.urlencode({"q": query, "kl": "wt-wt"}).encode("utf-8")
    request = urllib.request.Request("https://html.duckduckgo.com/html/", data=payload, method="POST")
    request.add_header("User-Agent", f"AlphaAI/{VERSION} (+local inference; no API keys)")
    request.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(request, timeout=min(context.timeout_s, 20.0)) as response:
            html = response.read(2_000_000).decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        raise ToolExecutionError(f"Web search failed: {exc}") from exc

    results: list[dict[str, str]] = []
    for block in re.finditer(
        r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'(?:class="result__snippet"[^>]*>(.*?)</a>)?',
        html,
        flags=re.DOTALL,
    ):
        url, title, snippet = block.group(1), block.group(2), block.group(3) or ""
        results.append(
            {
                "title": _strip_tags(title).strip(),
                "url": _decode_ddg_url(url),
                "snippet": _strip_tags(snippet).strip(),
            }
        )
        if len(results) >= limit:
            break
    if not results:
        # DuckDuckGo may respond with different markup or a challenge page.
        if "anomaly" in html.lower() or "challenge" in html.lower() or "not found" in html.lower():
            raise ToolExecutionError(
                "DuckDuckGo returned a challenge/empty page for this query. "
                "Retry later or use http.fetch against a specific search/docs URL."
            )
        raise ToolExecutionError("Web search returned no parsable results.")
    return {"query": query, "provider": "duckduckgo-html", "count": len(results), "results": results}


def _strip_tags(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value)


def _decode_ddg_url(url: str) -> str:
    if url.startswith("//"):
        url = "https:" + url
    parsed = urllib.parse.urlparse(url)
    params = urllib.parse.parse_qs(parsed.query)
    if "uddg" in params:
        return urllib.parse.unquote(params["uddg"][0])
    return url


# ---------------------------------------------------------------------------
# system introspection
# ---------------------------------------------------------------------------
SYSTEM_INFO = Tool(
    tool_id="system.info",
    name="System info",
    description="Report the real AlphaAI version, hardware (CPU, RAM, accelerators), disk space and available runtimes.",
    category="system",
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
    timeout_s=20.0,
    tags=("system", "read"),
)


def _system_info(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    from ...engines.hardware import describe_hardware, detect_hardware

    hardware = detect_hardware([context.config.paths.models_dir, context.config.paths.state_dir])
    return {
        "alphaai": {"version": VERSION, "attribution": ATTRIBUTION_SHORT, "name": DISPLAY_NAME},
        "hardware": hardware.to_dict(),
        "summary": describe_hardware(hardware),
        "config_source": context.config.source,
        "model_engines": context.config.engines.enabled,
    }


CLOCK = Tool(
    tool_id="clock.now",
    name="Current time",
    description="Return the current UTC time plus epoch seconds and the local timezone offset.",
    category="system",
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
    timeout_s=2.0,
    tags=("system", "deterministic"),
)


def _clock(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    now = dt.datetime.now(dt.timezone.utc)
    return {
        "utc": now.isoformat(),
        "epoch": now.timestamp(),
        "local": dt.datetime.now().astimezone().isoformat(),
    }


SKILL_RUN = Tool(
    tool_id="skill.run",
    name="Run AlphaAI skill",
    description=(
        "Invoke another AlphaAI skill by id with a JSON input object. Used for tool automation and "
        "agent workflows; requires the skill to be enabled."
    ),
    category="orchestration",
    parameters={
        "type": "object",
        "properties": {
            "skill_id": {"type": "string", "maxLength": 100},
            "input": {"type": "object"},
        },
        "required": ["skill_id"],
        "additionalProperties": False,
    },
    timeout_s=60.0,
    tags=("orchestration",),
)


def _skill_run(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    manager = context.metadata.get("skill_manager")
    if manager is None:
        raise ToolExecutionError(
            "No AlphaAI skill manager is attached to this tool call.",
            remediation="Run skills through the AlphaAI runtime/API, which wires the skill manager.",
        )
    result = manager.run_result(arguments["skill_id"], arguments.get("input") or {}, run_id=context.run_id)
    return result.to_dict()


# ---------------------------------------------------------------------------
# registry construction
# ---------------------------------------------------------------------------
#: tool id -> real handler. The tool descriptors above are declared before the
#: handler functions exist, so handlers are bound here — a tool can therefore
#: never be listed without an implementation, and an unbound tool refuses to run.
_TOOL_HANDLERS: dict[str, Any] = {
    "calculator.evaluate": _calculator,
    "text.transform": _text_transform,
    "text.regex": _text_regex,
    "json.path": _json_path,
    "file.read": _file_read,
    "file.list": _file_list,
    "file.write": _file_write,
    "file.hash": _file_hash,
    "file.stat": _file_stat,
    "python.sandbox": _python_sandbox,
    "http.fetch": _http_fetch,
    "web.search": _web_search,
    "system.info": _system_info,
    "clock.now": _clock,
    "skill.run": _skill_run,
}


def build_tools() -> list[Tool]:
    """Return every AlphaAI built-in tool, with its real handler bound."""

    tools = [
        CALCULATOR,
        TEXT_TRANSFORM,
        TEXT_REGEX,
        JSON_PATH,
        FILE_READ,
        FILE_LIST,
        FILE_WRITE,
        FILE_HASH,
        FILE_STAT,
        PYTHON_SANDBOX,
        HTTP_FETCH,
        WEB_SEARCH,
        SYSTEM_INFO,
        CLOCK,
        SKILL_RUN,
    ]
    for tool in tools:
        if tool.handler is None:
            tool.handler = _TOOL_HANDLERS.get(tool.tool_id)
    unbound = [tool.tool_id for tool in tools if tool.handler is None]
    if unbound:  # pragma: no cover - installation check
        raise ToolExecutionError(
            f"AlphaAI built-in tools have no handler bound: {', '.join(unbound)}",
            remediation="Reinstall AlphaAI (`pip install -e .`); the installation is incomplete.",
        )
    return tools
