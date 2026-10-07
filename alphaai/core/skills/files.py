"""File-oriented AlphaAI skills: file analysis and document processing.

Both skills go through the AlphaAI tool layer (``file.read``/``file.stat``), so
the sandbox root and the tool permission policy always apply. Real bytes are
inspected; unsupported formats produce a structured error instead of a made-up
extraction.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import re
import unicodedata
from typing import Any, Sequence

from ..errors import SkillExecutionError
from .base import Skill, SkillContext

# ---------------------------------------------------------------------------
# real format sniffing from magic bytes
# ---------------------------------------------------------------------------
_MAGIC_SIGNATURES: tuple[tuple[bytes, str, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png", "PNG image"),
    (b"\xff\xd8\xff", "image/jpeg", "JPEG image"),
    (b"GIF87a", "image/gif", "GIF image"),
    (b"GIF89a", "image/gif", "GIF image"),
    (b"%PDF-", "application/pdf", "PDF document"),
    (b"PK\x03\x04", "application/zip", "ZIP archive (docx/xlsx/pptx/zip)"),
    (b"\x7fELF", "application/x-executable", "ELF binary"),
    (b"GGUF", "application/gguf", "GGUF model weights"),
    (b"\x1f\x8b", "application/gzip", "gzip archive"),
    (b"BZh", "application/bzip2", "bzip2 archive"),
    (b"ID3", "audio/mpeg", "MP3 audio"),
    (b"OggS", "audio/ogg", "Ogg audio"),
    (b"RIFF", "application/riff", "RIFF container (wav/avi/webp)"),
    (b"\x00asm", "application/wasm", "WebAssembly module"),
    (b"SQLite format 3\x00", "application/sqlite", "SQLite database"),
)

_LANGUAGE_BY_SUFFIX = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "typescript", ".jsx": "javascript",
    ".go": "go", ".rs": "rust", ".java": "java", ".c": "c", ".h": "c", ".cpp": "cpp", ".cs": "csharp",
    ".rb": "ruby", ".php": "php", ".sh": "shell", ".sql": "sql", ".json": "json", ".yaml": "yaml",
    ".yml": "yaml", ".toml": "toml", ".md": "markdown", ".html": "html", ".css": "css", ".csv": "csv",
}


def _sniff(sample: bytes) -> tuple[str, str]:
    for signature, mime, description in _MAGIC_SIGNATURES:
        if sample.startswith(signature):
            return mime, description
    return "application/octet-stream", "unrecognised binary"


def _is_probably_text(sample: bytes) -> bool:
    if not sample:
        return True
    if b"\x00" in sample:
        return False
    printable = sum(1 for byte in sample if 9 <= byte <= 13 or 32 <= byte <= 126 or byte >= 0x80)
    return printable / len(sample) > 0.9


# ===========================================================================
# 1. file analysis
# ===========================================================================
class FileAnalysisSkill(Skill):
    skill_id = "skill.file_analysis"
    name = "File analysis"
    description = (
        "Inspect real files inside the AlphaAI sandbox: size, timestamps, SHA-256, magic-byte type "
        "detection, text/binary classification, line and word counts, and structural hints for "
        "JSON/CSV/Markdown/code files."
    )
    category = "files"
    required_tools = ("file.stat", "file.read", "file.hash", "file.list")
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "paths": {"type": "array", "items": {"type": "string"}},
            "directory": {"type": "string", "description": "Analyse every file in this directory."},
            "include_hash": {"type": "boolean", "default": True},
            "preview_chars": {"type": "integer", "minimum": 0, "maximum": 2000, "default": 240},
            "recursive": {"type": "boolean", "default": False},
            "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 25},
        },
    }
    output_schema = {"type": "object", "properties": {"count": {"type": "integer"}, "files": {"type": "array"}}}
    timeout_s = 60.0
    tags = ("files", "inspection")

    def run(self, inputs: dict[str, Any], context: SkillContext) -> dict[str, Any]:
        targets = list(inputs.get("paths") or [])
        if inputs.get("path"):
            targets.append(inputs["path"])
        if inputs.get("directory"):
            listing = context.tool_output(
                "file.list",
                {
                    "path": inputs["directory"],
                    "recursive": bool(inputs.get("recursive")),
                    "limit": int(inputs.get("limit", 25)) * 4,
                },
            )
            targets.extend(
                entry["path"] for entry in listing.get("entries", []) if entry.get("type") == "file"
            )
        if not targets:
            raise SkillExecutionError("Provide 'path', 'paths' or 'directory'.")

        limit = int(inputs.get("limit", 25))
        include_hash = inputs.get("include_hash", True)
        preview_chars = int(inputs.get("preview_chars", 240))
        analysed: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []

        for target in targets[:limit]:
            try:
                analysed.append(self._analyse_one(target, context, include_hash=include_hash, preview_chars=preview_chars))
            except Exception as exc:  # noqa: BLE001 - report per-file failures honestly
                errors.append({"path": target, "error": f"{type(exc).__name__}: {exc}"})

        return {
            "count": len(analysed),
            "files": analysed,
            "errors": errors,
            "sandbox_root": str(context.tools.config.tools.sandbox_root) if context.tools else None,
        }

    def _analyse_one(self, target: str, context: SkillContext, *, include_hash: bool, preview_chars: int) -> dict[str, Any]:
        stat = context.tool_output("file.stat", {"path": target})
        entry: dict[str, Any] = {
            "path": stat["path"],
            "name": stat["path"].rsplit("/", 1)[-1],
            "type": stat["type"],
            "size_bytes": stat["size_bytes"],
            "modified": stat["modified"],
            "suffix": stat["suffix"],
            "language": _LANGUAGE_BY_SUFFIX.get(stat["suffix"]),
        }
        if include_hash and stat["type"] == "file":
            entry["sha256"] = context.tool_output("file.hash", {"path": target, "algorithm": "sha256"})["digest"]

        if stat["type"] != "file":
            return entry

        read = context.tool_output("file.read", {"path": target, "limit": 20000})
        entry["sha256"] = read.get("sha256", entry.get("sha256"))
        if read.get("encoding") == "binary":
            sample = base64.b64decode(read.get("base64", "")[:64])
            mime, description = _sniff(sample)
            entry.update({"text": False, "mime_type": mime, "format": description, "truncated": read.get("truncated", False)})
            return entry

        text = read.get("content", "")
        lines = text.splitlines()
        words = re.findall(r"\S+", text)
        entry.update(
            {
                "text": True,
                "total_lines": read.get("total_lines", len(lines)),
                "lines_read": len(lines),
                "words_sampled": len(words),
                "characters_sampled": len(text),
                "encoding": read.get("encoding", "utf-8"),
                "sha256": read.get("sha256"),
            }
        )
        if preview_chars:
            entry["preview"] = text[:preview_chars]
        structure = _structure_of(entry["suffix"], text)
        if structure:
            entry["structure"] = structure
        return entry


def _structure_of(suffix: str, text: str) -> dict[str, Any] | None:
    try:
        if suffix == ".json":
            payload = json.loads(text)
            if isinstance(payload, dict):
                return {"kind": "object", "keys": list(payload)[:50], "key_count": len(payload)}
            if isinstance(payload, list):
                sample = payload[0] if payload else None
                return {
                    "kind": "array",
                    "length": len(payload),
                    "item_keys": sorted(sample) if isinstance(sample, dict) else None,
                }
        if suffix in {".csv", ".tsv"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            reader = csv.reader(io.StringIO(text), delimiter=delimiter)
            rows = list(reader)
            header = rows[0] if rows else []
            return {"kind": "table", "columns": header, "column_count": len(header), "row_count": max(0, len(rows) - 1)}
        if suffix in {".md", ".markdown"}:
            headings = re.findall(r"^(#{1,6})\s+(.*)$", text, flags=re.MULTILINE)
            links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text)
            return {
                "kind": "markdown",
                "heading_count": len(headings),
                "outline": [{"level": len(hashes), "title": title.strip()} for hashes, title in headings[:25]],
                "link_count": len(links),
            }
        if suffix in _LANGUAGE_BY_SUFFIX:
            return {
                "kind": "code",
                "language": _LANGUAGE_BY_SUFFIX[suffix],
                "comment_lines": len([line for line in text.splitlines() if line.strip().startswith(("#", "//", "/*", "*"))]),
                "blank_lines": len([line for line in text.splitlines() if not line.strip()]),
            }
    except (json.JSONDecodeError, csv.Error, UnicodeDecodeError, ValueError):
        return {"kind": "unparsed", "note": "Structure detection failed; the file may be truncated or malformed."}
    return None


# ===========================================================================
# 2. document processing
# ===========================================================================
_NORMALISERS = ("nfkc", "collapse_whitespace", "strip_control", "remove_empty_lines", "unify_quotes")


class DocumentProcessingSkill(Skill):
    skill_id = "skill.document_processing"
    name = "Document processing"
    description = (
        "Extract and normalise real document text: plain text, Markdown, HTML (tags/entities stripped), "
        "JSON, JSONL and CSV, plus PDF when the optional 'pypdf' package is installed. Produces "
        "structure information and token-aware chunks."
    )
    category = "files"
    required_tools = ("file.read",)
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "text": {"type": "string", "maxLength": 2000000},
            "format": {
                "type": "string",
                "enum": ["auto", "text", "markdown", "html", "json", "jsonl", "csv", "pdf"],
            },
            "normalise": {"type": "array", "items": {"type": "string", "enum": list(_NORMALISERS)}},
            "chunk_size": {"type": "integer", "minimum": 100, "maximum": 20000, "default": 1200},
            "chunk_overlap": {"type": "integer", "minimum": 0, "maximum": 5000, "default": 120},
            "max_chunks": {"type": "integer", "minimum": 1, "maximum": 5000, "default": 200},
            "include_normalised_text": {"type": "boolean", "default": False},
        },
    }
    output_schema = {"type": "object", "properties": {"format": {"type": "string"}, "chunks": {"type": "array"}}}
    timeout_s = 60.0
    tags = ("documents", "extraction")

    def run(self, inputs: dict[str, Any], context: SkillContext) -> dict[str, Any]:
        text = inputs.get("text")
        fmt = inputs.get("format") or "auto"
        source = "inline"

        if text is None:
            path = inputs.get("path")
            if not path:
                raise SkillExecutionError("Provide 'text' or a readable 'path'.")
            source = path
            if fmt == "auto":
                fmt = _format_from_suffix(path)
            if fmt == "pdf":
                text = _extract_pdf(context, path)
            else:
                read = context.tool_output("file.read", {"path": path, "limit": 20000})
                if read.get("encoding") == "binary":
                    raise SkillExecutionError(
                        f"'{path}' is a binary file ({read.get('size_bytes')} bytes) and cannot be read as text.",
                        remediation="Use skill.file_analysis for binary inspection, or provide a text document.",
                    )
                text = read.get("content", "")

        text = str(text)
        if not text.strip():
            raise SkillExecutionError("The document is empty.")

        if fmt == "auto":
            fmt = _detect_format(text)

        structure: dict[str, Any] = {}
        if fmt == "html":
            links = re.findall(r'href="([^"]+)"', text, flags=re.IGNORECASE)
            structure = {"title": (re.search(r"<title[^>]*>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL) or [None, None])[1] if re.search(r"<title", text, re.IGNORECASE) else None, "link_count": len(links), "links": links[:50]}
            text = _html_to_text(text)
        elif fmt == "markdown":
            headings = re.findall(r"^(#{1,6})\s+(.*)$", text, flags=re.MULTILINE)
            structure = {"heading_count": len(headings), "outline": [{"level": len(h), "title": t.strip()} for h, t in headings[:50]]}
        elif fmt == "json":
            payload = json.loads(text)
            if isinstance(payload, dict):
                structure = {"kind": "object", "keys": list(payload)[:50]}
                text = _flatten_json(payload)
            elif isinstance(payload, list):
                structure = {"kind": "array", "length": len(payload)}
                text = "\n".join(_flatten_json(item) for item in payload[:500])
        elif fmt == "jsonl":
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
            structure = {"kind": "jsonl", "records": len(records), "keys": sorted({key for record in records if isinstance(record, dict) for key in record})[:50]}
            text = "\n".join(_flatten_json(record) for record in records[:500])
        elif fmt == "csv":
            rows = list(csv.reader(io.StringIO(text)))
            header = rows[0] if rows else []
            structure = {"kind": "table", "columns": header, "row_count": max(0, len(rows) - 1)}
            text = "\n".join(" | ".join(row) for row in rows[:500])

        normalised = _normalise(text, inputs.get("normalise") or ["nfkc", "collapse_whitespace", "strip_control"])
        chunks = _chunk(normalised, int(inputs.get("chunk_size", 1200)), int(inputs.get("chunk_overlap", 120)), int(inputs.get("max_chunks", 200)))

        output: dict[str, Any] = {
            "source": source,
            "format": fmt,
            "characters": len(normalised),
            "words": len(normalised.split()),
            "chunk_count": len(chunks),
            "chunks": chunks,
            "structure": structure,
            "normalisation_applied": inputs.get("normalise") or ["nfkc", "collapse_whitespace", "strip_control"],
            "average_chunk_chars": round(len(normalised) / len(chunks), 1) if chunks else 0,
        }
        if inputs.get("include_normalised_text"):
            output["normalised_text"] = normalised[:50000]
        return output


def _format_from_suffix(path: str) -> str:
    suffix = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
    return {
        ".md": "markdown", ".markdown": "markdown", ".html": "html", ".htm": "html", ".json": "json",
        ".jsonl": "jsonl", ".ndjson": "jsonl", ".csv": "csv", ".tsv": "csv", ".pdf": "pdf",
    }.get(suffix, "text")


def _detect_format(text: str) -> str:
    stripped = text.lstrip()
    if stripped.startswith("<!DOCTYPE") or stripped.startswith("<html") or "<body" in stripped[:2000].lower():
        return "html"
    lines = [line for line in text.splitlines() if line.strip()]
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            json.loads(text)
            return "json"
        except json.JSONDecodeError:
            if lines and all(_looks_like_json(line) for line in lines[:5]):
                return "jsonl"
    if lines and all(line.count(",") >= 1 for line in lines[:5]) and "," in lines[0]:
        return "csv"
    if re.search(r"^(#{1,6})\s+", text, flags=re.MULTILINE):
        return "markdown"
    return "text"


def _looks_like_json(line: str) -> bool:
    try:
        json.loads(line)
        return True
    except json.JSONDecodeError:
        return False


def _html_to_text(html: str) -> str:
    import html as html_module

    without_scripts = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    with_breaks = re.sub(r"<(br|/p|/div|/li|/h[1-6]|/tr)[^>]*>", "\n", without_scripts, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", with_breaks)
    return html_module.unescape(text)


def _flatten_json(value: Any, prefix: str = "") -> str:
    if isinstance(value, dict):
        return "\n".join(_flatten_json(item, f"{prefix}{key}") for key, item in value.items())
    if isinstance(value, list):
        return "\n".join(_flatten_json(item, f"{prefix}[{index}]") for index, item in enumerate(value))
    return f"{prefix}: {value}" if prefix else str(value)


def _normalise(text: str, operations: Sequence[str]) -> str:
    result = text
    for operation in operations:
        if operation == "nfkc":
            result = unicodedata.normalize("NFKC", result)
        elif operation == "collapse_whitespace":
            result = re.sub(r"[ \t]+", " ", result)
            result = re.sub(r"\n{3,}", "\n\n", result).strip()
        elif operation == "strip_control":
            result = "".join(char for char in result if char == "\n" or char == "\t" or unicodedata.category(char)[0] != "C")
        elif operation == "remove_empty_lines":
            result = "\n".join(line for line in result.splitlines() if line.strip())
        elif operation == "unify_quotes":
            result = result.translate(str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'}))
    return result


def _chunk(text: str, size: int, overlap: int, max_chunks: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    if size <= overlap:
        overlap = max(0, size // 10)
    position = 0
    while position < len(text) and len(chunks) < max_chunks:
        end = min(len(text), position + size)
        if end < len(text):
            boundary = max(text.rfind("\n", position, end), text.rfind(". ", position, end))
            if boundary > position + size * 0.5:
                end = boundary + 1
        chunk = text[position:end]
        chunks.append(
            {
                "index": len(chunks),
                "start": position,
                "end": end,
                "characters": len(chunk),
                "estimated_tokens": max(1, len(chunk) // 4),
                "text": chunk,
            }
        )
        if end >= len(text):
            break
        position = max(end - overlap, position + 1)
    return chunks


def _extract_pdf(context: SkillContext, path: str) -> str:
    """Extract real PDF text when an optional extractor is installed."""

    read = context.tool_output("file.read", {"path": path, "limit": 1})
    if not read.get("sha256"):
        raise SkillExecutionError(f"Cannot read '{path}'.")
    try:
        import pypdf  # type: ignore
    except ModuleNotFoundError:
        try:
            import PyPDF2 as pypdf  # type: ignore
        except ModuleNotFoundError as exc:
            raise SkillExecutionError(
                f"PDF extraction for '{path}' needs the optional PDF library, which is not installed.",
                remediation=(
                    "Install it with `pip install 'pypdf>=4'`. AlphaAI does not simulate PDF extraction: "
                    "without the library it reports this error."
                ),
            ) from exc
    import io as _io

    raw = base64.b64decode(read.get("base64") or "")
    if not raw:
        resolved = context.tool("file.read", {"path": path})
        raise SkillExecutionError(
            "Reading a full PDF requires raw bytes; AlphaAI reads up to tools.max_output_bytes per call.",
            remediation="Increase tools.max_output_bytes or convert the PDF to text first.",
            details={"truncated": bool(resolved.metadata.get("truncated"))},
        )
    try:
        reader = pypdf.PdfReader(_io.BytesIO(raw))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # noqa: BLE001 - library-specific failures
        raise SkillExecutionError(f"PDF parsing failed: {type(exc).__name__}: {exc}") from exc
    return "\n\n".join(pages)


def build_skills() -> list[Skill]:
    return [FileAnalysisSkill(), DocumentProcessingSkill()]
