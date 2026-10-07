"""Text-oriented AlphaAI skills: transformation, summarization, translation.

* ``skill.text_transform`` runs deterministic text pipelines — always real.
* ``skill.summarization`` defaults to a real extractive algorithm (TF-IDF
  sentence ranking with position and length priors) and can delegate to a live
  engine in ``engine`` mode.
* ``skill.translation`` is engine-backed only. AlphaAI does not ship a fake
  translator: with no engine available the skill reports a structured error.
"""

from __future__ import annotations

import math
import re
from typing import Any, Sequence

from ..errors import SkillExecutionError
from .base import Skill, SkillContext

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+|\n{2,}")
_WORD = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "of", "to", "in", "on", "for", "with",
    "at", "by", "from", "as", "is", "are", "was", "were", "be", "been", "being", "it", "its", "this",
    "that", "these", "those", "there", "here", "we", "you", "they", "he", "she", "i", "not", "no",
    "so", "such", "also", "can", "will", "would", "should", "could", "do", "does", "did", "has",
    "have", "had", "which", "who", "whom", "what", "when", "where", "how", "why", "into", "over",
    "under", "about", "more", "most", "some", "any", "all", "each", "other", "only", "very", "just",
}


# ===========================================================================
# 1. text transformation
# ===========================================================================
_TRANSFORM_OPS = (
    "replace", "regex_sub", "regex_extract", "upper", "lower", "title", "capitalize", "strip",
    "collapse_whitespace", "remove_empty_lines", "slugify", "snake_case", "kebab_case", "camel_case",
    "wrap", "indent", "prefix_lines", "template", "truncate", "reverse", "sort_lines", "unique_lines",
    "join_lines", "split_lines", "append", "prepend",
)


class TextTransformSkill(Skill):
    skill_id = "skill.text_transform"
    name = "Text transformation"
    description = (
        "Apply a deterministic pipeline of real text operations (replace, regex substitution, case "
        "conversion, wrapping, templating, line operations) and report exactly what changed."
    )
    category = "text"
    input_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "maxLength": 1000000},
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string", "enum": list(_TRANSFORM_OPS)},
                        "name": {"type": "string"},
                        "value": {},
                        "pattern": {"type": "string", "maxLength": 500},
                        "replacement": {"type": "string", "maxLength": 10000},
                        "ignore_case": {"type": "boolean"},
                        "count": {"type": "integer", "minimum": 1, "maximum": 100000},
                        "width": {"type": "integer", "minimum": 1, "maximum": 500},
                        "prefix": {"type": "string", "maxLength": 200},
                        "drop_empty": {"type": "boolean"},
                    },
                    "required": ["op"],
                },
            },
        },
        "required": ["text", "steps"],
    }
    output_schema = {"type": "object", "properties": {"text": {"type": "string"}, "steps": {"type": "array"}}}
    timeout_s = 15.0
    tags = ("text", "deterministic")

    def run(self, inputs: dict[str, Any], context: SkillContext) -> dict[str, Any]:
        text = inputs["text"]
        steps = inputs.get("steps") or []
        if not steps:
            raise SkillExecutionError("Provide at least one transformation step.")
        trace: list[dict[str, Any]] = []
        for index, step in enumerate(steps):
            operation = step["op"]
            before = text
            text = self._apply(operation, text, step, index)
            trace.append(
                {
                    "step": index + 1,
                    "name": step.get("name") or operation,
                    "op": operation,
                    "length_before": len(before),
                    "length_after": len(text),
                    "changed": before != text,
                }
            )
        return {
            "text": text,
            "steps": trace,
            "length": len(text),
            "words": len(_WORD.findall(text)),
            "lines": text.count("\n") + 1,
        }

    def _apply(self, operation: str, text: str, step: dict[str, Any], index: int) -> str:
        value = step.get("value")
        if operation == "replace":
            if value is None:
                raise SkillExecutionError(f"Step {index + 1}: replace requires 'value' (the substring to find).")
            replacement = step.get("replacement")
            if replacement is None:
                raise SkillExecutionError(f"Step {index + 1}: replace requires 'replacement'.")
            count = int(step.get("count", 0))
            if step.get("ignore_case"):
                return re.sub(re.escape(str(value)), replacement.replace("\\", "\\\\"), text, count=count or 0, flags=re.IGNORECASE)
            return text.replace(str(value), replacement, count if count else -1)
        if operation == "regex_sub":
            pattern = step.get("pattern")
            if not pattern:
                raise SkillExecutionError(f"Step {index + 1}: regex_sub requires 'pattern'.")
            flags = re.MULTILINE | (re.IGNORECASE if step.get("ignore_case") else 0)
            try:
                return re.sub(pattern, step.get("replacement", ""), text, count=int(step.get("count", 0)), flags=flags)
            except re.error as exc:
                raise SkillExecutionError(f"Step {index + 1}: invalid regex: {exc}") from exc
        if operation == "regex_extract":
            pattern = step.get("pattern")
            if not pattern:
                raise SkillExecutionError(f"Step {index + 1}: regex_extract requires 'pattern'.")
            try:
                matches = re.findall(pattern, text, flags=re.MULTILINE)
            except re.error as exc:
                raise SkillExecutionError(f"Step {index + 1}: invalid regex: {exc}") from exc
            flat = ["".join(item) if isinstance(item, tuple) else item for item in matches]
            return "\n".join(flat)
        if operation == "upper":
            return text.upper()
        if operation == "lower":
            return text.lower()
        if operation == "title":
            return text.title()
        if operation == "capitalize":
            return text.capitalize()
        if operation == "strip":
            return text.strip()
        if operation == "collapse_whitespace":
            return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()
        if operation == "remove_empty_lines":
            return "\n".join(line for line in text.splitlines() if line.strip())
        if operation == "slugify":
            return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
        if operation == "snake_case":
            return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
        if operation == "kebab_case":
            return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
        if operation == "camel_case":
            parts = [part for part in re.split(r"[^A-Za-z0-9]+", text) if part]
            return (parts[0].lower() + "".join(part.capitalize() for part in parts[1:])) if parts else ""
        if operation == "wrap":
            width = int(step.get("width", 80))
            lines: list[str] = []
            for paragraph in text.split("\n"):
                words = paragraph.split(" ")
                current = ""
                for word in words:
                    if not current:
                        current = word
                    elif len(current) + 1 + len(word) <= width:
                        current += " " + word
                    else:
                        lines.append(current)
                        current = word
                lines.append(current)
            return "\n".join(lines)
        if operation == "indent":
            prefix = step.get("prefix") or (step.get("value") if isinstance(step.get("value"), str) else "  ")
            return "\n".join((prefix + line) if line.strip() else line for line in text.splitlines())
        if operation == "prefix_lines":
            prefix = str(step.get("prefix") or step.get("value") or "")
            return "\n".join(prefix + line for line in text.splitlines())
        if operation == "template":
            variables = step.get("value")
            if not isinstance(variables, dict):
                raise SkillExecutionError(f"Step {index + 1}: template requires 'value' to be an object of variables.")
            try:
                return text.format(**variables)
            except (KeyError, IndexError, ValueError) as exc:
                raise SkillExecutionError(f"Step {index + 1}: template substitution failed: {exc}") from exc
        if operation == "truncate":
            limit = int(step.get("count") or step.get("width") or 500)
            suffix = "" if len(text) <= limit else "…"
            return text[:limit] + suffix
        if operation == "reverse":
            return text[::-1]
        if operation == "sort_lines":
            return "\n".join(sorted(text.splitlines(), key=str.lower))
        if operation == "unique_lines":
            seen: list[str] = []
            for line in text.splitlines():
                if line not in seen:
                    seen.append(line)
            return "\n".join(seen)
        if operation == "join_lines":
            separator = step.get("prefix") if isinstance(step.get("prefix"), str) else " "
            return separator.join(line.strip() for line in text.splitlines() if line.strip())
        if operation == "split_lines":
            return "\n".join(text.split(str(step.get("value") or ";")))
        if operation == "append":
            return text + str(value if value is not None else "")
        if operation == "prepend":
            return str(value if value is not None else "") + text
        raise SkillExecutionError(f"Step {index + 1}: unsupported operation '{operation}'.")


# ===========================================================================
# 2. summarization
# ===========================================================================
class SummarizationSkill(Skill):
    skill_id = "skill.summarization"
    name = "Summarization"
    description = (
        "Summarise real text. Default mode 'extractive' ranks the original sentences by TF-IDF with "
        "position and length priors (deterministic, no model needed); mode 'engine' delegates to an "
        "available AlphaAI engine."
    )
    category = "text"
    required_tools = ("file.read",)
    input_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "maxLength": 2000000},
            "path": {"type": "string"},
            "mode": {"type": "string", "enum": ["extractive", "engine"]},
            "max_sentences": {"type": "integer", "minimum": 1, "maximum": 200},
            "ratio": {"type": "number", "minimum": 0.01, "maximum": 1.0},
            "language": {"type": "string", "maxLength": 40},
        },
    }
    output_schema = {"type": "object", "properties": {"summary": {"type": "string"}, "mode": {"type": "string"}}}
    timeout_s = 60.0
    tags = ("text", "summarization")

    def run(self, inputs: dict[str, Any], context: SkillContext) -> dict[str, Any]:
        text = inputs.get("text")
        if text is None and inputs.get("path"):
            text = context.tool_output("file.read", {"path": inputs["path"], "limit": 20000}).get("content", "")
        if not text or not str(text).strip():
            raise SkillExecutionError("Provide 'text' or a readable 'path' to summarise.")

        mode = inputs.get("mode") or "extractive"
        sentences = _split_sentences(str(text))
        if not sentences:
            raise SkillExecutionError("No sentences were found in the supplied text.")

        if mode == "extractive":
            return self._extractive(sentences, str(text), inputs)

        language = inputs.get("language")
        target = inputs.get("max_sentences") or max(1, round(len(sentences) * float(inputs.get("ratio") or 0.2)))
        prompt = (
            f"Summarise the following text in at most {target} sentences"
            + (f" (language: {language})" if language else "")
            + ". Return only the summary, grounded strictly in the text.\n\n"
            + str(text)
        )
        result = context.require_engine(prompt, task="summarization", engine_id=inputs.get("engine_id"))
        return {
            "mode": "engine",
            "summary": result.text.strip(),
            "engine": {"engine_id": result.engine_id, "model": result.model, "runtime": result.runtime},
            "engine_id": result.engine_id,
            "source_sentences": len(sentences),
            "usage": result.usage.to_dict(),
            "attribution": result.attribution,
        }

    def _extractive(self, sentences: Sequence[str], text: str, inputs: dict[str, Any]) -> dict[str, Any]:
        total = len(sentences)
        target = int(inputs.get("max_sentences") or max(1, round(total * float(inputs.get("ratio") or 0.25))))
        target = max(1, min(target, total))

        document_frequency: dict[str, int] = {}
        sentence_tokens: list[list[str]] = []
        for sentence in sentences:
            tokens = [token.lower() for token in _WORD.findall(sentence)]
            sentence_tokens.append(tokens)
            for token in set(tokens):
                document_frequency[token] = document_frequency.get(token, 0) + 1

        scored: list[tuple[int, float, str]] = []
        for index, (sentence, tokens) in enumerate(zip(sentences, sentence_tokens)):
            content = [token for token in tokens if token not in _STOPWORDS and len(token) > 2]
            if not content:
                score = 0.0
            else:
                tf = {}
                for token in content:
                    tf[token] = tf.get(token, 0) + 1
                score = sum((1 + math.log(count)) * math.log(1 + total / (1 + document_frequency.get(token, 1))) for token, count in tf.items())
                score /= math.sqrt(len(content))
            position_prior = 1.25 if index == 0 else (1.1 if index < max(2, total * 0.1) else 1.0)
            length_penalty = 0.6 if len(content) < 4 else (1.0 if len(content) <= 60 else 0.85)
            scored.append((index, score * position_prior * length_penalty, sentence))

        ranked = sorted(scored, key=lambda item: (-item[1], item[0]))
        chosen = sorted(ranked[:target], key=lambda item: item[0])
        summary = " ".join(sentence.strip() for _, _, sentence in chosen)
        return {
            "mode": "extractive",
            "summary": summary,
            "selected_sentences": [
                {"index": index, "score": round(score, 6), "text": sentence.strip()} for index, score, sentence in chosen
            ],
            "source_sentences": total,
            "compression": round(len(summary) / max(len(text), 1), 4),
            "algorithm": "tfidf_sentence_ranking_with_position_prior",
        }


def _split_sentences(text: str) -> list[str]:
    raw = _SENTENCE_SPLIT.split(text.strip())
    sentences: list[str] = []
    for chunk in raw:
        candidate = chunk.strip()
        if not candidate:
            continue
        if len(candidate) > 400:
            # split very long paragraphs on semicolons as a secondary boundary
            parts = [part.strip() for part in re.split(r"(?<=[;:])\s+", candidate) if part.strip()]
            sentences.extend(parts)
        else:
            sentences.append(candidate)
    return sentences


# ===========================================================================
# 3. translation (engine-backed, honest about availability)
# ===========================================================================
class TranslationSkill(Skill):
    skill_id = "skill.translation"
    name = "Translation"
    description = (
        "Translate text using a real local AlphaAI engine. Requires an available engine with the "
        "'multilingual' capability — AlphaAI has no offline dictionary masquerading as a translator, so "
        "without a local model this skill reports an explicit error."
    )
    category = "language"
    required_capabilities = ()
    requires_engine = True
    input_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "maxLength": 200000},
            "path": {"type": "string"},
            "target_language": {"type": "string", "maxLength": 60},
            "source_language": {"type": "string", "maxLength": 60},
            "register": {"type": "string", "enum": ["neutral", "formal", "informal", "technical"]},
            "glossary": {"type": "object", "description": "Terms that must be preserved verbatim."},
            "engine_id": {"type": "string"},
        },
        "required": ["target_language"],
    }
    output_schema = {"type": "object", "properties": {"translation": {"type": "string"}, "target_language": {"type": "string"}}}
    timeout_s = 120.0
    tags = ("language", "engine-backed")

    def run(self, inputs: dict[str, Any], context: SkillContext) -> dict[str, Any]:
        text = inputs.get("text")
        if text is None and inputs.get("path"):
            text = context.tool_output("file.read", {"path": inputs["path"], "limit": 20000}).get("content", "")
        if not text or not str(text).strip():
            raise SkillExecutionError("Provide 'text' or a readable 'path' to translate.")

        target = inputs["target_language"]
        source = inputs.get("source_language") or "auto-detect"
        register = inputs.get("register") or "neutral"
        glossary = inputs.get("glossary") or {}

        instruction = (
            f"Translate the text below from {source} into {target}. "
            f"Register: {register}. Preserve formatting, numbers, code and proper nouns. "
            "Return only the translation."
        )
        if glossary:
            terms = "; ".join(f"{key} -> {value}" for key, value in glossary.items())
            instruction += f" Always translate these terms exactly like this: {terms}."
        prompt = f"{instruction}\n\nTEXT:\n{text}"

        result = context.require_engine(
            prompt,
            task="translation",
            engine_id=inputs.get("engine_id"),
            required_capabilities=["chat", "multilingual"],
        )
        return {
            "translation": result.text.strip(),
            "source_language": source,
            "target_language": target,
            "register": register,
            "glossary_applied": sorted(glossary) if glossary else [],
            "engine": {"engine_id": result.engine_id, "model": result.model, "runtime": result.runtime},
            "engine_id": result.engine_id,
            "usage": result.usage.to_dict(),
            "attribution": result.attribution,
        }


def build_skills() -> list[Skill]:
    return [TextTransformSkill(), SummarizationSkill(), TranslationSkill()]
