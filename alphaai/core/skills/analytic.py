"""Analytic AlphaAI skills: calculator, structured reasoning, data analysis.

Everything in this module computes real results: exact arithmetic, Gaussian
elimination, brute-force SAT, difference analysis, OLS regression and
descriptive statistics. When an input cannot be answered, the skill reports a
structured error instead of a fabricated answer.
"""

from __future__ import annotations

import itertools
import json
import math
import re
import statistics
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..calc import CalculationError, evaluate
from ..errors import SkillExecutionError
from .base import Skill, SkillContext

# ===========================================================================
# 1. calculator
# ===========================================================================
class CalculatorSkill(Skill):
    skill_id = "skill.calculator"
    name = "Calculator"
    description = (
        "Evaluate one or more mathematical expressions exactly (int/float/Fraction), with math "
        "functions and constants. Deterministic, no model required."
    )
    category = "mathematics"
    input_schema = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "maxLength": 1000},
            "expressions": {"type": "array", "items": {"type": "string", "maxLength": 1000}},
        },
    }
    output_schema = {
        "type": "object",
        "properties": {"results": {"type": "array"}, "count": {"type": "integer"}},
    }
    timeout_s = 5.0
    tags = ("math", "deterministic")

    def run(self, inputs: dict[str, Any], context: SkillContext) -> dict[str, Any]:
        expressions = inputs.get("expressions") or ([inputs["expression"]] if inputs.get("expression") else [])
        if not expressions:
            raise SkillExecutionError("Provide 'expression' or 'expressions'.")
        results = []
        for expression in expressions:
            try:
                results.append(evaluate(expression).to_dict())
            except CalculationError as exc:
                raise SkillExecutionError(f"Cannot evaluate {expression!r}: {exc}") from exc
        return {
            "count": len(results),
            "results": results,
            "last": results[-1]["text"],
        }


# ===========================================================================
# 2. structured reasoning
# ===========================================================================
class ReasoningSkill(Skill):
    skill_id = "skill.reasoning"
    name = "Structured reasoning"
    description = (
        "Solve structured problems deterministically: exact linear systems, boolean satisfiability, "
        "numeric sequence extrapolation, audited calculation chains and weighted option comparison "
        "with sensitivity analysis."
    )
    category = "reasoning"
    input_schema = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["linear_system", "boolean_sat", "sequence", "evaluate_steps", "compare_options"],
            },
            "matrix": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "vector": {"type": "array", "items": {"type": "number"}},
            "clauses": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
            "variables": {"type": "integer", "minimum": 1, "maximum": 16},
            "values": {"type": "array", "items": {"type": "number"}},
            "predict": {"type": "integer", "minimum": 1, "maximum": 20},
            "steps": {"type": "array", "items": {"type": "object"}},
            "options": {"type": "array", "items": {"type": "object"}},
            "weights": {"type": "object"},
        },
        "required": ["mode"],
    }
    output_schema = {"type": "object", "properties": {"mode": {"type": "string", "solved": {"type": "boolean"}}}}
    timeout_s = 20.0
    tags = ("reasoning", "deterministic")

    def run(self, inputs: dict[str, Any], context: SkillContext) -> dict[str, Any]:
        mode = inputs["mode"]
        if mode == "linear_system":
            return _solve_linear_system(inputs, context)
        if mode == "boolean_sat":
            return _boolean_sat(inputs, context)
        if mode == "sequence":
            return _sequence(inputs, context)
        if mode == "evaluate_steps":
            return _evaluate_steps(inputs, context)
        if mode == "compare_options":
            return _compare_options(inputs, context)
        raise SkillExecutionError(f"Unsupported reasoning mode '{mode}'.")


def _solve_linear_system(inputs: dict[str, Any], context: SkillContext) -> dict[str, Any]:
    matrix = inputs.get("matrix")
    vector = inputs.get("vector")
    if not matrix or not vector:
        raise SkillExecutionError("linear_system requires 'matrix' (list of rows) and 'vector'.")
    if len(matrix) != len(vector):
        raise SkillExecutionError("Matrix rows must match the length of 'vector'.")
    rows = [[Fraction(str(value)) for value in row] for row in matrix]
    rhs = [Fraction(str(value)) for value in vector]
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise SkillExecutionError("All matrix rows must have the same number of columns.")

    steps: list[str] = []
    row = 0
    pivots: list[int] = []
    for col in range(width):
        pivot = next((index for index in range(row, len(rows)) if rows[index][col] != 0), None)
        if pivot is None:
            continue
        if pivot != row:
            rows[row], rows[pivot] = rows[pivot], rows[row]
            rhs[row], rhs[pivot] = rhs[pivot], rhs[row]
            steps.append(f"swap R{row + 1} <-> R{pivot + 1}")
        factor = rows[row][col]
        rows[row] = [value / factor for value in rows[row]]
        rhs[row] /= factor
        steps.append(f"R{row + 1} /= {_fmt(factor)} (pivot column {col + 1})")
        for other in range(len(rows)):
            if other == row or rows[other][col] == 0:
                continue
            multiplier = rows[other][col]
            rows[other] = [value - multiplier * base for value, base in zip(rows[other], rows[row])]
            rhs[other] -= multiplier * rhs[row]
            steps.append(f"R{other + 1} -= {_fmt(multiplier)} * R{row + 1}")
        pivots.append(col)
        row += 1
        if row == len(rows):
            break

    inconsistent = any(all(value == 0 for value in line) and rhs[index] != 0 for index, line in enumerate(rows))
    rank = len(pivots)
    unique = (not inconsistent) and rank == width and rank == len(rows)
    solution = None
    if unique:
        # After Gauss-Jordan elimination each pivot row reads [0..1..0 | rhs], so the
        # variable's value is the right-hand side, not the pivot cell itself.
        solution = {f"x{col + 1}": _fmt(rhs[index]) for index, col in enumerate(pivots)}
    return {
        "mode": "linear_system",
        "solved": unique,
        "consistent": not inconsistent,
        "unique_solution": unique,
        "rank": rank,
        "variables": width,
        "equations": len(rows),
        "solution": solution,
        "steps": steps,
        "summary": (
            "Unique exact solution found." if unique else
            "System is inconsistent (no solution)." if inconsistent else
            "System is underdetermined (infinitely many solutions)."
        ),
    }


def _boolean_sat(inputs: dict[str, Any], context: SkillContext) -> dict[str, Any]:
    clauses = inputs.get("clauses")
    if not clauses:
        raise SkillExecutionError("boolean_sat requires 'clauses', e.g. [[1, -2], [2, 3]] where 1 = x1 true.")
    variables = int(inputs.get("variables") or max(abs(literal) for clause in clauses for literal in clause))
    if variables > 16:
        raise SkillExecutionError("boolean_sat supports at most 16 variables (brute-force search).")

    def clause_satisfied(assignment: Sequence[bool], clause: Sequence[int]) -> bool:
        for literal in clause:
            index = abs(literal) - 1
            if index >= variables:
                continue
            value = assignment[index]
            if (literal > 0 and value) or (literal < 0 and not value):
                return True
        return False

    models: list[dict[str, bool]] = []
    for combination in itertools.product([False, True], repeat=variables):
        if all(clause_satisfied(combination, clause) for clause in clauses):
            models.append({f"x{index + 1}": combination[index] for index in range(variables)})
            if len(models) >= 25:
                break
    return {
        "mode": "boolean_sat",
        "solved": bool(models),
        "satisfiable": bool(models),
        "variables": variables,
        "clauses": len(clauses),
        "model": models[0] if models else None,
        "model_count_found": len(models),
        "models": models[:5],
        "summary": (
            f"Satisfiable; {len(models)} solution(s) found (search capped at 25)."
            if models
            else "Unsatisfiable: no assignment satisfies all clauses."
        ),
    }


def _sequence(inputs: dict[str, Any], context: SkillContext) -> dict[str, Any]:
    values = inputs.get("values")
    if not values or len(values) < 3:
        raise SkillExecutionError("sequence requires at least three numbers in 'values'.")
    numbers = [float(value) for value in values]
    predict = int(inputs.get("predict") or 3)

    kind = "unknown"
    detail: dict[str, Any] = {}
    next_values: list[float] = []

    diffs = [numbers[index + 1] - numbers[index] for index in range(len(numbers) - 1)]
    ratios = [
        numbers[index + 1] / numbers[index]
        for index in range(len(numbers) - 1)
        if numbers[index] != 0
    ]
    if len(set(round(diff, 9) for diff in diffs)) == 1:
        kind = "arithmetic"
        step = diffs[0]
        detail = {"difference": step}
        next_values = [numbers[-1] + step * (index + 1) for index in range(predict)]
    elif len(ratios) == len(diffs) and len(set(round(ratio, 9) for ratio in ratios)) == 1:
        kind = "geometric"
        ratio = ratios[0]
        detail = {"ratio": ratio}
        next_values = [numbers[-1] * ratio ** (index + 1) for index in range(predict)]
    elif _is_fibonacci_like(numbers):
        kind = "fibonacci_like"
        a, b = numbers[-2], numbers[-1]
        for _ in range(predict):
            a, b = b, a + b
            next_values.append(b)
        detail = {"rule": "a(n) = a(n-1) + a(n-2)"}
    else:
        constant = _polynomial_difference_order(numbers)
        if constant is not None:
            kind = f"polynomial_degree_{constant}"
            working = list(numbers)
            for _ in range(predict):
                working.append(
                    _extend_by_differences(working, constant)
                )
            next_values = working[len(numbers):]
            detail = {"degree": constant, "finest_constant_difference": None}
    return {
        "mode": "sequence",
        "solved": kind != "unknown",
        "kind": kind,
        "detail": detail,
        "input": numbers,
        "next_values": next_values,
        "summary": (
            f"Detected {kind}; predicted {len(next_values)} next value(s)."
            if kind != "unknown"
            else "No arithmetic, geometric, Fibonacci-like or polynomial pattern was detected."
        ),
    }


def _is_fibonacci_like(numbers: Sequence[float]) -> bool:
    return all(abs(numbers[index] - (numbers[index - 1] + numbers[index - 2])) < 1e-9 for index in range(2, len(numbers)))


def _polynomial_difference_order(numbers: Sequence[float], max_order: int = 5) -> int | None:
    working = list(numbers)
    for order in range(1, max_order + 1):
        working = [working[index + 1] - working[index] for index in range(len(working) - 1)]
        if len(working) >= 2 and len(set(round(value, 9) for value in working)) == 1:
            return order
    return None


def _extend_by_differences(numbers: list[float], order: int) -> float:
    """Extend a sequence by one term using its finite-difference table."""

    table: list[list[float]] = [list(numbers)]
    for _ in range(order):
        previous = table[-1]
        table.append([previous[index + 1] - previous[index] for index in range(len(previous) - 1)])
    table[-1].append(table[-1][-1] if table[-1] else 0.0)
    for level in range(len(table) - 2, -1, -1):
        table[level].append(table[level][-1] + table[level + 1][-1])
    return table[0][-1]


def _evaluate_steps(inputs: dict[str, Any], context: SkillContext) -> dict[str, Any]:
    steps = inputs.get("steps")
    if not steps:
        raise SkillExecutionError("evaluate_steps requires 'steps': [{name, description, expression}].")
    values: dict[str, float] = {}
    results: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        expression = str(step.get("expression") or "").strip()
        if not expression:
            raise SkillExecutionError(f"Step {index} has no 'expression'.")
        substituted = _substitute_names(expression, values)
        try:
            result = evaluate(substituted)
        except CalculationError as exc:
            raise SkillExecutionError(f"Step {index} ({expression!r}) failed: {exc}") from exc
        name = str(step.get("name") or f"step{index + 1}")
        values[name] = result.approximate
        results.append(
            {
                "step": index + 1,
                "name": name,
                "description": step.get("description") or "",
                "expression": expression,
                "resolved_expression": substituted,
                "value": result.text,
                "approximate": result.approximate,
            }
        )
    return {
        "mode": "evaluate_steps",
        "solved": True,
        "steps": results,
        "variables": values,
        "final_value": results[-1]["value"] if results else None,
        "summary": f"Evaluated {len(results)} step(s) with exact arithmetic.",
    }


def _substitute_names(expression: str, values: dict[str, float]) -> str:
    resolved = expression
    for name, value in values.items():
        resolved = re.sub(rf"\b{re.escape(name)}\b", f"({value!r})", resolved)
    return resolved


def _compare_options(inputs: dict[str, Any], context: SkillContext) -> dict[str, Any]:
    options = inputs.get("options")
    weights = inputs.get("weights") or {}
    if not options:
        raise SkillExecutionError("compare_options requires 'options': [{name, values: {criterion: number}}].")
    criteria = sorted({key for option in options for key in (option.get("values") or {})})
    if not criteria:
        raise SkillExecutionError("Each option needs a 'values' object of criterion -> number.")
    for criterion in criteria:
        weights.setdefault(criterion, 1.0)

    total_weight = sum(float(weights[criterion]) for criterion in criteria) or 1.0

    def score(option: dict[str, Any], weight_map: dict[str, float]) -> float:
        values = option.get("values") or {}
        return sum(float(values.get(criterion, 0.0)) * float(weight_map.get(criterion, 0.0)) for criterion in criteria) / total_weight

    ranking = sorted(
        ({"name": option.get("name") or f"option{index + 1}", "score": round(score(option, weights), 4)} for index, option in enumerate(options)),
        key=lambda item: -item["score"],
    )

    sensitivity: list[dict[str, Any]] = []
    for criterion in criteria:
        for delta in (0.9, 1.1):
            perturbed = dict(weights)
            perturbed[criterion] = float(weights[criterion]) * delta
            perturbed_ranking = sorted(
                (
                    {"name": option.get("name") or f"option{index + 1}", "score": round(score(option, perturbed), 4)}
                    for index, option in enumerate(options)
                ),
                key=lambda item: -item["score"],
            )
            sensitivity.append(
                {
                    "criterion": criterion,
                    "weight_delta": f"{'+10%' if delta > 1 else '-10%'}",
                    "order": [item["name"] for item in perturbed_ranking],
                    "order_changed": [item["name"] for item in perturbed_ranking] != [item["name"] for item in ranking],
                }
            )
    return {
        "mode": "compare_options",
        "solved": True,
        "criteria": criteria,
        "weights": weights,
        "ranking": ranking,
        "winner": ranking[0]["name"] if ranking else None,
        "sensitivity": sensitivity,
        "summary": (
            f"Ranked {len(ranking)} option(s); '{ranking[0]['name']}' wins with score {ranking[0]['score']}."
            if ranking
            else "No options to compare."
        ),
    }


def _fmt(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


# ===========================================================================
# 3. data analysis
# ===========================================================================
class DataAnalysisSkill(Skill):
    skill_id = "skill.data_analysis"
    name = "Data analysis"
    description = (
        "Analyse real tabular data (CSV / JSON / JSONL text or a sandboxed file path): describe columns, "
        "quantiles, value counts, group-by aggregation, Pearson correlation and least-squares trend."
    )
    category = "data"
    required_tools = ("file.read",)
    input_schema = {
        "type": "object",
        "properties": {
            "data": {"type": ["string", "array"], "description": "CSV/JSON text, a JSON array of objects, or omit to use 'path'."},
            "path": {"type": "string", "description": "Path inside the sandbox root."},
            "format": {"type": "string", "enum": ["auto", "csv", "json", "jsonl", "tsv"]},
            "operations": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["describe", "quantiles", "value_counts", "group_by", "correlate", "trend", "filter"],
                },
            },
            "percentiles": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 100}},
            "group_by": {"type": "string"},
            "aggregate": {"type": "string", "enum": ["mean", "sum", "count", "min", "max", "median", "stdev"]},
            "value_column": {"type": "string"},
            "x_column": {"type": "string"},
            "y_column": {"type": "string"},
            "column": {"type": "string"},
            "operator": {"type": "string", "enum": ["eq", "ne", "gt", "gte", "lt", "lte", "contains"]},
            "value": {},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        },
    }
    output_schema = {
        "type": "object",
        "properties": {"rows": {"type": "integer"}, "columns": {"type": "array"}},
    }
    timeout_s = 30.0
    tags = ("data", "statistics", "deterministic")

    def run(self, inputs: dict[str, Any], context: SkillContext) -> dict[str, Any]:
        records, source = _load_records(inputs, context)
        if not records:
            raise SkillExecutionError("No data rows were found to analyse.")
        columns = sorted({key for row in records for key in row})
        types = {column: _infer_type([row.get(column) for row in records]) for column in columns}
        operations = inputs.get("operations") or ["describe"]
        result: dict[str, Any] = {
            "source": source,
            "rows": len(records),
            "columns": columns,
            "column_types": types,
            "operations": operations,
        }

        if "describe" in operations:
            result["describe"] = {column: _describe_column([row.get(column) for row in records], types[column]) for column in columns}
        if "quantiles" in operations:
            percentiles = inputs.get("percentiles") or [25, 50, 75, 90, 99]
            result["quantiles"] = {
                column: _quantiles([row.get(column) for row in records], percentiles)
                for column in columns
                if types[column] == "number"
            }
        if "value_counts" in operations:
            limit = int(inputs.get("limit", 10))
            targets = [inputs["column"]] if inputs.get("column") else [column for column in columns if types[column] != "number"]
            result["value_counts"] = {
                column: _value_counts([row.get(column) for row in records], limit) for column in targets if column in columns
            }
        if "group_by" in operations:
            group_column = inputs.get("group_by")
            if not group_column:
                raise SkillExecutionError("group_by operation requires 'group_by'.")
            value_column = inputs.get("value_column")
            aggregate = inputs.get("aggregate") or "mean"
            result["group_by"] = _group_by(records, group_column, value_column, aggregate)
        if "correlate" in operations:
            numeric = [column for column in columns if types[column] == "number"]
            pairs = []
            for left, right in itertools.combinations(numeric, 2):
                correlation = _pearson(
                    [row.get(left) for row in records],
                    [row.get(right) for row in records],
                )
                if correlation is not None:
                    pairs.append({"left": left, "right": right, "pearson_r": round(correlation, 6), "abs_r": round(abs(correlation), 6)})
            pairs.sort(key=lambda item: -item["abs_r"])
            result["correlations"] = pairs[: int(inputs.get("limit", 20))]
        if "trend" in operations:
            y_column = inputs.get("y_column") or inputs.get("value_column")
            if not y_column:
                raise SkillExecutionError("trend operation requires 'y_column'.")
            x_column = inputs.get("x_column")
            result["trend"] = _trend(records, x_column, y_column)
        if "filter" in operations:
            result["filter"] = _filter(records, inputs, int(inputs.get("limit", 20)))
        return result


def _load_records(inputs: dict[str, Any], context: SkillContext) -> tuple[list[dict[str, Any]], str]:
    data = inputs.get("data")
    path = inputs.get("path")
    fmt = inputs.get("format") or "auto"
    text: str
    source: str
    if data is None and path:
        output = context.tool_output("file.read", {"path": path, "limit": 20000})
        text = output.get("content", "")
        source = f"file:{path}"
        if fmt == "auto":
            fmt = Path(path).suffix.lstrip(".").lower()
    elif isinstance(data, list):
        return [row if isinstance(row, dict) else {"value": row} for row in data], "inline:json"
    elif isinstance(data, str):
        text = data
        source = "inline:text"
    else:
        raise SkillExecutionError("Provide 'data' (CSV/JSON text or array) or 'path'.")
    return _parse_text(text, fmt, source), source


def _parse_text(text: str, fmt: str, source: str) -> list[dict[str, Any]]:
    if fmt in {"auto", ""}:
        stripped = text.lstrip()
        first_line = text.splitlines()[0] if text.splitlines() else ""
        if stripped.startswith("[") or stripped.startswith("{"):
            fmt = "json"
        elif "\t" in first_line:
            fmt = "tsv"
        else:
            fmt = "csv"
    if fmt == "json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            if isinstance(payload.get("data"), list):
                payload = payload["data"]
            else:
                payload = [payload]
        return [row if isinstance(row, dict) else {"value": row} for row in payload]
    if fmt == "jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if fmt in {"csv", "tsv"}:
        import csv
        import io

        delimiter = "\t" if fmt == "tsv" else ","
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        return [dict(row) for row in reader]
    raise SkillExecutionError(f"Unsupported data format '{fmt}'.")


def _infer_type(values: Iterable[Any]) -> str:
    seen = [value for value in values if value not in (None, "")]
    if not seen:
        return "empty"
    if all(_is_number(value) for value in seen):
        return "number"
    if all(str(value).lower() in {"true", "false"} for value in seen):
        return "boolean"
    return "text"


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _numbers(values: Iterable[Any]) -> list[float]:
    numbers = []
    for value in values:
        if value is None or value == "":
            continue
        try:
            numbers.append(float(value))
        except (TypeError, ValueError):
            continue
    return numbers


def _describe_column(raw: list[Any], kind: str) -> dict[str, Any]:
    present = [value for value in raw if value not in (None, "")]
    description: dict[str, Any] = {
        "type": kind,
        "count": len(present),
        "missing": len(raw) - len(present),
        "unique": len({str(value) for value in present}),
    }
    if kind == "number":
        numbers = _numbers(present)
        if numbers:
            description.update(
                {
                    "min": min(numbers),
                    "max": max(numbers),
                    "mean": round(statistics.fmean(numbers), 6),
                    "median": round(statistics.median(numbers), 6),
                    "stdev": round(statistics.stdev(numbers), 6) if len(numbers) > 1 else 0.0,
                    "sum": round(sum(numbers), 6),
                }
            )
    else:
        counts = _value_counts(present, 5)
        description["most_common"] = counts
    return description


def _quantiles(raw: list[Any], percentiles: Sequence[float]) -> dict[str, Any]:
    numbers = sorted(_numbers(raw))
    if not numbers:
        return {}
    result: dict[str, Any] = {"min": numbers[0], "max": numbers[-1], "count": len(numbers)}
    for percentile in percentiles:
        result[f"p{percentile:g}"] = round(_percentile(numbers, float(percentile)), 6)
    return result


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * (percentile / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _value_counts(raw: list[Any], limit: int) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for value in raw:
        if value in (None, ""):
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return [{"value": value, "count": count} for value, count in ranked]


def _group_by(records: list[dict[str, Any]], group_column: str, value_column: str | None, aggregate: str) -> list[dict[str, Any]]:
    groups: dict[str, list[Any]] = {}
    for row in records:
        key = str(row.get(group_column, ""))
        groups.setdefault(key, []).append(row.get(value_column) if value_column else 1)
    output: list[dict[str, Any]] = []
    for key, values in sorted(groups.items()):
        entry: dict[str, Any] = {"group": key, "count": len(values)}
        numbers = _numbers(values)
        if aggregate == "count":
            entry["value"] = len(values)
        elif aggregate == "sum":
            entry["value"] = round(sum(numbers), 6)
        elif aggregate == "min":
            entry["value"] = min(numbers) if numbers else None
        elif aggregate == "max":
            entry["value"] = max(numbers) if numbers else None
        elif aggregate == "median":
            entry["value"] = round(statistics.median(numbers), 6) if numbers else None
        elif aggregate == "stdev":
            entry["value"] = round(statistics.stdev(numbers), 6) if len(numbers) > 1 else 0.0
        else:
            entry["value"] = round(statistics.fmean(numbers), 6) if numbers else None
        entry["aggregate"] = aggregate
        output.append(entry)
    return output


def _pearson(left: list[Any], right: list[Any]) -> float | None:
    xs: list[float] = []
    ys: list[float] = []
    for x_raw, y_raw in zip(left, right):
        if _is_number(x_raw) and _is_number(y_raw):
            xs.append(float(x_raw))
            ys.append(float(y_raw))
    if len(xs) < 3:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    var_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x * var_y)


def _trend(records: list[dict[str, Any]], x_column: str | None, y_column: str) -> dict[str, Any]:
    ys: list[float] = []
    xs: list[float] = []
    for index, row in enumerate(records):
        raw_y = row.get(y_column)
        if not _is_number(raw_y):
            continue
        if x_column:
            raw_x = row.get(x_column)
            if not _is_number(raw_x):
                continue
            xs.append(float(raw_x))
        else:
            xs.append(float(index))
        ys.append(float(raw_y))
    if len(xs) < 2:
        raise SkillExecutionError("trend requires at least two numeric points.")
    n = len(xs)
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        raise SkillExecutionError("trend requires variation in the x values.")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    intercept = mean_y - slope * mean_x
    predicted = [slope * x + intercept for x in xs]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, predicted))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r_squared = 1 - ss_res / ss_tot if ss_tot else 1.0
    return {
        "x": x_column or "row_index",
        "y": y_column,
        "points": n,
        "slope": round(slope, 8),
        "intercept": round(intercept, 8),
        "r_squared": round(r_squared, 8),
        "direction": "increasing" if slope > 0 else ("decreasing" if slope < 0 else "flat"),
        "equation": f"{y_column} = {round(slope, 6)} * x + {round(intercept, 6)}",
    }


def _filter(records: list[dict[str, Any]], inputs: dict[str, Any], limit: int) -> dict[str, Any]:
    column = inputs.get("column")
    operator = inputs.get("operator") or "eq"
    value = inputs.get("value")
    if not column:
        raise SkillExecutionError("filter operation requires 'column'.")
    matched: list[dict[str, Any]] = []
    for row in records:
        raw = row.get(column)
        if raw is None:
            continue
        if operator == "eq" and str(raw) == str(value):
            matched.append(row)
        elif operator == "ne" and str(raw) != str(value):
            matched.append(row)
        elif operator == "contains" and str(value).lower() in str(raw).lower():
            matched.append(row)
        elif operator in {"gt", "gte", "lt", "lte"} and _is_number(raw) and _is_number(value):
            left, right = float(raw), float(value)
            if {
                "gt": left > right,
                "gte": left >= right,
                "lt": left < right,
                "lte": left <= right,
            }[operator]:
                matched.append(row)
    return {"column": column, "operator": operator, "value": value, "matches": len(matched), "rows": matched[:limit]}


def build_skills() -> list[Skill]:
    return [CalculatorSkill(), ReasoningSkill(), DataAnalysisSkill()]
