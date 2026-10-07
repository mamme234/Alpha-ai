"""Safe arithmetic evaluation for AlphaAI.

A real evaluator: the expression is parsed with :mod:`ast`, only whitelisted node
types and functions are allowed, and the result is computed exactly (integers
stay exact, floats are ordinary IEEE-754 doubles). No ``eval`` of arbitrary code,
no fake answers.
"""

from __future__ import annotations

import ast
import math
import operator
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

_CONSTANTS: dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
    "nan": math.nan,
}

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_COMPARE_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "sqrt": math.sqrt,
    "isqrt": math.isqrt,
    "cbrt": getattr(math, "cbrt", lambda x: x ** (1 / 3)),
    "exp": math.exp,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "degrees": math.degrees,
    "radians": math.radians,
    "floor": math.floor,
    "ceil": math.ceil,
    "trunc": math.trunc,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "lcm": math.lcm,
    "perm": math.perm,
    "comb": math.comb,
    "hypot": math.hypot,
    "dist": math.dist,
    "fsum": math.fsum,
    "prod": math.prod,
    "fabs": math.fabs,
    "copysign": math.copysign,
    "remainder": math.remainder,
    "fmod": math.fmod,
    "mean": lambda *values: sum(values) / len(values),
    "median": lambda *values: sorted(values)[len(values) // 2]
    if len(values) % 2
    else (sorted(values)[len(values) // 2 - 1] + sorted(values)[len(values) // 2]) / 2,
    "fraction": Fraction,
}

_ATTRIBUTES = {
    "math": {
        name: getattr(math, name)
        for name in dir(math)
        if not name.startswith("_") and name not in {"prod", "dist"}
    }
}


class CalculationError(ValueError):
    """Raised when an expression is invalid or uses disallowed syntax."""


@dataclass(slots=True)
class CalculationResult:
    """Outcome of a real evaluation."""

    expression: str
    value: Any
    text: str
    exact: str
    approximate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression": self.expression,
            "value": self.value,
            "text": self.text,
            "exact": self.exact,
            "approximate": self.approximate,
        }


def _render(value: Any) -> tuple[str, float]:
    if isinstance(value, bool):
        return ("true" if value else "false"), float(value)
    if isinstance(value, int):
        return str(value), float(value)
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return str(value.numerator), float(value)
        return f"{value.numerator}/{value.denominator}", float(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan", value
        if math.isinf(value):
            return ("inf" if value > 0 else "-inf"), value
        if value.is_integer() and abs(value) < 1e16:
            return str(int(value)), value
        return repr(value), value
    return str(value), float(value)


def evaluate(expression: str) -> CalculationResult:
    """Evaluate ``expression`` exactly and return the real result."""

    if expression is None or not str(expression).strip():
        raise CalculationError("Empty expression.")
    expression = str(expression).strip()
    if len(expression) > 1000:
        raise CalculationError("Expression is too long (limit 1000 characters).")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculationError(f"Cannot parse expression: {exc.msg}") from exc

    value = _eval_node(tree.body)
    text, approximate = _render(value)
    exact = f"{text}" if isinstance(value, int) else f"{value!r}"
    return CalculationResult(
        expression=expression,
        value=value,
        text=text,
        exact=exact,
        approximate=approximate,
    )


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool)):
            return node.value
        raise CalculationError(f"Unsupported literal: {node.value!r}")

    if isinstance(node, ast.BinOp):
        op = _BINARY_OPS.get(type(node.op))
        if op is None:
            raise CalculationError(f"Unsupported operator: {type(node.op).__name__}")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)) and abs(right) > 10_000:
            raise CalculationError("Exponent is too large (limit 10000).")
        try:
            return op(left, right)
        except ZeroDivisionError as exc:
            raise CalculationError("Division by zero.") from exc
        except OverflowError as exc:
            raise CalculationError("Result is too large to compute.") from exc

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise CalculationError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op(_eval_node(node.operand))

    if isinstance(node, ast.Compare):
        left = _eval_node(node.left)
        for op_node, comparator in zip(node.ops, node.comparators):
            op = _COMPARE_OPS.get(type(op_node))
            if op is None:
                raise CalculationError(f"Unsupported comparison: {type(op_node).__name__}")
            right = _eval_node(comparator)
            if not op(left, right):
                return False
            left = right
        return True

    if isinstance(node, ast.BoolOp):
        values = [_eval_node(value) for value in node.values]
        if isinstance(node.op, ast.And):
            return all(values)
        if isinstance(node.op, ast.Or):
            return any(values)
        raise CalculationError("Unsupported boolean operator.")

    if isinstance(node, ast.IfExp):
        return _eval_node(node.body) if _eval_node(node.test) else _eval_node(node.orelse)

    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise CalculationError(f"Unknown name '{node.id}'. Only constants and functions are available.")

    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id in _ATTRIBUTES:
            table = _ATTRIBUTES[node.value.id]
            if node.attr in table:
                return table[node.attr]
            raise CalculationError(f"Unknown attribute '{node.value.id}.{node.attr}'.")
        raise CalculationError("Only math.* attributes are allowed.")

    if isinstance(node, ast.Call):
        func = _eval_node(node.func)
        if not callable(func):
            raise CalculationError("Expression calls something that is not a function.")
        if node.keywords:
            raise CalculationError("Keyword arguments are not supported.")
        args = [_eval_node(arg) for arg in node.args]
        try:
            return func(*args)
        except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
            raise CalculationError(f"{type(exc).__name__}: {exc}") from exc

    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(element) for element in node.elts)

    if isinstance(node, ast.List):
        return [_eval_node(element) for element in node.elts]

    raise CalculationError(f"Disallowed syntax: {type(node).__name__}")


def available_functions() -> list[str]:
    """Names AlphaAI's calculator exposes (used in tool docs and tests)."""

    return sorted(_FUNCTIONS) + sorted(_CONSTANTS)
