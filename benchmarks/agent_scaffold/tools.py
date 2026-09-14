"""Fixed tool environment for agent-scaffold evaluation.

Copied into the candidate subprocess. Must not import ``tasks`` (held-out
answers live there and are not on the child's filesystem).
"""

from __future__ import annotations

from typing import Any, Callable

# Entity facts used by the lookup tool.
KB: dict[str, dict[str, Any]] = {
    "paris": {"country": "france", "population_m": 2.1, "river": "seine"},
    "france": {"capital": "paris", "currency": "euro", "neighbor": "germany"},
    "berlin": {"country": "germany", "population_m": 3.6, "river": "spree"},
    "germany": {"capital": "berlin", "currency": "euro", "neighbor": "france"},
    "tokyo": {"country": "japan", "population_m": 14.0, "river": "sumida"},
    "japan": {"capital": "tokyo", "currency": "yen", "neighbor": "korea"},
    "cairo": {"country": "egypt", "population_m": 10.0, "river": "nile"},
    "egypt": {"capital": "cairo", "currency": "pound", "neighbor": "libya"},
    "ottawa": {"country": "canada", "population_m": 1.0, "river": "ottawa"},
    "canada": {"capital": "ottawa", "currency": "cad", "neighbor": "usa"},
}

# Keyword → entity for the search tool.
SEARCH_INDEX: dict[str, str] = {
    "paris": "paris",
    "france": "france",
    "french capital": "paris",
    "berlin": "berlin",
    "germany": "germany",
    "tokyo": "tokyo",
    "japan": "japan",
    "cairo": "cairo",
    "egypt": "egypt",
    "ottawa": "canada",
    "canada": "canada",
    "seine": "paris",
    "spree": "berlin",
    "nile": "cairo",
    "yen": "japan",
    "euro france": "france",
}


class ToolBudget:
    """Tracks tool-call counts; scaffolds should stop when exhausted."""

    def __init__(self, max_calls: int = 12):
        self.max_calls = max_calls
        self.calls = 0
        self.trace: list[tuple[str, str, Any]] = []

    @property
    def remaining(self) -> int:
        return max(0, self.max_calls - self.calls)

    def record(self, name: str, arg: str, result: Any) -> None:
        self.calls += 1
        self.trace.append((name, arg, result))


def make_tools(budget: ToolBudget) -> dict[str, Callable[..., Any]]:
    """Return the tool dict passed into ``run_agent(question, tools)``."""

    def search(query: str) -> str:
        if budget.remaining <= 0:
            return "ERROR: tool budget exhausted"
        q = str(query).strip().lower()
        hit = SEARCH_INDEX.get(q)
        if hit is None:
            # Soft match: any key contained in query or vice versa.
            for key, entity in SEARCH_INDEX.items():
                if key in q or q in key:
                    hit = entity
                    break
        result = hit if hit is not None else "NOT_FOUND"
        budget.record("search", q, result)
        return result

    def lookup(entity: str, field: str) -> str:
        if budget.remaining <= 0:
            return "ERROR: tool budget exhausted"
        e = str(entity).strip().lower()
        f = str(field).strip().lower()
        row = KB.get(e)
        if row is None:
            result: Any = "UNKNOWN_ENTITY"
        else:
            result = row.get(f, "UNKNOWN_FIELD")
        budget.record("lookup", f"{e}.{f}", result)
        return str(result)

    def calculate(expression: str) -> str:
        if budget.remaining <= 0:
            return "ERROR: tool budget exhausted"
        expr = str(expression).strip()
        # Extremely small safe calculator: digits, + - * / . ( ) and spaces.
        allowed = set("0123456789+-*/(). ")
        if not expr or any(ch not in allowed for ch in expr):
            result = "INVALID_EXPR"
        else:
            try:
                value = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 — sandboxed literals only
                result = str(float(value))
            except Exception:
                result = "CALC_ERROR"
        budget.record("calculate", expr, result)
        return result

    return {
        "search": search,
        "lookup": lookup,
        "calculate": calculate,
        # Scaffolds may inspect remaining budget.
        "budget_remaining": lambda: budget.remaining,
    }
