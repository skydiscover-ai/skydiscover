"""Baseline agent scaffold for multi-hop tool use.

Evolve ``run_agent`` — the harness itself (tool policy, memory, retries,
stopping rules). The evaluator provides fixed tools over a synthetic KB.
"""

from __future__ import annotations

from typing import Any, Callable


# EVOLVE-BLOCK-START
def run_agent(question: str, tools: dict[str, Callable[..., Any]]) -> str:
    """Answer ``question`` using only the provided tools.

    Available tools:
      - tools["search"](query: str) -> str          # entity id or NOT_FOUND
      - tools["lookup"](entity: str, field: str) -> str
      - tools["calculate"](expression: str) -> str  # float as string
      - tools["budget_remaining"]() -> int

    Return a short final answer string (entity name, number, or currency code).
    Keep tool use under the budget — wasted calls hurt the efficiency score.
    """
    q = question.lower()
    memory: dict[str, str] = {}

    def remember(key: str, value: str) -> None:
        if value and value not in {"NOT_FOUND", "UNKNOWN_ENTITY", "UNKNOWN_FIELD", "INVALID_EXPR"}:
            memory[key] = value

    # Naive keyword entity extraction + one or two lookups.
    seeds = [
        "france",
        "paris",
        "germany",
        "berlin",
        "japan",
        "tokyo",
        "egypt",
        "cairo",
        "canada",
        "ottawa",
    ]
    entity = None
    for seed in seeds:
        if seed in q:
            hit = tools["search"](seed)
            remember("entity", hit)
            entity = hit
            break
    if entity is None:
        # Fall back to first search on a coarse paraphrase of the question.
        entity = tools["search"](q.split("?")[0][-40:])
        remember("entity", entity)

    if "capital of" in q and entity:
        ans = tools["lookup"](entity, "capital")
        remember("answer", ans)
        return ans

    if "country is" in q or "country whose capital" in q:
        if "capital is" in q or "capital of" in q:
            # e.g. currency of country whose capital is Paris
            pass
        ans = tools["lookup"](entity, "country") if entity else "unknown"
        remember("answer", ans)
        if "country is" in q:
            return ans

    if "currency" in q:
        # Prefer country entity.
        country = memory.get("entity") or entity
        if country in {"paris", "berlin", "tokyo", "cairo", "ottawa"}:
            country = tools["lookup"](country, "country")
            remember("country", country)
        ans = tools["lookup"](country, "currency")
        return ans

    if "river" in q:
        place = memory.get("entity") or entity
        if "capital of" in q and place:
            place = tools["lookup"](place, "capital")
        ans = tools["lookup"](place, "river")
        return ans

    if "population" in q:
        place = memory.get("entity") or entity
        if "capital of" in q and place:
            place = tools["lookup"](place, "capital")
        ans = tools["lookup"](place, "population_m")
        return ans

    if "neighbor" in q:
        country = memory.get("entity") or entity
        if country in {"paris", "berlin", "tokyo", "cairo", "ottawa"}:
            country = tools["lookup"](country, "country")
        ans = tools["lookup"](country, "neighbor")
        return ans

    if "compute" in q or "+" in q:
        # Pull two population numbers if mentioned.
        nums = []
        for city in ("berlin", "paris", "tokyo", "cairo", "ottawa"):
            if city in q:
                nums.append(tools["lookup"](city, "population_m"))
        if len(nums) >= 2:
            return tools["calculate"](f"{nums[0]}+{nums[1]}")
        return tools["calculate"]("0")

    return memory.get("answer") or (entity or "unknown")


# EVOLVE-BLOCK-END
