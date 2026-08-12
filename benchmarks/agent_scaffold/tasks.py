"""Synthetic multi-hop tool-use tasks for agent-scaffold discovery (#1).

No external APIs: a fixed knowledge base + deterministic task suite so the
evaluator can score scaffolds offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Task:
    task_id: str
    question: str
    answer: str
    # Hints for the README / humans — not passed to the agent.
    notes: str = ""


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


TASKS: list[Task] = [
    Task(
        "t01",
        "What is the capital of France?",
        "paris",
        "single lookup",
    ),
    Task(
        "t02",
        "What country is Berlin in?",
        "germany",
        "single lookup",
    ),
    Task(
        "t03",
        "What is the currency of Japan?",
        "yen",
        "single hop country→currency",
    ),
    Task(
        "t04",
        "What river runs through the capital of Egypt?",
        "nile",
        "two hop: egypt→cairo→river",
    ),
    Task(
        "t05",
        "What is the population in millions of the capital of Germany?",
        "3.6",
        "two hop: germany→berlin→population",
    ),
    Task(
        "t06",
        "What currency is used in the country whose capital is Paris?",
        "euro",
        "two hop: paris→france→currency",
    ),
    Task(
        "t07",
        "Name a neighbor of the country that has capital Tokyo.",
        "korea",
        "two hop: tokyo→japan→neighbor",
    ),
    Task(
        "t08",
        "What river runs through Paris?",
        "seine",
        "single lookup",
    ),
    Task(
        "t09",
        "What is the capital of Canada?",
        "ottawa",
        "single lookup",
    ),
    Task(
        "t10",
        "What is the population in millions of Cairo?",
        "10.0",
        "single lookup",
    ),
    Task(
        "t11",
        "Compute 3.6 + 2.1 (populations of Berlin and Paris).",
        "5.7",
        "needs calculator after lookups or direct calc",
    ),
    Task(
        "t12",
        "What currency does the neighbor of France use? Answer with that neighbor's currency.",
        "euro",
        "france→germany→currency (germany also euro)",
    ),
]


def normalize_answer(text: str) -> str:
    return " ".join(str(text).strip().lower().split())
