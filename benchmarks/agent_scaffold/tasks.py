"""Synthetic multi-hop tool-use tasks for agent-scaffold discovery (#1).

No external APIs: a deterministic task suite so the evaluator can score
scaffolds offline. The lookup KB lives in ``tools.py`` (copied into the
candidate subprocess); this module is parent-only and is not on the
candidate's filesystem.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    task_id: str
    question: str
    answer: str
    # Hints for the README / humans — not passed to the agent.
    notes: str = ""


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
