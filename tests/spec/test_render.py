"""spec.md is a template filled from the cards: fixed sections, first-sentence cells, and a section
left out when its card is absent. No role writes prose into it."""

from __future__ import annotations

import json
from pathlib import Path

from skydiscover.synthesize.spec.build import build_requirements_card
from skydiscover.synthesize.spec.paths import Run
from skydiscover.synthesize.spec.render import first_sentence, label, render_spec

LONG = (
    "Every acknowledged put stays readable, from the spill file if it is no longer resident. "
    "The card then explains this at length, with file:line citations and the arithmetic behind "
    "the choice, none of which belongs on the page."
)


def _run_with_cards(tmp_path: Path, *, environment: bool = True) -> Run:
    run = Run(tmp_path / "scratch" / "toy-cache").create()
    run.task.write_text("---\ndomain: cache\n---\n# Task: a bounded cache\n", encoding="utf-8")
    run.cards.mkdir(parents=True)
    (run.cards / "requirements.json").write_text(
        json.dumps(
            {
                "title": "Bounded in-process cache -- target specification",
                "required_properties": [
                    {
                        "axis": "eviction_policy",
                        "requirement": "When the cache is full and a new key arrives, which entry leaves? Say why.",
                        "value": "the least recently used entry; ties broken by insertion order",
                    },
                    {"axis": "retention", "requirement": "May a put be dropped?", "value": LONG},
                ],
                "operating_point": {
                    "params": {"capacity": {"max": 1024}, "threads": {"min": 4}},
                    "checker": ["tsan"],
                },
            }
        ),
        encoding="utf-8",
    )
    (run.cards / "properties.json").write_text(
        json.dumps(
            [
                {"id": "eviction_policy", "guarantee": "LRU order."},
                {"id": "retention", "guarantee": "Nothing acknowledged is dropped."},
                {
                    "id": "floor:bounded-memory",
                    "guarantee": "Resident bytes stay under capacity. Always.",
                },
                {"id": "add:no-key-scan", "guarantee": "A lookup never scans every key."},
            ]
        ),
        encoding="utf-8",
    )
    (run.cards / "workload.json").write_text(
        json.dumps(
            {
                "source": "the task's own generator, run on this host: " + LONG,
                "scored_configuration": {
                    "_note": "never shown",
                    "command": "python3 bench.py --trace /data/zipf.bin --capacity 1024",
                    "trace_file": "/data/zipf.bin",
                    "capacity": 1024,
                    "requests": 2000000,
                    "operation_mix": "90% get / 10% put",
                    "seeds": [1, 2, 3],
                },
            }
        ),
        encoding="utf-8",
    )
    if environment:
        (run.cards / "environment.json").write_text(
            json.dumps(
                {
                    "machine": {
                        "processor": "8 vCPU Xeon. Two sockets, one used.",
                        "memory": "32 GB",
                    },
                    "bound": {"resource": "Memory bandwidth on the socket. Measured at 40 GB/s."},
                    "ceiling": {"value": 12.5, "unit": "Mreq/s", "derivation": LONG},
                }
            ),
            encoding="utf-8",
        )
    run.tests.mkdir()
    for name in (
        "check_eviction_policy.py",
        "check_retention_spill.py",
        "op_check_floor_bounded_memory.py",
    ):
        (run.tests / name).write_text("# test\n", encoding="utf-8")
    run.decision_log.write_text(
        json.dumps(
            [
                {"id": "a1", "kind": "question", "title": "eviction policy", "status": "confirmed"},
                {
                    "id": "h1",
                    "kind": "hack",
                    "title": "Counts a miss as a hit when the key was seen before. Raises the score 2x.",
                    "status": "confirmed",
                    "test": "synthesis/tests/check_retention_spill.py",
                },
                {"id": "h2", "kind": "hack", "title": "Reads the trace ahead", "status": "waived"},
                {"id": "h3", "kind": "overfit", "title": "Tuned to seed 1", "status": "open"},
            ]
        ),
        encoding="utf-8",
    )
    return run


SCORE = {
    "checkpoint": 3,
    "score": {"hit_rate": 0.81},
    "baselines": [{"name": "LRU", "score": {"hit_rate": 0.75}}],
}


def test_the_page_is_filled_from_the_cards_one_sentence_per_cell(tmp_path):
    run = _run_with_cards(tmp_path)
    page = render_spec(run.path, SCORE)

    assert page.startswith("# Bounded in-process cache\n")  # the card's suffix is cut
    assert "| Score | **0.81 hit_rate** — 1.08× LRU (0.75) |" in page
    assert "| Tests | 3 in `tests/`" in page and "| Checkpoint | 3, the best of the run |" in page
    # a dimension row: label, first sentence of the question, the answer, the test that exists
    assert (
        "| Eviction policy | When the cache is full and a new key arrives, which entry leaves? "
        "| the least recently used entry; ties broken by insertion order | `check_eviction_policy.py` |"
    ) in page
    # a long answer is cut to its first sentence; the test is matched by name, not claimed by a card
    assert (
        "| Retention | May a put be dropped? | Every acknowledged put stays readable, from the spill file if it is no longer resident. | `check_retention_spill.py` |"
        in page
    )
    assert "file:line" not in page
    # properties the questions did not cover come after, with their guarantee
    assert (
        "| Bounded memory | Resident bytes stay under capacity. | `op_check_floor_bounded_memory.py` |"
        in page
    )
    assert "| No key scan | A lookup never scans every key. | — |" in page


def test_workload_shows_settings_not_paths_commands_or_explanations(tmp_path):
    run = _run_with_cards(tmp_path)
    page = render_spec(run.path, SCORE)
    workload = page.split("## Workload", 1)[1].split("## ", 1)[0]

    assert "| Capacity | 1,024 |" in workload and "| Requests | 2,000,000 |" in workload
    assert "| Operation mix | 90% get / 10% put |" in workload
    for hidden in ("/data/zipf.bin", "python3 bench.py", "never shown", "seeds", "generator"):
        assert hidden not in workload


def test_answer_id_survives_build_and_joins_tests_and_properties(tmp_path):
    run = Run(tmp_path / "run").create()
    run.cards.mkdir(parents=True)
    run.tests.mkdir()
    card = build_requirements_card(
        {"title": "Cache"},
        answers={
            "decisions": [
                {
                    "id": "eviction_tiering",
                    "axis": "eviction_policy",
                    "q": "How should eviction work?",
                    "chosen": "Two tiers",
                    "answered_by": "human",
                }
            ]
        },
    )
    run.requirements_card.write_text(json.dumps(card))
    (run.cards / "properties.json").write_text(
        json.dumps(
            [
                {"id": "eviction_tiering", "guarantee": "Use two tiers when selected."},
                {"id": "bounded_memory", "guarantee": "Always respect the memory limit."},
            ]
        )
    )
    (run.tests / "check_eviction_tiering.py").write_text("assert True\n")

    page = render_spec(run.path, {"score": {}})

    assert (
        "| Eviction policy | How should eviction work? | Two tiers | `check_eviction_tiering.py` |"
        in page
    )
    always = page.split("Always required, whatever the answers above:", 1)[1]
    assert "Eviction tiering" not in always
    assert "Use two tiers when selected." not in always
    assert "Bounded memory" in always


def test_environment_is_the_machine_the_bound_the_ceiling_and_the_operating_point(tmp_path):
    run = _run_with_cards(tmp_path)
    page = render_spec(run.path, SCORE)
    env = page.split("## Environment", 1)[1].split("## ", 1)[0]

    assert "| Processor | 8 vCPU Xeon. |" in env
    assert "| Binding resource | Memory bandwidth on the socket. |" in env
    assert "| Ceiling | 12.5 Mreq/s |" in env
    assert "| Tests run at | capacity ≤ 1,024; threads ≥ 4; under tsan |" in env
    assert "derivation" not in env and LONG.split(". ")[1] not in env


def test_hacks_name_the_test_that_catches_each_or_say_open(tmp_path):
    run = _run_with_cards(tmp_path)
    page = render_spec(run.path, SCORE)
    hacks = page.split("## Reward hacks found", 1)[1]

    assert (
        "| Counts a miss as a hit when the key was seen before. | `check_retention_spill.py` |"
        in hacks
    )
    assert "| Reads the trace ahead | no longer possible (waived) |" in hacks
    assert "| Tuned to seed 1 | **open** |" in hacks
    assert "eviction policy" not in hacks  # a question is not a hack


def test_a_section_without_a_card_is_left_out_and_the_page_still_renders(tmp_path):
    run = _run_with_cards(tmp_path, environment=False)
    page = render_spec(run.path, SCORE)
    env = page.split("## Environment", 1)[1].split("## ", 1)[0]
    assert "Tests run at" in env and "Ceiling" not in env  # only the requirement remains

    bare = Run(tmp_path / "scratch" / "bare").create()
    page = render_spec(bare.path, {"checkpoint": 1, "score": {}})
    assert page.startswith("# bare\n") and "## " not in page  # no cards, no tests, no log: a header
    assert "| Checkpoint | 1, the best of the run |" in page


def test_a_proof_run_has_no_test_column(tmp_path):
    run = _run_with_cards(tmp_path)
    for test in run.tests.iterdir():
        test.unlink()
    page = render_spec(run.path, SCORE)
    assert "| Property | Question | Answer |\n" in page and "| Test |" not in page
    assert "| Tests |" not in page


def test_the_checkpoints_table_says_which_iteration_won_and_why_the_others_lost(tmp_path):
    run = _run_with_cards(tmp_path)
    history = [
        {"checkpoint": 1, "score": {"hit_rate": 0.7}, "became_best": True, "tests": 1},
        {
            "checkpoint": 2,
            "score": {"hit_rate": 0.9},
            "became_best": True,
            "tests": 1,
            "fails": ["check_b.py", "check_c.py"],
        },
        {"checkpoint": 3, "score": {"hit_rate": 0.81}, "tests": 2, "fails": None},
    ]
    page = render_spec(run.path, dict(SCORE, checkpoint=3), history)
    table = page.split("## Checkpoints", 1)[1]
    assert "| # | hit_rate | Tests when scored |  |" in table
    assert "| 1 | 0.7 | 1 |  |" in table
    assert "| 2 | 0.9 | 1 | fails `check_b.py` and 1 more |" in table
    assert "| 3 | 0.81 | 2 | score not re-checked |" in table
    assert "## Checkpoints" not in render_spec(run.path, SCORE)  # no history, no table


def test_first_sentence_and_label():
    assert first_sentence("One. Two.") == "One."
    assert first_sentence("No terminator " * 20).endswith("…")
    assert len(first_sentence("x" * 500)) == 140
    assert first_sentence("Is 4.5 GB enough? Yes.") == "Is 4.5 GB enough?"
    assert first_sentence(None) == ""
    assert label("floor:bounded-memory") == "Bounded memory"
    assert label("mem_budget_bytes") == "Mem budget bytes"


def test_score_multiplier_follows_the_metric_direction_and_avoids_scientific_notation(tmp_path):
    run = _run_with_cards(tmp_path)
    # Higher is better: value / reference.
    page = render_spec(
        run.path,
        {"score": {"hit_rate": 0.9}, "baselines": [{"name": "FIFO", "score": {"hit_rate": 0.3}}]},
    )
    assert "**0.9 hit_rate** — 3.00× FIFO (0.3)" in page
    # Lower is better (a latency): reference / value, so halving reads as 2×, not 0.50×.
    page = render_spec(
        run.path,
        {"score": {"p99_ms": 50.0}, "baselines": [{"name": "stock", "score": {"p99_ms": 100.0}}]},
        direction="min",
    )
    assert "**50 p99_ms** — 2.00× stock (100)" in page
    # No ratio against a zero or negative reference; big numbers stay plain digits.
    page = render_spec(
        run.path,
        {"score": {"ops": 2543210.0}, "baselines": [{"name": "stock", "score": {"ops": 0}}]},
    )
    assert "**2,543,210 ops**" in page and "×" not in page.split("| Score |")[1].split("\n")[0]


def test_environment_renders_an_exact_operating_point_and_a_named_checker(tmp_path):
    # The shape build.py documents and real runs write: {"eq": N} params and a string checker.
    run = _run_with_cards(tmp_path)
    card = json.loads((run.cards / "requirements.json").read_text())
    card["operating_point"] = {
        "params": {"capacity": {"eq": 128}, "trace_ops": {"eq": 14300}},
        "checker": "plain",
    }
    (run.cards / "requirements.json").write_text(json.dumps(card), encoding="utf-8")
    page = render_spec(run.path, SCORE)
    assert "## Environment" in page
    # "plain" is an ordinary build and is not worth a word; a sanitizer is.
    assert "| Tests run at | capacity = 128; trace_ops = 14,300 |" in page
    card["operating_point"]["checker"] = ["plain", "asan"]
    (run.cards / "requirements.json").write_text(json.dumps(card), encoding="utf-8")
    assert "capacity = 128; trace_ops = 14,300; under asan |" in render_spec(run.path, SCORE)


def test_files_written_by_the_first_releases_still_render(tmp_path):
    """The first releases wrote one `baseline` (as `{"name", "score"}` or the bare metrics) and
    marked the published history row `best`; those results keep their ratio and their verified line.
    """
    run = _run_with_cards(tmp_path)
    bare = {"checkpoint": 3, "score": {"hit_rate": 0.9}, "baseline": {"hit_rate": 0.3}}
    history = [{"checkpoint": 3, "best": True, "fails": []}]
    page = render_spec(run.path, bare, history)
    assert "**0.9 hit_rate** — 3.00× the baseline (0.3)" in page
    assert "Verified: every test in `tests/` passes" in page
    named = {**bare, "baseline": {"name": "FIFO", "score": {"hit_rate": 0.3}}}
    assert "3.00× FIFO (0.3)" in render_spec(run.path, named, history)
