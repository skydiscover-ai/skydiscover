"""The decision log (spec/decisions.py): the agent's defaults, the user's answers, and the
rule that the user's answer is never overwritten. answers.json is exported from the log after every change.
"""

import json

import pytest

from skydiscover.synthesize.spec import decisions
from skydiscover.synthesize.spec.findings import Finding, Findings
from skydiscover.synthesize.spec.paths import Domain, Run

QUESTIONS = {
    "questions": [
        {
            "id": "durability",
            "axis": "durability",
            "q": "Must writes survive a crash?",
            "options": ["no", "fsync per write"],
        },
        {
            "id": "eviction",
            "axis": "eviction",
            "q": "What is evicted first when memory is full?",
            "options": ["LRU", "LFU"],
            "default": "LFU",
        },
        {"id": "open-ended", "axis": "open-ended", "q": "Anything else?", "options": []},
    ]
}


def _run(tmp_path, questions=QUESTIONS) -> Run:
    run = Run(tmp_path / "run").create()
    run.task.write_text("---\ndomain: kv store\n---\n", encoding="utf-8")
    run.questions.write_text(json.dumps(questions), encoding="utf-8")
    return run


def _answers(run: Run) -> dict:
    return {d["id"]: d for d in json.loads(run.answers.read_text(encoding="utf-8"))["decisions"]}


def _rows(run: Run) -> dict:
    return {f.title: f for f in Findings(run.decision_log).all()}


# answer


def test_answer_records_the_default_of_every_question_as_the_ai(tmp_path):
    run = _run(tmp_path)
    added = decisions.answer(run)
    # no default declared -> open; declared default -> taken; no options -> open
    assert [f.detail for f in added] == ["", "LFU", ""]
    assert all(f.answered_by == "ai" and f.status == "open" for f in added)
    answers = _answers(run)
    assert "durability" not in answers  # the first option listed is never taken as the answer
    assert answers["eviction"]["chosen"] == "LFU"
    assert "open-ended" not in answers  # an empty answer is not a requirement


def test_answer_is_idempotent_and_never_touches_the_users_answer(tmp_path):
    run = _run(tmp_path)
    decisions.set_answer(run, "durability", "fsync per write")  # the user answered in the chat
    decisions.answer(run)
    decisions.answer(run)
    rows = _rows(run)
    assert len(rows) == 3
    d = rows["Must writes survive a crash?"]
    assert (d.detail, d.answered_by, d.status) == ("fsync per write", "human", "confirmed")
    assert _answers(run)["durability"]["answered_by"] == "human"


def test_a_reworded_question_requires_confirmation(tmp_path):
    run = _run(tmp_path)
    decisions.set_answer(run, "durability", "fsync per write")
    reworded = json.loads(json.dumps(QUESTIONS))
    reworded["questions"][0]["q"] = "Do writes have to survive a crash?"
    run.questions.write_text(json.dumps(reworded), encoding="utf-8")
    decisions.answer(run)
    assert "durability" not in _answers(run)
    assert _rows(run)["Must writes survive a crash?"].detail == "fsync per write"
    decisions.set_answer(run, "durability", "fsync per write")
    assert _answers(run)["durability"]["chosen"] == "fsync per write"


def test_an_answer_whose_question_is_gone_is_not_exported(tmp_path):
    run = _run(tmp_path)
    decisions.answer(run)
    run.questions.write_text(
        json.dumps({"questions": QUESTIONS["questions"][1:]}), encoding="utf-8"
    )
    decisions.export(run)
    assert set(_answers(run)) == {"eviction"}
    assert len(_rows(run)) == 3  # the log keeps history; only the export is trimmed


# set / drop


def test_set_by_row_number_or_question_id_records_the_users_answer(tmp_path):
    run = _run(tmp_path)
    decisions.answer(run)
    decisions.set_answer(run, "1", "fsync per write")
    decisions.set_answer(run, "eviction", "LRU", note="hot keys dominate")
    a = _answers(run)
    assert a["durability"]["chosen"] == "fsync per write"
    assert (a["eviction"]["chosen"], a["eviction"]["rationale"]) == ("LRU", "hot keys dominate")
    assert all(d["answered_by"] == "human" for d in a.values())
    assert all(f.status == "confirmed" for f in _rows(run).values() if f.answered_by == "human")


def test_set_the_same_value_makes_the_default_the_users_answer(tmp_path):
    # "keep the AI's answer" is just setting it yourself: same value, now the user's, never re-asked
    run = _run(tmp_path)
    decisions.answer(run)
    f = decisions.set_answer(run, "1", "no")
    assert (f.detail, f.answered_by, f.status) == ("no", "human", "confirmed")


def test_the_agent_records_its_own_call_as_an_ai_answer(tmp_path):
    run = _run(tmp_path)
    f = decisions.set_answer(
        run, "durability", "fsync per write", by="ai", note="the trace is write-heavy"
    )
    assert (f.answered_by, f.status) == ("ai", "open")  # the agent's answer, not the user's
    assert _answers(run)["durability"]["answered_by"] == "ai"


def test_drop_removes_the_answer_from_the_specification_but_keeps_the_row(tmp_path):
    run = _run(tmp_path)
    decisions.answer(run)
    f = decisions.drop(run, "eviction", by="human")  # the user's own call, said explicitly
    assert (f.answered_by, f.status) == ("human", "waived")
    assert "eviction" not in _answers(run)
    assert "eviction" not in {x.title for x in Findings(run.decision_log).for_spec()}
    assert len(_rows(run)) == 3


def test_an_ai_answer_never_overwrites_the_users_answer(tmp_path):
    run = _run(tmp_path)
    decisions.set_answer(run, "durability", "fsync per write")
    with pytest.raises(ValueError):
        decisions.set_answer(run, "durability", "no", by="ai")
    with pytest.raises(ValueError):
        decisions.drop(run, "1", by="ai")
    assert _answers(run)["durability"]["chosen"] == "fsync per write"


def test_an_unknown_target_is_an_error_not_a_new_row(tmp_path):
    run = _run(tmp_path)
    decisions.answer(run)
    with pytest.raises(ValueError):
        decisions.set_answer(run, "99", "x")
    with pytest.raises(ValueError):
        decisions.drop(run, "no-such-question")
    assert len(_rows(run)) == 3


def test_a_finding_the_auditor_added_can_be_ruled_on_too(tmp_path):
    run = _run(tmp_path)
    Findings(run.decision_log).add(
        Finding(
            id="h1",
            title="hack: stale fill",
            kind="hack",
            detail="regenerates values",
            severity="defect",
        )
    )
    decisions.answer(run)
    f = decisions.drop(run, "1")  # the hack row came first; a bare drop is the agent's own act
    assert (f.id, f.status, f.answered_by) == ("h1", "waived", "ai")
    assert set(_answers(run)) == {
        "eviction"
    }  # non-question rows (and open questions) are never exported


# cli


def test_the_cli_round_trip(tmp_path, capsys):
    run = _run(tmp_path)
    assert decisions.main([str(run.path), "answer"]) == 0
    assert decisions.main([str(run.path), "set", "eviction", "LRU"]) == 0
    assert decisions.main([str(run.path), "set", "1", "no"]) == 0
    assert decisions.main([str(run.path), "drop", "3", "--by", "ai"]) == 0
    assert decisions.main([str(run.path), "list"]) == 0
    out = capsys.readouterr().out
    assert "[you] Must writes survive a crash?" in out and "(your answer)" in out
    assert "[you] What is evicted first when memory is full?" in out
    assert "[AI ] Anything else?" in out and "(set aside)" in out
    assert _answers(run)["eviction"]["chosen"] == "LRU"


# knowledge base


def test_save_keeps_the_users_answers_and_active_hacks_only(tmp_path, monkeypatch):
    monkeypatch.setenv("SKYDISCOVER_HOME", str(tmp_path / "home"))
    run = _run(tmp_path)
    decisions.answer(run)
    decisions.set_answer(run, "durability", "fsync per write")
    Findings(run.decision_log).add(
        Finding(
            id="h1",
            title="hack: stale fill",
            kind="hack",
            detail="regenerates values",
            severity="defect",
        )
    )
    memory = decisions.kept_decisions(run.path)
    assert memory.path == Domain("kv store").decisions
    assert decisions.save(Findings(run.decision_log), memory) == (2, 2)
    assert {f.title for f in memory.all()} == {"Must writes survive a crash?", "hack: stale fill"}
    # idempotent, and a later AI answer can never replace the user's answer in memory
    assert decisions.save(Findings(run.decision_log), memory) == (0, 2)
    qid = decisions.question_id(QUESTIONS["questions"][0])
    memory.upsert(Finding(id=qid, title="x", kind="spec", detail="no", answered_by="ai"))
    kept = next(f for f in memory.all() if f.title == "Must writes survive a crash?")
    assert kept.detail == "fsync per write"


def test_kept_decisions_without_a_domain_fails_closed(tmp_path):
    run = Run(tmp_path / "run").create()
    run.task.write_text("# no front matter\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no domain"):
        decisions.kept_decisions(run.path)
