"""Contract replay: play the lead exactly as SKILL.md, artifacts.md, state.md and verification.md say,
with hand-written minimal artifacts and no model, and check at every step that the runtime does what
the document promises. One run of a small pure-Python cache domain, Phase 1 to `run finish`, then a
second run in the same domain to see the knowledge base reused. Each assertion names the document line
it checks, so a failure reads as "the runtime broke this sentence of the docs".

Runs in a few seconds; needs only python3 (no compiler, no network, no model).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "skydiscover" / "synthesize" / "workflow" / "scripts"
HOOK = REPO / "skydiscover" / "synthesize" / "workflow" / "hooks" / "delivery_check.sh"

REFERENCE = """
    from collections import OrderedDict
    class _C:
        def __init__(self, cap): self.cap, self.d = cap, OrderedDict()
        def get(self, k):
            if k in self.d: self.d.move_to_end(k); return self.d[k]
            return None
        def put(self, k, v):
            self.d[k] = v; self.d.move_to_end(k)
            while len(self.d) > self.cap: self.d.popitem(last=False)
        def size(self): return len(self.d)
    def create_cache(capacity): return _C(capacity)
    """
QUESTIONS = {
    "title": "Requirements",
    "domain": "cache",
    "questions": [
        {
            "id": "eviction",
            "axis": "eviction",
            "q": "What must the cache keep when it is full?",
            "why": "the score",
            "options": [{"value": "the recently used keys"}, {"value": "the frequently used keys"}],
            "default": "the frequently used keys",
            "multi": False,
        },
        {
            "id": "history",
            "axis": "history",
            "q": "May the cache remember keys it has evicted?",
            "why": "memory",
            "options": [{"value": "no"}, {"value": "a bounded number"}],
            "default": None,
            "multi": False,
        },
    ],
}


class Lead:
    """Runs the documented commands the way the lead agent would, from the project directory."""

    def __init__(self, work: Path, home: Path):
        self.work, self.home = work, home
        self.env = {**os.environ, "SKYDISCOVER_HOME": str(home), "PYTHONPATH": str(REPO)}
        self.env.pop("SKYDISCOVER_RUN", None)
        self.env.pop("SKYDISCOVER_IMPL", None)

    def sh(self, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            list(args), cwd=self.work, env=env or self.env, text=True, capture_output=True
        )

    def mod(self, m: str, *args: str) -> subprocess.CompletedProcess:
        return self.sh(sys.executable, "-m", f"skydiscover.synthesize.{m}", *args)

    def script(self, name: str, *args: str) -> subprocess.CompletedProcess:
        return self.sh(sys.executable, str(SCRIPTS / name), *args)


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")


def _jwrite(p: Path, doc) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def _jread(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def _out(r: subprocess.CompletedProcess) -> str:
    return r.stdout + r.stderr


@pytest.fixture(scope="module")
def lead(tmp_path_factory) -> Lead:
    work = tmp_path_factory.mktemp("project")
    home = tmp_path_factory.mktemp("home")
    return Lead(work, home)


@pytest.fixture(scope="module")
def S() -> dict:
    """State the ordered tests below hand to each other (paths the run creates)."""
    return {}


# Phase 1: Initial Specification


def test_step1_create_the_run_and_name_the_domain(lead, S):
    r = lead.mod("spec.paths", "run", "contract-cache")
    assert r.returncode == 0, _out(r)
    run = lead.work / r.stdout.strip()
    assert run == lead.work / ".skydiscover" / "contract-cache", "SKILL: .skydiscover/<slug>"
    assert (run / "README.md").is_file(), "artifacts.md: Run.create writes a README"
    for d in ("specification", "synthesis", "review"):
        assert (run / d).is_dir(), "artifacts.md: one folder per phase"
    _write(
        run / "task.md",
        """
        ---
        domain: cache
        checked_by: tests
        ---
        Build a fast fixed-capacity cache. Score: app hit rate, baseline FIFO.
        Interface: create_cache(capacity) -> get(key), put(key, value), size().
        """,
    )
    r = lead.mod("spec.paths", "domain", "cache")
    domain = Path(r.stdout.strip())
    assert domain.parent == lead.home, "SKILL 1.2: the domain maps to ~/.skydiscover/<domain>/"
    S.update(run=run, spec=run / "specification", syn=run / "synthesis", domain=domain)


def test_step1_the_discovery_artifacts_pass_the_lint_and_run_check(lead, S):
    spec = S["spec"]
    src = spec / "sources" / "toycache"
    _write(src / "lru.py", "class LRU: pass\n")
    for cmd in (
        ["git", "init", "-q", str(src)],
        ["git", "-C", str(src), "add", "."],
        ["git", "-C", str(src), "-c", "user.email=a@b", "-c", "user.name=t", "commit", "-qm", "i"],
    ):
        assert lead.sh(*cmd).returncode == 0
    _jwrite(
        spec / "references" / "toycache" / "spec.json",
        {
            "source": "example/toycache",
            "axes": {
                "eviction": {
                    "property": "which key is evicted",
                    "guarantee": "least recently used",
                    "evidence": "lru.py:1",
                }
            },
        },
    )
    _jwrite(spec / "references" / "toycache" / "verification.json", {"checked": 1, "ok": 1})
    _jwrite(
        spec / "references" / "axes.json",
        {
            "domain": "cache",
            "axes": [{"id": "eviction", "question": "keep what?", "rationale": "score"}],
            "mechanisms": ["lru", "tinylfu", "ghost list"],
        },
    )
    _jwrite(spec / "references" / "skeleton.json", {"shared": [], "domain_specific": {}})
    _jwrite(spec / "references" / "tests.json", {"eviction": []})
    _write(spec / "references" / "tests" / "toycache" / "test_lru.py", "def test(): pass\n")
    _jwrite(spec / "questions.json", QUESTIONS)

    r = lead.mod("spec.requirements", "check", str(spec / "questions.json"))
    assert (
        r.returncode == 0
    ), "spec-builder.md (discovery): behavior-phrased questions pass the lint\n" + _out(r)
    bad = json.loads(json.dumps(QUESTIONS))
    bad["questions"][0]["options"] = [{"value": "lru"}, {"value": "tinylfu"}]
    _jwrite(spec / "bad.json", bad)
    r = lead.mod("spec.requirements", "check", str(spec / "bad.json"))
    assert (
        r.returncode != 0 and "lru" in _out(r).lower()
    ), "requirements.py: the lint forbids the mechanisms axes.json lists\n" + _out(r)
    (spec / "bad.json").unlink()


def test_step2_decisions_defaults_the_users_answer_and_the_log(lead, S):
    run, spec = S["run"], S["spec"]
    assert lead.mod("spec.decisions", str(run), "answer").returncode == 0
    answers = _jread(spec / "answers.json")
    assert [d["id"] for d in answers["decisions"]] == [
        "eviction"
    ], "state.md: 'answer' fills only questions with a declared default; the other stays open"
    assert (
        lead.mod("spec.decisions", str(run), "set", "history", "a bounded number").returncode == 0
    )
    r = lead.mod("spec.decisions", str(run), "set", "history", "no", "--by", "ai")
    assert r.returncode != 0, "state.md: an AI answer never overwrites the user's"
    hist = next(d for d in _jread(spec / "answers.json")["decisions"] if d["id"] == "history")
    assert hist["chosen"] == "a bounded number" and hist["answered_by"] == "human"
    r = lead.mod("spec.decisions", str(run), "list")
    assert "[you]" in r.stdout and "[AI ]" in r.stdout, "SKILL: 'list' shows who decided each"
    assert len(_jread(run / "decision_log.json")) == 2
    assert (run / ".severity_snapshot.json").is_file(), "state.md: the CLI keeps the snapshot"


def test_step3_build_the_cards_and_run_check(lead, S):
    run, spec = S["run"], S["spec"]
    _jwrite(
        spec / "spec.json",
        {
            "title": "Scan-resistant cache",
            "interface": {"create_cache": "create_cache(capacity)"},
            "axes": {"eviction": {"guarantee": "frequent keys survive a scan"}},
            "operating_point": {
                "params": {"capacity": {"eq": 128}, "ops": {"min": 6000}},
                "checker": "plain",
            },
        },
    )
    r = lead.mod(
        "spec.build",
        str(spec / "spec.json"),
        "--answers",
        str(spec / "answers.json"),
        "-d",
        str(spec / "cards"),
    )
    assert r.returncode == 0, _out(r)
    card = _jread(spec / "cards" / "requirements.json")
    assert card["operating_point"]["params"]["capacity"] == {
        "eq": 128
    }, "verification.md: the operating point declared in spec.json reaches the card"
    r = lead.mod("spec.run", "check", str(run))
    assert r.returncode == 0 and "PASSED" in r.stdout, "SKILL: 'run check' passes\n" + _out(r)
    assert "optional" in r.stdout, "run.py: workload/environment cards are reported, not required"


# Phase 2: Synthesis Loop


def test_setup_the_reference_a_mutant_and_a_validated_test(lead, S):
    syn = S["syn"]
    iface, ref, mut = (
        syn / "evaluator" / "interface",
        syn / "evaluator" / "reference",
        syn / "evaluator" / "mutants" / "overflow",
    )
    _write(iface / "README.md", "create_cache(capacity) -> get/put/size\n")
    # A source file in the interface is folded into the published artifact/ (artifacts.md), so
    # best/artifact/ is not byte-identical to synthesis/impl/; the release checks must know that.
    _write(iface / "cache_api.py", "REQUIRED = ('get', 'put', 'size')\n")
    _write(ref / "cache.py", REFERENCE)
    _write(
        mut / "cache.py",
        textwrap.dedent(REFERENCE).replace("> self.cap:", "> self.cap + 1:"),
    )
    tests = syn / "tests"
    # verification.md: the suite is tests/ plus a test.sh the evaluator writes; the harness only
    # runs `bash test.sh [file...]` with SKYDISCOVER_IMPL (here a directory holding cache.py).
    _write(
        tests / "test.sh",
        """
        #!/usr/bin/env bash
        cd "$(dirname "$0")"
        tests=("$@"); [ ${#tests[@]} -eq 0 ] && tests=(*.py)
        impl="$SKYDISCOVER_IMPL"; [ -d "$impl" ] || impl="$(dirname "$impl")"
        rc=0
        for t in "${tests[@]}"; do
          PYTHONPATH="$impl:$SKYDISCOVER_INTERFACE" python3 "$t" || { echo "FAIL $t"; rc=1; }
        done
        exit $rc
        """,
    )
    test = tests / "capacity.py"
    _write(
        test,
        """
        import cache as m, sys
        c = m.create_cache(4)
        for i in range(10): c.put(i, i)
        sys.exit(0 if c.size() <= 4 else 1)
        """,
    )
    args = ("--interface", str(iface), "--reference", str(ref), "--mutant", str(mut))
    r = lead.script("validate_test.py", *args, "--test", str(test))
    assert (
        r.returncode == 0
    ), "evaluator (correctness): a reference *directory* under .skydiscover/ validates\n" + _out(r)
    weak = tests / "weak.py"
    _write(weak, "import cache, sys\nsys.exit(0)\n")
    r = lead.script("validate_test.py", *args, "--test", str(weak))
    assert r.returncode != 0, "verification.md: a test the mutant passes is refused"
    weak.unlink()
    S.update(iface=iface, ref=ref, mut=mut, tests=tests)


def test_builder_run_tests_and_the_delivery_hook(lead, S):
    run, syn, ref, mut = S["run"], S["syn"], S["ref"], S["mut"]
    impl = syn / "impl" / "cache.py"
    _write(impl, (ref / "cache.py").read_text())
    r = lead.script("run_tests.py", "--run", str(run))
    assert (
        r.returncode == 0
    ), "scripts README: 'run_tests.py --run' passes a correct candidate\n" + _out(r)
    impl.write_text((mut / "cache.py").read_text())
    r = lead.script("run_tests.py", "--run", str(run))
    assert r.returncode == 1, "run_tests.py: a broken candidate is exit 1, not 2\n" + _out(r)
    impl.write_text((ref / "cache.py").read_text())
    r = lead.sh("bash", str(HOOK), env={**lead.env, "SKYDISCOVER_RUN": str(run)})
    assert r.returncode == 0, "SKILL: SKYDISCOVER_RUN is all the delivery hook needs\n" + _out(r)
    S["impl"] = impl


def test_evaluator_checkpoint_binds_a_leaderboard_row(lead, S):
    run, syn = S["run"], S["syn"]
    r = lead.mod("spec.checkpoint", "snapshot", str(run))
    assert r.returncode != 0 and "leaderboard" in _out(
        r
    ), "SKILL iteration 3: the checkpoint refuses an unscored artifact"
    _jwrite(
        syn / "bench" / "leaderboard.json",
        [
            # Two baselines measured with the same inputs (the tutorial compares against FIFO and
            # LRU); the candidate names FIFO as its headline comparison.
            {"role": "baseline", "name": "FIFO", "metrics": {"app_hit_rate": 0.30}, "config": {},
             "input_digests": json.loads(lead.mod("spec.checkpoint", "inputs", str(run)).stdout)},
            {"role": "baseline", "name": "LRU", "metrics": {"app_hit_rate": 0.32}, "config": {},
             "input_digests": json.loads(lead.mod("spec.checkpoint", "inputs", str(run)).stdout)},
            {
                "impl": "synthesis/impl/cache.py",
                "baseline": "FIFO",
                "metrics": {"app_hit_rate": 0.39},
                "iteration": 1,
                "config": {},
                "input_digests": json.loads(lead.mod("spec.checkpoint", "inputs", str(run)).stdout),
            },
        ],  # fmt: skip
    )
    r = lead.mod("spec.checkpoint", "snapshot", str(run), "--became-best")
    assert r.returncode == 0, _out(r)
    pub = sorted((lead.work / "outputs" / "synthesize").glob("contract-cache_*"))
    assert (
        len(pub) == 1 and len(list((pub[0] / "checkpoints").iterdir())) == 1
    ), "artifacts.md: one checkpoint per scored iteration under outputs/synthesize/<slug>_<ts>/"
    cp = pub[0] / "checkpoints" / "checkpoint_1"
    assert {p.name for p in cp.iterdir() if not p.name.startswith(".")} == {
        "artifact",
        "score.json",
        "tests.json",
    }, "artifacts.md: a checkpoint is artifact/, score.json and tests.json, with hidden verification records"
    assert _jread(cp / "tests.json") == [
        "capacity.py"
    ], "artifacts.md: tests.json lists the tests the checkpoint was scored against"
    assert _jread(cp / "score.json")["score"] == {
        "app_hit_rate": 0.39
    }, "artifacts.md: score.json's `score` is the first metric of the leaderboard entry"
    assert [b["name"] for b in _jread(cp / "score.json")["baselines"]] == [
        "FIFO",
        "LRU",
    ], "artifacts.md: every comparable baseline, the named one first"
    assert "not covered by the audit" in _out(
        r
    ), "checkpoint.py: a new best without an audit stamp warns and names the auditor"
    S["pub"] = pub[0]


def test_auditor_defect_blocks_until_the_builder_closes_it(lead, S):
    run = S["run"]
    r = lead.mod(
        "spec.findings", "add", str(run),
        "--title", "counts hits on misses", "--kind", "hack", "--severity", "defect",
        "--detail", "stale values", "--test", "capacity.py",
        "--mutant", "synthesis/evaluator/mutants/overflow",
    )  # fmt: skip
    assert r.returncode == 0, _out(r)
    fid = [x for x in _jread(run / "decision_log.json") if x["kind"] == "hack"][0]["id"]
    assert (
        _jread(run / ".severity_snapshot.json")["severities"][fid] == "defect"
    ), "state.md: 'findings add' records the row's first severity"
    r = lead.script("run_tests.py", "--run", str(run), "--production-ready")
    out = _out(r)
    assert r.returncode != 0
    assert "DEFECT" in out.upper(), "verification.md 2: an open defect blocks the release\n" + out
    assert "audit" in out.lower(), "verification.md 4: the missing audit stamp is named too\n" + out
    r = lead.mod(
        "spec.findings", "set-status", str(run), "--id", fid,
        "--status", "waived", "--who", "ai", "--note", "fixed",
    )  # fmt: skip
    assert (
        r.returncode == 0 and r.stdout.strip() == "ok"
    ), "state.md: the coding agent closes a fixed defect"
    assert "set aside" in lead.mod("spec.decisions", str(run), "list").stdout
    S["fid"] = fid


def test_release_checks_pass_once_every_condition_holds(lead, S):
    run, tests, impl, ref = S["run"], S["tests"], S["impl"], S["ref"]
    _write(
        tests / "load.py",
        """
        import cache as m, sys
        c = m.create_cache(128)
        for i in range(7000):
            c.put(i % 300, i); c.get(i % 100)
        sys.exit(0 if c.size() <= 128 else 1)
        """,
    )
    r = lead.script(
        "validate_test.py", "--interface", str(S["iface"]), "--reference", str(ref),
        "--mutant", str(S["mut"]), "--test", str(tests / "load.py"),
    )  # fmt: skip
    assert (
        r.returncode == 0
    ), "verification.md: an operating-point test is validated like any other\n" + _out(r)
    board = _jread(S["syn"] / "bench/leaderboard.json")
    board.append(
        {
            **board[-1],
            "input_digests": json.loads(lead.mod("spec.checkpoint", "inputs", str(run)).stdout),
        }
    )
    _jwrite(S["syn"] / "bench/leaderboard.json", board)
    r = lead.mod("spec.checkpoint", "stamp-audit", str(run))
    assert r.returncode == 0 and (S["syn"] / "audit" / "completeness.json").is_file(), _out(r)
    r = lead.script("run_tests.py", "--run", str(run), "--production-ready")
    assert r.returncode == 0, "verification.md: all conditions hold, the release passes\n" + _out(r)

    os.utime(impl)
    r = lead.script("run_tests.py", "--run", str(run), "--production-ready")
    assert r.returncode == 0, "check_release: the audit stamp is by digest; a touch is not a change"
    impl.write_text(impl.read_text() + "# tweak\n")
    r = lead.script("run_tests.py", "--run", str(run), "--production-ready")
    assert r.returncode != 0, "check_release: changed bytes make the stamp stale"
    impl.write_text((ref / "cache.py").read_text())

    rows = _jread(run / "decision_log.json")
    for x in rows:
        if x["id"] == S["fid"]:
            x["severity"], x["status"] = "advisory", "open"
    _jwrite(run / "decision_log.json", rows)
    r = lead.script("run_tests.py", "--run", str(run), "--production-ready")
    assert r.returncode != 0 and "severity" in _out(
        r
    ), "state.md: a hand-relabelled defect is refused"
    for x in rows:
        if x["id"] == S["fid"]:
            x["severity"], x["status"] = "defect", "waived"
    _jwrite(run / "decision_log.json", rows)


# Phase 3: Final Deliverables


def test_run_finish_publishes_and_feeds_the_knowledge_base(lead, S):
    run, spec, domain = S["run"], S["spec"], S["domain"]
    for name in ("security", "reliability", "attack"):
        _write(run / "review" / f"{name}.md", f"# {name}\nnothing found\n")
    _write(run / "report.md", "# Report\n0.39 vs FIFO 0.39. 1 of 1 iterations. Scoped.\n")
    # Every release condition held in the previous step, so the claim is made the documented way:
    # finish re-runs the release checks on the published copy (best/artifact/, the audited
    # candidate with the interface folded in) before anything is deleted.
    r = lead.mod(
        "spec.run", "finish", str(run), "--export-to", str(lead.work), "--production-ready"
    )
    assert r.returncode == 0, (
        "SKILL Phase 3: 'run finish --production-ready' exits 0 when every release check holds\n"
        + _out(r)
    )
    best = S["pub"] / "best"
    assert (
        best / "artifact" / "cache_api.py"
    ).is_file(), "artifacts.md: the interface's source files are folded into best/artifact/"
    assert (
        "AUDIT PASSED" in (best / ".verification" / "checks.log").read_text()
    ), "verification.md 4: the audit stamp covers the published copy"
    assert {p.name for p in best.iterdir() if not p.name.startswith(".")} == {
        "artifact",
        "tests",
        "score.json",
        "spec.md",
    }, "artifacts.md: best/ is artifact/, tests/, score.json, spec.md"
    page = (best / "spec.md").read_text()
    assert "1.30× FIFO (0.3), 1.22× LRU (0.32)" in page, "spec.md: the margin against each baseline"
    assert "Verified: every test in `tests/` passes" in page, "spec.md: the final check is stated"
    assert (
        "## Properties" in page and "capacity.py" in page
    ), "spec.md names each property and its test"
    assert "## Reward hacks found" in page, "spec.md lists the hacks the Auditor found"
    assert (domain / "tests" / "capacity.py").is_file(), "state.md: the run's tests are saved"
    assert (domain / "tests" / "test.sh").is_file(), "state.md: and the test.sh that runs them"
    assert any(
        f["answered_by"] == "human" for f in _jread(domain / "decisions.json")
    ), "state.md: the user's answers are carried into the knowledge base"
    assert not run.exists(), "SKILL Phase 3: run finish deletes the run directory"
    marker = run.parent / f"{run.name}.done"
    assert (lead.work / marker.read_text().strip()) == S["pub"], "the .done marker names the result"
    assert all(
        p.name.startswith(".") for p in domain.rglob("*") if "lock" in p.name
    ), "knowledge base: lock files are hidden"


def test_a_second_run_in_the_domain_reuses_what_the_first_learned(lead, S):
    r = lead.mod("spec.paths", "run", "contract-cache-2")
    run2 = lead.work / r.stdout.strip()
    _write(run2 / "task.md", "---\ndomain: cache\n---\nsame\n")
    _jwrite(run2 / "specification" / "questions.json", QUESTIONS)
    r = lead.mod("spec.kept_tests", "lookup", "cache", "--props", "capacity")
    assert (
        r.returncode == 0 and "CANDIDATE" in r.stdout
    ), "SKILL Setup 3: kept_tests lookup finds it"
    assert lead.mod("spec.decisions", str(run2), "answer").returncode == 0
    hist = next(
        (
            d
            for d in _jread(run2 / "specification" / "answers.json")["decisions"]
            if d["id"] == "history"
        ),
        None,
    )
    assert (
        hist is not None and hist["chosen"] == "a bounded number" and hist["answered_by"] == "human"
    ), "state.md: the next run never re-asks a question the user settled"
