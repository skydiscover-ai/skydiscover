"""The delivery check: run_tests.py and the release checks in check_release.py.

Loaded from their files (they are standalone scripts, not package modules).

What is pinned down:
  * the suite is opaque: run_tests runs its test.sh and reads the exit code, nothing else;
  * a suite that cannot run (no test.sh, out of time) is 'could not check' (2), never a failure
    blamed on the implementation (1);
  * the severity snapshot (kept by the findings CLI) is the only record of what a finding's severity
    was, so its absence blocks the release instead of disabling the downgrade check;
  * a defect is closed by the user or by a proven fix, nothing else.
"""

import importlib.util
import json
import os
import pathlib
import sys
import textwrap

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "skydiscover" / "synthesize" / "workflow" / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_sky_{name}", _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"_sky_{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


check_release = _load("check_release")
run_tests = _load("run_tests")

# A suite whose tests read the implementation as a text file: a test passes iff the file holds
# the word the test names. Language-free, so the harness is exercised and nothing else.
TEST_SH = textwrap.dedent("""\
    #!/usr/bin/env bash
    cd "$(dirname "$0")"
    tests=("$@"); [ ${#tests[@]} -eq 0 ] && tests=(*.t)
    rc=0
    for t in "${tests[@]}"; do
      if grep -q "$(cat "$t")" "$SKYDISCOVER_IMPL"; then echo "PASS $t"; else echo "FAIL $t"; rc=1; fi
    done
    exit $rc
    """)


def _suite(tmp_path, tests=("d1",), impl_text="d1 ok"):
    """(tests dir, impl file, interface dir) with test.sh, one test per name, and an impl."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(exist_ok=True)
    (tests_dir / "test.sh").write_text(TEST_SH)
    for t in tests:
        (tests_dir / f"{t}.t").write_text(t)
    impl = tmp_path / "impl.txt"
    impl.write_text(impl_text)
    iface = tmp_path / "iface"
    iface.mkdir(exist_ok=True)
    return tests_dir, impl, iface


def _write_log(tmp_path, rows):
    (tmp_path / "decision_log.json").write_text(json.dumps(rows), encoding="utf-8")
    return tmp_path / "decision_log.json"


def _snapshot(tmp_path, severities):
    (tmp_path / ".severity_snapshot.json").write_text(
        json.dumps({"severities": severities}), encoding="utf-8"
    )


def _row(fid="D1", severity="defect", status="waived", **kw):
    row = {
        "id": fid,
        "title": f"{fid} title",
        "kind": "spec",
        "detail": "a requirement",
        "status": status,
        "severity": severity,
    }
    row.update(kw)
    return row


# the snapshot may not vanish


def test_a_missing_snapshot_blocks_a_log_that_has_findings(tmp_path, capsys):
    """Without the snapshot the downgrade check cannot run, so the release is blocked."""
    log = _write_log(tmp_path, [_row(severity="advisory", status="open")])
    assert check_release.defects(log, *_suite(tmp_path)) == 2
    assert "no severity snapshot" in capsys.readouterr().err


def test_deleting_the_snapshot_does_not_hide_a_downgrade(tmp_path, capsys):
    log = _write_log(tmp_path, [_row(severity="advisory", status="open")])
    _snapshot(tmp_path, {"D1": "defect"})
    assert check_release.defects(log, *_suite(tmp_path)) == 2
    assert "changed severity" in capsys.readouterr().err
    (tmp_path / ".severity_snapshot.json").unlink()
    assert check_release.defects(log, *_suite(tmp_path)) == 2


def test_a_corrupt_snapshot_blocks(tmp_path, capsys):
    log = _write_log(tmp_path, [_row(severity="advisory", status="open")])
    (tmp_path / ".severity_snapshot.json").write_text("{not json", encoding="utf-8")
    assert check_release.defects(log, *_suite(tmp_path)) == 2
    assert "no severity snapshot" in capsys.readouterr().err


def test_a_snapshot_neutered_to_empty_blocks_like_a_missing_one(tmp_path, capsys):
    """Emptying the snapshot to {} erases the same evidence deleting it does, so it is refused too."""
    log = _write_log(tmp_path, [_row(severity="advisory", status="open")])
    _snapshot(tmp_path, {})
    assert check_release.defects(log, *_suite(tmp_path)) == 2
    assert "no severity snapshot" in capsys.readouterr().err


def test_the_findings_cli_keeps_the_snapshot_so_a_normal_run_can_be_released(tmp_path, capsys):
    """Rows are only ever written through Findings with track_severity (the CLI and decisions), which
    records each id's first severity; so a log that grew the normal way is checkable, and a defect
    relabelled by hand is still caught."""
    from skydiscover.synthesize.spec.findings import Finding, Findings

    log = tmp_path / "decision_log.json"
    store = Findings(log, track_severity=True)
    store.add(Finding(id="D1", title="torn write", kind="spec", detail="x", severity="defect"))
    store.set_status("D1", "waived", who="ai", test="check_torn.py", mutant="torn.py")
    snap = json.loads((tmp_path / ".severity_snapshot.json").read_text())
    assert snap == {"severities": {"D1": "defect"}}
    assert Findings(log).all()[0].test == "check_torn.py"
    # A hand edit that relabels the defect does not change the record, and is caught.
    rows = json.loads(log.read_text())
    rows[0]["severity"] = "advisory"
    log.write_text(json.dumps(rows))
    assert check_release.defects(log, *_suite(tmp_path)) == 2
    assert "changed severity" in capsys.readouterr().err


def test_emptying_the_log_and_deleting_the_snapshot_is_refused(tmp_path, capsys):
    """Emptying the log to [] and deleting the snapshot is the cheapest bypass: without the snapshot
    nothing proves a defect was not erased, so its absence is refused rather than read as clean."""
    log = _write_log(tmp_path, [])
    assert not (tmp_path / ".severity_snapshot.json").exists()
    assert check_release.defects(log, *_suite(tmp_path)) == 2
    assert "no severity snapshot" in capsys.readouterr().err


def test_emptying_the_log_does_not_hide_a_defect_the_snapshot_recorded(tmp_path, capsys):
    """Deleting one row is caught by the per-row check; overwriting the log with [] used to skip it.
    The snapshot still names the defect, so the release is blocked."""
    log = _write_log(tmp_path, [])
    _snapshot(tmp_path, {"D1": "defect", "A1": "advisory"})
    assert check_release.defects(log, *_suite(tmp_path)) == 2
    assert "D1" in capsys.readouterr().err
    _snapshot(tmp_path, {"A1": "advisory"})
    assert check_release.defects(log, *_suite(tmp_path)) == 0


def test_an_unchanged_severity_passes(tmp_path):
    log = _write_log(tmp_path, [_row(severity="advisory", status="open")])
    _snapshot(tmp_path, {"D1": "advisory"})
    assert check_release.defects(log, *_suite(tmp_path)) == 0


def test_a_vanished_defect_is_a_downgrade(tmp_path, capsys):
    log = _write_log(tmp_path, [_row("D2", severity="advisory", status="open")])
    _snapshot(tmp_path, {"D1": "defect", "D2": "advisory"})
    assert check_release.defects(log, *_suite(tmp_path)) == 2
    assert "D1: was a defect, now 'absent'" in capsys.readouterr().err


# only the user closes a defect without a fix


def test_a_severity_never_changes_whoever_is_named_on_the_row(tmp_path):
    """The user sets a defect aside by waiving it (status), not by relabelling it (severity)."""
    for who in ("human", "ai"):
        log = _write_log(tmp_path, [_row(severity="advisory", status="open", answered_by=who)])
        _snapshot(tmp_path, {"D1": "defect"})
        assert check_release.defects(log, *_suite(tmp_path)) == 2
    log = _write_log(tmp_path, [_row(severity="defect", status="waived", answered_by="human")])
    _snapshot(tmp_path, {"D1": "defect"})
    assert check_release.defects(log, *_suite(tmp_path)) == 0


def test_an_open_defect_blocks(tmp_path, capsys):
    log = _write_log(tmp_path, [_row(status="open")])
    _snapshot(tmp_path, {"D1": "defect"})
    assert check_release.defects(log, *_suite(tmp_path)) == 2
    assert "1 open defect(s)" in capsys.readouterr().err


def test_a_defect_the_user_closed_needs_no_test(tmp_path, capsys):
    log = _write_log(tmp_path, [_row(answered_by="human")])
    _snapshot(tmp_path, {"D1": "defect"})
    assert check_release.defects(log, *_suite(tmp_path)) == 0
    assert "CLOSED BY THE USER: D1" in capsys.readouterr().out


def test_a_defect_the_agent_closed_needs_a_test_and_a_mutant(tmp_path, capsys):
    tests, impl, iface = _suite(tmp_path, tests=("d1",), impl_text="d1 fixed")
    (tmp_path / "mutant.txt").write_text("broken")  # the test's word is not in it
    _snapshot(tmp_path, {"D1": "defect"})

    log = _write_log(tmp_path, [_row(answered_by="ai")])
    assert check_release.defects(log, tests, impl, iface) == 2
    assert "no test recorded" in capsys.readouterr().err

    log = _write_log(tmp_path, [_row(answered_by="ai", test="d1.t")])
    assert check_release.defects(log, tests, impl, iface) == 2
    assert "no mutant is recorded" in capsys.readouterr().err

    # the test passes the impl and fails the mutant: the fix is proven
    log = _write_log(tmp_path, [_row(answered_by="ai", test="d1.t", mutant="mutant.txt")])
    assert check_release.defects(log, tests, impl, iface) == 0
    assert "1 fixed defect(s)" in capsys.readouterr().out

    # an always-green test proves nothing
    (tmp_path / "mutant.txt").write_text("d1 also here")
    assert check_release.defects(log, tests, impl, iface) == 2
    assert "does not catch mutant" in capsys.readouterr().err

    # the mutant is named relative to the run and may be a directory (a broken copy of a codebase)
    mdir = tmp_path / "synthesis" / "evaluator" / "mutants" / "d1"
    mdir.mkdir(parents=True)
    (mdir / "m.txt").write_text("broken")
    log = _write_log(
        tmp_path, [_row(answered_by="ai", test="d1.t", mutant="synthesis/evaluator/mutants/d1")]
    )
    # a directory: grep on a directory fails, so the mutant is caught; the harness passes the path through
    assert check_release.defects(log, tests, impl, iface) == 0


# the suite is opaque: test.sh decides


def test_a_passing_suite_exits_0(tmp_path, capsys):
    tests, impl, iface = _suite(tmp_path, tests=("d1", "d2"), impl_text="d1 d2")
    assert run_tests.run_suite(tests, impl, iface) == 0
    assert "TESTS PASSED" in capsys.readouterr().out


def test_a_real_failure_exits_1_and_relays_the_output(tmp_path, capsys):
    tests, impl, iface = _suite(tmp_path, tests=("d1", "d2"), impl_text="d1 only")
    assert run_tests.run_suite(tests, impl, iface) == 1
    err = capsys.readouterr().err
    assert "TESTS FAILED" in err and "FAIL d2.t" in err and "PASS d1.t" in err


def test_no_test_sh_is_could_not_check(tmp_path, capsys):
    tests, impl, iface = _suite(tmp_path)
    (tests / "test.sh").unlink()
    assert run_tests.run_suite(tests, impl, iface) == 2
    assert "no test.sh" in capsys.readouterr().err


def test_a_missing_implementation_is_could_not_check(tmp_path, capsys):
    tests, impl, iface = _suite(tmp_path)
    impl.unlink()
    assert run_tests.run_suite(tests, impl, iface) == 2
    assert "implementation not found" in capsys.readouterr().err


def test_a_suite_that_does_not_finish_is_could_not_check(tmp_path, capsys, monkeypatch):
    tests, impl, iface = _suite(tmp_path)
    (tests / "test.sh").write_text("#!/usr/bin/env bash\nsleep 5\n")
    monkeypatch.setattr(run_tests.suite, "SLOW_SECS", 1)
    assert run_tests.run_suite(tests, impl, iface) == 2
    assert "did not finish" in capsys.readouterr().err


def test_test_sh_sees_the_impl_the_interface_and_runs_in_the_suite_dir(tmp_path):
    tests, impl, iface = _suite(tmp_path)
    (tests / "test.sh").write_text(
        '#!/usr/bin/env bash\n[ "$PWD" = "$(cd "$(dirname "$0")" && pwd)" ] || exit 3\n'
        'printf "%s\\n%s\\n" "$SKYDISCOVER_IMPL" "$SKYDISCOVER_INTERFACE" > "$PWD/seen"\n'
    )
    assert run_tests.run_suite(tests, impl, iface) == 0
    assert (tests / "seen").read_text().split() == [str(impl.resolve()), str(iface.resolve())]


def test_the_cli_runs_a_suite_end_to_end(tmp_path, capsys):
    tests, impl, iface = _suite(tmp_path)
    argv = ["--impl", str(impl), "--suite", str(tests), "--interface", str(iface)]
    assert run_tests.main(argv) == 0
    assert "TESTS PASSED" in capsys.readouterr().out
    impl.write_text("nothing")
    assert run_tests.main(argv) == 1


def test_test_narrows_the_run_to_the_named_files(tmp_path, capsys):
    tests, impl, iface = _suite(tmp_path, tests=("d1", "d2"), impl_text="d1 only")
    assert run_tests.run_suite(tests, impl, iface, ["d1.t"]) == 0
    assert "test.sh d1.t exited 0" in capsys.readouterr().out
    assert run_tests.run_suite(tests, impl, iface, ["d2.t"]) == 1
    assert "FAIL d2.t" in capsys.readouterr().err
    assert run_tests.run_suite(tests, impl, iface, ["nope.t"]) == 2
    assert "no such test" in capsys.readouterr().err
    argv = ["--impl", str(impl), "--suite", str(tests), "--test", "d1.t"]
    assert run_tests.main(argv) == 0


def test_the_cli_needs_a_run_or_a_suite_and_an_impl(tmp_path):
    with pytest.raises(SystemExit):
        run_tests.main(["--suite", str(tmp_path)])
    with pytest.raises(SystemExit):
        run_tests.main(["--production-ready"])


# audit


def test_the_audit_stamp_must_cover_the_impl_bytes_and_be_well_formed(tmp_path, capsys):
    from skydiscover.synthesize.spec.checkpoint import _tree_digest, input_digests

    impl = tmp_path / "synthesis/impl"
    impl.mkdir(parents=True)
    (impl / "a.cc").write_text("", encoding="utf-8")
    audit = tmp_path / "synthesis/audit"
    audit.mkdir()
    assert check_release.audit(impl, audit) == 2
    assert "missing" in capsys.readouterr().err
    stamp = audit / "completeness.json"
    stamp.write_text("{}", encoding="utf-8")
    assert check_release.audit(impl, audit) == 2
    assert "malformed" in capsys.readouterr().err
    # A stamp with the digest is compared by content, whatever the clocks say.
    stamp.write_text(
        json.dumps(
            {
                "findings": [],
                "artifact_id": _tree_digest(impl),
                "input_digests": input_digests(tmp_path, audit=True),
            }
        ),
        encoding="utf-8",
    )
    assert check_release.audit(impl, audit) == 0
    (impl / "a.cc").write_text("// changed", encoding="utf-8")
    assert check_release.audit(impl, audit) == 2
    assert "different implementation bytes" in capsys.readouterr().err
    # A stamp with no digest binds to nothing, so it is refused whatever the clocks say.
    stamp.write_text(json.dumps({"findings": []}), encoding="utf-8")
    assert check_release.audit(impl, audit) == 2
    assert "no artifact_id" in capsys.readouterr().err


def test_stamp_audit_writes_a_stamp_that_covers_the_candidate(tmp_path):
    from skydiscover.synthesize.spec.checkpoint import stamp_audit
    from skydiscover.synthesize.spec.paths import Run

    run = Run(tmp_path / "run").create()
    run.impl.mkdir(parents=True)
    (run.impl / "m.py").write_text("x = 1\n", encoding="utf-8")
    stamp = stamp_audit(run.path, ["F-1"])
    assert stamp == run.completeness
    assert check_release.audit(run.impl, run.audit) == 0
    assert json.loads(stamp.read_text())["findings"] == ["F-1"]


# a bare drop is the agent's act, not the user's


def test_a_bare_drop_no_longer_closes_a_defect(tmp_path, capsys):
    """The impersonation chain from #129: the auditor files a defect, the agent runs a plain
    `decisions <run> drop <row>` with no flags, and before the fix the gate printed DEFECTS
    CLOSED BY THE USER. A bare drop is now attributed to the AI, and an AI-waived defect
    without a proven fix keeps blocking."""
    from skydiscover.synthesize.spec import decisions
    from skydiscover.synthesize.spec.findings import Finding, Findings
    from skydiscover.synthesize.spec.paths import Run

    run = Run(tmp_path / ".skydiscover" / "demo").create()
    run.task.write_text("---\ndomain: demo\n---\n# Task\n")
    Findings(run.decision_log, track_severity=True).add(
        Finding(
            id="hack-1",
            title="drops requests under load",
            kind="hack",
            detail="quality averaged only over completed requests",
            severity="defect",
            answered_by="ai",
        )
    )
    assert check_release.defects(run.decision_log, *_suite(tmp_path)) == 2  # open: blocked

    decisions.drop(run, "hack-1")  # the agent's bare command, exactly as before the fix
    row = json.loads(run.decision_log.read_text())[0]
    assert row["answered_by"] == "ai"  # no user authority was minted
    assert check_release.defects(run.decision_log, *_suite(tmp_path)) == 2  # still blocked

    decisions.drop(run, "hack-1", by="human")  # the user's explicit call
    assert check_release.defects(run.decision_log, *_suite(tmp_path)) == 0  # closes it


def test_audit_covers_the_candidate_actually_named(tmp_path):
    # The stamp covers synthesis/impl/; a candidate elsewhere (--impl, SKYDISCOVER_IMPL) must be
    # audited as the bytes it is, not waved through on the stamp of a different tree.
    from spec.paths import Run

    run = Run(tmp_path / "run").create()
    run.impl.mkdir(parents=True, exist_ok=True)
    (run.impl / "cache.py").write_text("def create_cache(c):\n    return None\n")
    other = tmp_path / "other.py"
    other.write_text("def create_cache(c):\n    return None\n")
    assert run_tests._audited(run, str(run.impl / "cache.py")) == run.impl
    assert run_tests._audited(run, str(run.impl)) == run.impl
    assert run_tests._audited(run, str(other)) == other


def test_the_audit_stamp_covers_the_published_copy_of_the_candidate(tmp_path, capsys):
    """`run finish` checks best/artifact/ in a temporary directory: the audited synthesis/impl/
    with the interface folded in, named by --impl from outside the run. The stamp was written for
    synthesis/impl/, so the check must recognise that published form (and still read the run's
    own requirements, tests, evaluator and decisions, not the directory holding the copy)."""
    from skydiscover.synthesize.spec.checkpoint import _stage_artifact, stamp_audit
    from skydiscover.synthesize.spec.paths import Run

    run = Run(tmp_path / "project" / ".skydiscover" / "demo").create()
    run.impl.mkdir(parents=True)
    (run.impl / "__init__.py").write_text("import cache_api\ndef create_cache(c): return None\n")
    run.interface.mkdir(parents=True)
    (run.interface / "__init__.py").write_text("")
    (run.interface / "cache_api.py").write_text("REQUIRED = ('get', 'put', 'size')\n")
    stamp_audit(run.path, [])
    assert check_release.audit(run.impl, run.audit) == 0

    # The published copy lives elsewhere and carries the interface package the impl lacks.
    published = _stage_artifact(run.impl, run.interface, tmp_path / "out" / "best")
    assert (published / "interface" / "cache_api.py").is_file()
    assert check_release.audit(published, run.audit) == 0
    assert "AUDIT PASSED" in capsys.readouterr().out

    # Bytes the auditor did not read are still refused, wherever they live.
    (published / "__init__.py").write_text("def create_cache(c): return 'other'\n")
    assert check_release.audit(published, run.audit) == 2
    assert "different implementation bytes" in capsys.readouterr().err
    (run.impl / "__init__.py").write_text("def create_cache(c): return 'changed'\n")
    assert check_release.audit(run.impl, run.audit) == 2
    assert "different implementation bytes" in capsys.readouterr().err


def test_run_exposes_the_run_dir_to_the_tests(tmp_path, monkeypatch):
    """--run makes SKYDISCOVER_RUN available to the tests and interfaces that need the run's tree."""
    from spec.paths import Run

    run = Run(tmp_path / "run").create()
    for d in (run.impl, run.tests, run.interface):
        d.mkdir(parents=True, exist_ok=True)
    (run.impl / "cache.py").write_text("def create_cache(c):\n    return None\n")
    (run.tests / "test.sh").write_text("exit 0\n")
    seen = {}

    def fake_suite(tests, impl, interface, names=()):
        seen["env"] = os.environ.get("SKYDISCOVER_RUN")
        return 2  # stop here; the rest of the gate is not under test

    monkeypatch.delenv("SKYDISCOVER_RUN", raising=False)
    monkeypatch.setattr(run_tests, "run_suite", fake_suite)
    assert run_tests.main(["--run", str(run.path), "--production-ready"]) == 2
    assert seen == {"env": str(run.path.resolve())}


def test_a_proof_run_is_checked_by_the_suite_the_task_shipped(tmp_path, capsys):
    """checked_by: proof changes the loop (no benchmark), not the check: the task's test.sh runs
    the proof against the candidate in synthesis/impl/, like any other suite."""
    from spec.paths import Run

    run = Run(tmp_path / "run").create()
    run.task.write_text("---\nchecked_by: proof\n---\nProve it.\n")
    for d in (run.impl, run.tests, run.interface):
        d.mkdir(parents=True, exist_ok=True)
    (run.interface / "Spec.v").write_text("Theorem t : True.\n")
    (run.tests / "test.sh").write_text("bash proof.sh\n")
    (run.tests / "proof.sh").write_text(
        'grep -q "Theorem t" "$SKYDISCOVER_INTERFACE/Spec.v" && ! grep -rqw Admitted "$SKYDISCOVER_IMPL"\n'
    )
    (run.impl / "Proof.v").write_text("Proof. exact I. Qed.\n")
    assert run_tests.main(["--run", str(run.path)]) == 0
    assert "TESTS PASSED" in capsys.readouterr().out
    (run.impl / "Proof.v").write_text("Admitted.\n")
    assert run_tests.main(["--run", str(run.path)]) == 1
    assert "TESTS FAILED" in capsys.readouterr().err


def test_wall_secs_gives_up_with_exit_2_instead_of_outliving_the_hook(
    tmp_path, capsys, monkeypatch
):
    """The delivery hook is killed by its harness after a fixed time, and a killed hook is not a
    block. run_tests therefore stops itself first: exit 2, the check incomplete, nothing shipped."""
    import time

    from spec.paths import Run

    run = Run(tmp_path / "run").create()
    for d in (run.impl, run.tests, run.interface):
        d.mkdir(parents=True, exist_ok=True)
    (run.impl / "cache.py").write_text("def create_cache(c):\n    return None\n")
    (run.tests / "test.sh").write_text("exit 0\n")

    def slow_suite(tests, impl, interface, names=()):
        time.sleep(5)
        return 0

    monkeypatch.setattr(run_tests, "run_suite", slow_suite)
    started = time.monotonic()
    with pytest.raises(SystemExit) as stop:
        run_tests.main(["--run", str(run.path), "--wall-secs", "1"])
    assert stop.value.code == 2 and time.monotonic() - started < 4
    assert "not finished after 1s" in capsys.readouterr().err
