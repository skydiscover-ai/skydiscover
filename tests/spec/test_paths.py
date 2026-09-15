"""Tests for the path settings and the on-disk layout (spec/paths.py).

Precedence is environment > project config > default, and every failure mode is soft: a missing,
unreadable, or malformed config must read as "no settings" rather than take a run down.
"""

import importlib.util
import pathlib
import subprocess
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PKG = _ROOT / "skydiscover" / "synthesize" / "spec"


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, _PKG / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


settings = _load("paths.py", "_set_paths")


def _clean_env(monkeypatch):
    for var in ("SKYDISCOVER_HOME", "SKYDISCOVER_RUNS", "SKYDISCOVER_OUTPUTS"):
        monkeypatch.delenv(var, raising=False)


def test_the_results_root_moves_like_the_other_two(tmp_path, monkeypatch):
    """`outputs/synthesize` is a default, not a fixed layout: config.toml or $SKYDISCOVER_OUTPUTS
    moves it, and the CLI prints the root in effect for shell scripts."""
    _clean_env(monkeypatch)
    assert settings.outputs() == pathlib.Path("outputs/synthesize")
    kit = _cfg(tmp_path, 'outputs = "results"\n')
    assert settings.outputs(kit) == pathlib.Path("results")
    monkeypatch.setenv("SKYDISCOVER_OUTPUTS", "/srv/results")
    assert settings.outputs(kit) == pathlib.Path("/srv/results")
    r = subprocess.run(
        [sys.executable, "-m", "skydiscover.synthesize.spec.paths", "outputs"],
        capture_output=True,
        text=True,
    )
    assert r.stdout.strip() == "/srv/results"


def test_a_run_under_an_absolute_runs_root_belongs_to_no_project(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    assert settings.project_of(tmp_path / ".skydiscover" / "demo") == tmp_path.resolve()
    monkeypatch.setenv("SKYDISCOVER_RUNS", str(tmp_path / "elsewhere"))
    assert settings.project_of(tmp_path / "elsewhere" / "demo") is None


def _cfg(tmp_path, body):
    """A throwaway synthesize root holding config.toml (paths.py reads <kit>/config.toml)."""
    (tmp_path / "config.toml").write_text(body, encoding="utf-8")
    return tmp_path


def test_defaults_with_no_config(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    assert settings.runs(tmp_path) == pathlib.Path(".skydiscover")
    assert settings.home(tmp_path) == pathlib.Path.home() / ".skydiscover"


def test_config_toml_is_read(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    d = _cfg(tmp_path, 'runs = "runs/"\nhome = "/opt/shared-kb"\n')
    assert settings.runs(d) == pathlib.Path("runs/")
    assert settings.home(d) == pathlib.Path("/opt/shared-kb")


def test_the_shipped_config_is_the_one_in_effect(monkeypatch):
    # Called with no root, settings resolves <synthesize>/config.toml — the real file, so the
    # committed config and the code that reads it can never drift apart silently.
    _clean_env(monkeypatch)
    shipped = _PKG.parent / "config.toml"
    assert settings._config_file() == shipped
    assert set(settings._load()) <= {"runs", "home", "outputs"}


def test_a_deleted_config_still_works(tmp_path, monkeypatch):
    # Zero-config: no file at all -> every default holds, nothing raises.
    _clean_env(monkeypatch)
    empty = tmp_path / "no-config-here"
    empty.mkdir()
    assert settings._config_file(empty) is None
    assert settings.runs(empty) == pathlib.Path(".skydiscover")


def test_env_overrides_the_file(tmp_path, monkeypatch):
    # The hook communicates through the environment, so a checked-in file must never win over it.
    d = _cfg(tmp_path, 'runs = "from-file"\nhome = "/from/file"\n')
    monkeypatch.setenv("SKYDISCOVER_HOME", "/from/env")
    monkeypatch.setenv("SKYDISCOVER_RUNS", "from-env")
    assert settings.home(d) == pathlib.Path("/from/env")
    assert settings.runs(d) == pathlib.Path("from-env")


def test_malformed_config_falls_back_to_defaults(tmp_path, monkeypatch):
    # A config typo must not be able to break a run: unparseable reads as no settings.
    _clean_env(monkeypatch)
    d = _cfg(tmp_path, "runs = [unclosed\n")
    assert settings.runs(d) == pathlib.Path(".skydiscover")


def test_unknown_and_wrongly_typed_keys_are_ignored(tmp_path, monkeypatch):
    # Only the two documented string keys are settings; anything else (including a per-run decision
    # someone tries to smuggle in) is dropped.
    _clean_env(monkeypatch)
    d = _cfg(tmp_path, 'runs = 42\nbudget = "thorough"\ninvolvement = "pair"\nhome = "/ok"\n')
    assert settings.runs(d) == pathlib.Path(".skydiscover")  # wrong type -> default
    assert settings.home(d) == pathlib.Path("/ok")
    assert "budget" not in settings._load(d) and "involvement" not in settings._load(d)


def test_tilde_is_expanded(tmp_path, monkeypatch):
    _clean_env(monkeypatch)
    d = _cfg(tmp_path, 'home = "~/elsewhere"\n')
    assert settings.home(d) == pathlib.Path.home() / "elsewhere"


def test_every_store_hangs_off_the_one_home(monkeypatch):
    """kept_tests, findings, and the wiki all derive from paths.home() through Domain, so a configured
    home is honored by every store at once and there is one spelling of a domain."""
    monkeypatch.setenv("SKYDISCOVER_HOME", "/tmp/store-env-wins")
    from skydiscover.synthesize.spec import decisions, kept_tests
    from skydiscover.synthesize.spec.paths import Domain

    root = pathlib.Path("/tmp/store-env-wins")
    d = Domain("KV  Store")
    assert d.path == root / "kv-store"
    assert d.tests == root / "kv-store" / "tests"
    assert d.tests_index == root / "kv-store" / "tests" / "index.json"
    assert d.decisions == root / "kv-store" / "decisions.json"
    assert d.wiki == root / "kv-store" / "wiki"
    assert kept_tests._store("kv_store").tests == d.tests
    assert decisions.Domain("kv-store").decisions == d.decisions


def test_domain_slug_is_the_one_spelling():
    assert settings.domain_slug("KV Store") == "kv-store"
    assert settings.domain_slug("  kv--store ") == "kv-store"
    assert settings.domain_slug("Job_Scheduler v2") == "job-scheduler-v2"
    assert settings.domain_slug("缓存 系统") == "缓存-系统"  # any script, never one shared folder
    with pytest.raises(ValueError):
        settings.domain_slug("")
    with pytest.raises(ValueError):
        settings.domain_slug("---")
    assert settings.domains(pathlib.Path("/nonexistent")) == []


def test_a_domain_named_cache_gets_its_own_knowledge_base(tmp_path):
    """The tutorial's domain is `cache`; its folder must not be the clone cache, and the clone
    cache must not read as a domain."""
    d = settings.Domain("cache", tmp_path)
    assert d.path == tmp_path / "cache"
    assert settings.source_cache(tmp_path) == tmp_path / ".cache" / "sources"
    settings.source_cache(tmp_path).mkdir(parents=True)
    (tmp_path / "shared" / "wiki").mkdir(parents=True)
    d.tests.mkdir(parents=True)
    assert [x.slug for x in settings.domains(tmp_path)] == ["cache"]
    with pytest.raises(ValueError):
        settings.domain_slug("shared")


def test_run_layout_groups_by_phase(tmp_path):
    run = settings.Run(tmp_path / "kv")
    assert run.task == tmp_path / "kv" / "task.md"
    assert run.decision_log == tmp_path / "kv" / "decision_log.json"
    assert run.report == tmp_path / "kv" / "report.md"
    assert run.sources == tmp_path / "kv" / "specification" / "sources"
    assert run.references == tmp_path / "kv" / "specification" / "references"
    assert run.spec == tmp_path / "kv" / "specification" / "spec.json"
    assert run.reference_tests == tmp_path / "kv" / "specification" / "references" / "tests"
    assert run.questions == tmp_path / "kv" / "specification" / "questions.json"
    assert run.answers == tmp_path / "kv" / "specification" / "answers.json"
    cards = tmp_path / "kv" / "specification" / "cards"
    assert run.environment_card == cards / "environment.json"
    assert run.workload_card == cards / "workload.json"
    assert run.requirements_card == cards / "requirements.json"
    assert run.plan == tmp_path / "kv" / "synthesis" / "plan.md"
    assert run.impl == tmp_path / "kv" / "synthesis" / "impl"
    assert run.interface == tmp_path / "kv" / "synthesis" / "evaluator" / "interface"
    assert run.tests == tmp_path / "kv" / "synthesis" / "tests"
    assert run.leaderboard == tmp_path / "kv" / "synthesis" / "bench" / "leaderboard.json"
    assert run.profiles == tmp_path / "kv" / "synthesis" / "bench" / "profiles"
    assert run.completeness == tmp_path / "kv" / "synthesis" / "audit" / "completeness.json"
    assert run.severity == tmp_path / "kv" / ".severity_snapshot.json"
    assert run.review == tmp_path / "kv" / "review"
    assert run.domain() is None and not run.is_proof_run()


def test_run_create_writes_the_readme_and_phase_folders(tmp_path):
    run = settings.Run(tmp_path / "fast-cache").create()
    assert run.readme.is_file()
    text = run.readme.read_text()
    assert "outputs/synthesize/fast-cache_<timestamp>/best/" in text
    assert "specification/" in text and "synthesis/" in text and "review/" in text
    for d in run.phases():
        assert d.is_dir()
    run.create()  # idempotent
    assert len(list(run.path.iterdir())) == 4


def test_task_front_matter_names_the_domain(tmp_path):
    run = settings.Run(tmp_path / "kv").create()
    run.task.write_text("---\ndomain: KV Store\nchecked_by: proof  # formal\n---\nBuild a store.\n")
    assert run.domain() == "kv-store"
    assert run.is_proof_run()
    assert settings.resolve_decision_log(str(run.path)) == run.decision_log
    run.decision_log.write_text("[]")
    assert settings.resolve_decision_log(str(run.decision_log)) == run.decision_log


def test_run_dir_applies_the_configured_root(tmp_path, monkeypatch):
    # The lead asks for this path instead of spelling `.skydiscover/<slug>`, so a configured root is
    # actually honored rather than silently producing a second run tree.
    _clean_env(monkeypatch)
    assert settings.run_dir("kv-a1b2", tmp_path) == pathlib.Path(".skydiscover/kv-a1b2")
    d = _cfg(tmp_path, 'runs = "build/runs"\n')
    assert settings.run_dir("kv-a1b2", d) == pathlib.Path("build/runs/kv-a1b2")
    monkeypatch.setenv("SKYDISCOVER_RUNS", "from-env")
    assert settings.run_dir("kv-a1b2", d) == pathlib.Path("from-env/kv-a1b2")


@pytest.mark.parametrize("bad", ["../../etc", "a/b", ".hidden", "", "   ", "Upper", "a b"])
def test_run_dir_rejects_an_unsafe_slug(bad, monkeypatch):
    # The slug names a directory that is later exported and removed, so traversal, separators, and a
    # leading dot are refused rather than normalized.
    _clean_env(monkeypatch)
    with pytest.raises(ValueError):
        settings.run_dir(bad)


def test_run_subcommand_creates_and_prints_the_path_and_rejects_junk(capsys, monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    monkeypatch.setenv("SKYDISCOVER_RUNS", str(tmp_path / "runs"))
    monkeypatch.setenv("SKYDISCOVER_HOME", str(tmp_path / "home"))
    assert settings.main(["run", "kv-a1b2"]) == 0
    printed = capsys.readouterr().out.strip()
    assert printed.endswith("kv-a1b2") and (pathlib.Path(printed) / "README.md").is_file()
    assert settings.main(["domain", "KV", "Store"]) == 0
    assert capsys.readouterr().out.strip() == str(tmp_path / "home" / "kv-store")
    assert settings.main(["run", "../escape"]) == 2
    assert settings.main(["run"]) == 2  # missing slug
    assert settings.main(["bogus"]) == 2  # unknown argument


def test_a_knowledge_base_written_under_the_old_name_is_taken_over(tmp_path):
    """The first releases saved the user's answers as findings.json; the first touch renames it."""
    from skydiscover.synthesize.spec.paths import Domain

    kb = tmp_path / "cache"
    kb.mkdir()
    (kb / "findings.json").write_text("[]", encoding="utf-8")
    path = Domain("cache", root=tmp_path).decisions
    assert path == kb / "decisions.json" and path.is_file() and not (kb / "findings.json").exists()
    assert Domain("cache", root=tmp_path).decisions == path  # and again is a no-op
