"""The tutorial demo (play.py) must read the spec the way a real run writes it."""

import importlib.util
import pathlib
import sys

import pytest

_PLAY = (
    pathlib.Path(__file__).resolve().parents[1]
    / "skydiscover"
    / "synthesize"
    / "examples"
    / "tutorial"
    / "play.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("_sky_tutorial_play", _PLAY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_sky_tutorial_play"] = mod
    spec.loader.exec_module(mod)
    return mod


play = _load()


def test_interface_operations_accepts_every_spec_shape():
    # What a real run writes: a {name: signature} mapping (KeyError: 'operations' before the fix).
    real = {"get": "get(key) -> value | None", "put": "put(key, value)", "size": "size() -> int"}
    assert play.interface_operations(real) == ["get", "put", "size"]
    # The shape the demo originally assumed.
    assert play.interface_operations({"operations": ["get(key)  (hit or None)", "put(k, v)"]}) == [
        "get(key)  (hit or None)",
        "put(k, v)",
    ]
    # A bare list, and nothing at all, never raise.
    assert play.interface_operations(["a()", "b()"]) == ["a()", "b()"]
    assert play.interface_operations(None) == []
    assert play.interface_operations({}) == []


LRU = """
from collections import OrderedDict
class LRU:
    def __init__(self, cap): self.cap, self.d = cap, OrderedDict()
    def get(self, k):
        if k in self.d: self.d.move_to_end(k); return self.d[k]
        return None
    def put(self, k, v):
        self.d[k] = v; self.d.move_to_end(k)
        while len(self.d) > self.cap: self.d.popitem(last=False)
    def size(self): return len(self.d)
"""


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_step_4_scores_a_multi_file_candidate_the_way_the_harness_imports_it(tmp_path, monkeypatch):
    """An end-to-end synthesis usually produces a package, not one file. The loader must import
    it as the harness does: relative imports, `impl_under_test.` imports, and the pinned interface
    all resolve; the entry is the one the leaderboard scored, never a stray scaffold file."""
    from skydiscover.synthesize.spec.paths import Run

    monkeypatch.setenv("SKYDISCOVER_RUNS", str(tmp_path / ".skydiscover"))
    monkeypatch.chdir(tmp_path)
    p = _load()  # re-import so RUNS reads the patched environment
    run = Run(tmp_path / ".skydiscover" / "demo").create()
    _write(run.interface / "cache_api.py", "REQUIRED = ('get', 'put', 'size')\n")
    _write(
        run.impl / "__init__.py",
        "from .core import LRU\nfrom impl_under_test.policy import capacity_of\nimport cache_api\n"
        "def create_cache(capacity): return LRU(capacity_of(capacity))\n",
    )
    _write(run.impl / "core.py", LRU)
    _write(run.impl / "policy.py", "def capacity_of(c): return c\n")
    _write(run.impl / "scratch.py", "raise RuntimeError('scaffolding must not be imported')\n")

    run_found, entry = p.current_candidate()
    assert run_found.path == run.path and entry == run.impl
    score = p.best_so_far()
    assert score is not None and 0 < score < 1

    # The published result: artifact/ is the package, record.json says the entry is "."
    best = tmp_path / "outputs" / "synthesize" / "demo_1" / "best"
    for name in ("__init__.py", "core.py", "policy.py"):
        _write(
            best / "artifact" / name,
            (run.impl / name).read_text().replace("import cache_api\n", ""),
        )
    _write(best / ".verification" / "record.json", '{"entry": "."}')
    assert p.replay(p.load_winner(best))["app_hit_rate"] == score

    # A single-file artifact, and a directory nobody can pick one file from.
    single = tmp_path / "outputs" / "synthesize" / "demo_2" / "best"
    _write(single / "artifact" / "cache.py", LRU + "\ndef create_cache(c): return LRU(c)\n")
    _write(single / ".verification" / "record.json", '{"entry": "cache.py"}')
    assert p.replay(p.load_winner(single))["app_hit_rate"] == score
    _write(tmp_path / "loose" / "a.py", "x = 1\n")
    _write(tmp_path / "loose" / "b.py", "y = 2\n")
    with pytest.raises(ValueError, match="exactly one .py file"):
        p.load_candidate(tmp_path / "loose")


def test_play_imports_when_copied_into_a_project_directory(tmp_path):
    """Step 1 has the reader `cp play.py .`; a shallow project path like ~/my-cache used to die with
    IndexError on `HERE.parents[3]`. Copied out, the package comes from the environment."""
    import os
    import shutil
    import subprocess

    project = tmp_path / "my-cache"
    project.mkdir()
    shutil.copy(_PLAY, project / "play.py")
    shutil.copy(_PLAY.parent / "workload.py", project / "workload.py")
    env = {**os.environ, "PYTHONPATH": str(_PLAY.parents[4])}
    r = subprocess.run(
        [sys.executable, "-c", "import play; print(play.RUNS)"],
        cwd=project, env=env, capture_output=True, text=True,
    )  # fmt: skip
    assert r.returncode == 0 and r.stdout.strip() == ".skydiscover", r.stderr
    r = subprocess.run(
        [sys.executable, "-c", "import play"],
        cwd=project, env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True, text=True,
    )  # fmt: skip
    assert r.returncode == 0 or "activate the skydiscover venv" in r.stderr
