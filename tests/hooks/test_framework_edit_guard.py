"""The framework-edit guard: a run's agents may not patch the installed SkyDiscover."""

import importlib.util
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_GUARD = _ROOT / "skydiscover" / "synthesize" / "workflow" / "hooks" / "framework_edit_guard.py"


def _load():
    spec = importlib.util.spec_from_file_location("_sky_framework_edit_guard", _GUARD)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_sky_framework_edit_guard"] = mod
    spec.loader.exec_module(mod)
    return mod


guard = _load()
_FRAMEWORK_FILE = str(_ROOT / "skydiscover" / "synthesize" / "spec" / "checkpoint.py")


def test_blocks_an_edit_to_the_framework_from_a_user_project(tmp_path):
    # The live llm-router run edited spec/checkpoint.py in its checkout to get past a snapshot
    # refusal; that must be refused with a message that says what to do instead.
    for tool in ("Edit", "Write", "MultiEdit"):
        block, msg = guard.decide(
            {"tool_name": tool, "tool_input": {"file_path": _FRAMEWORK_FILE}, "cwd": str(tmp_path)}
        )
        assert block is True, tool
        assert "decision log" in msg and "checkpoint.py" in msg


def test_allows_edits_to_the_run_and_the_project(tmp_path):
    for path in (tmp_path / ".skydiscover/demo/synthesis/impl/cache.py", tmp_path / "workload.py"):
        block, _ = guard.decide(
            {"tool_name": "Edit", "tool_input": {"file_path": str(path)}, "cwd": str(tmp_path)}
        )
        assert block is False, path


def test_allows_framework_edits_inside_the_framework_repository_itself():
    # A developer working on SkyDiscover in its own checkout is not a run patching its checker.
    block, _ = guard.decide(
        {"tool_name": "Edit", "tool_input": {"file_path": _FRAMEWORK_FILE}, "cwd": str(_ROOT)}
    )
    assert block is False


def test_ignores_other_tools_and_malformed_payloads(tmp_path):
    assert guard.decide({"tool_name": "Bash", "tool_input": {"command": "ls"}}) == (False, "")
    assert guard.decide({"tool_name": "Edit", "tool_input": {}}) == (False, "")
    assert guard.decide({}) == (False, "")
