"""`skydiscover <command>` subcommands.

The two modules in the README (`optimize` / `synthesize`) are the CLI's shape too:
`skydiscover optimize` runs a search, `skydiscover init` wires the synthesis
skill, `skydiscover viewer` replays a run. `skydiscover` is the only console
script -- the flat `skydiscover-run` / `skydiscover-viewer` entry points are
gone, so there is exactly one way to invoke each.
"""

from __future__ import annotations

import pytest

from skydiscover import main as entry
from skydiscover.optimize import cli


def test_top_level_help_lists_every_route(capsys):
    with pytest.raises(SystemExit) as exc:
        entry.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for command in ("optimize", "viewer", "init"):
        assert command in out


@pytest.mark.parametrize("command", ["optimize", "viewer"])
def test_forwarded_help_names_the_subcommand(command, capsys):
    """`--help` must reach the forwarded parser, not be eaten by the parent."""
    with pytest.raises(SystemExit) as exc:
        entry.main([command, "--help"])
    assert exc.value.code == 0
    assert f"usage: skydiscover {command}" in capsys.readouterr().out


def test_optimize_forwards_arguments_to_the_search_cli(monkeypatch):
    seen = {}

    def fake_main(argv=None, prog=None):
        seen["argv"], seen["prog"] = argv, prog
        return 0

    monkeypatch.setattr(cli, "main", fake_main)
    assert entry.main(["optimize", "program.py", "evaluator.py", "--search", "evox"]) == 0
    assert seen["argv"] == ["program.py", "evaluator.py", "--search", "evox"]
    assert seen["prog"] == "skydiscover optimize"


def test_optimize_propagates_the_exit_code(monkeypatch):
    monkeypatch.setattr(cli, "main", lambda argv=None, prog=None: 3)
    assert entry.main(["optimize", "evaluator.py"]) == 3


def test_parse_args_reads_the_argv_it_is_given():
    args = cli.parse_args(["program.py", "evaluator.py", "--search", "evox", "-i", "7"])
    assert (args.initial_program, args.evaluation_file) == ("program.py", "evaluator.py")
    assert (args.search, args.iterations) == ("evox", 7)


def test_unknown_subcommand_is_rejected():
    with pytest.raises(SystemExit) as exc:
        entry.main(["synthesise"])  # not a command; `init` wires synthesis
    assert exc.value.code != 0


def test_flat_console_scripts_are_gone():
    """`skydiscover` is the only entry point; the flat aliases were removed."""
    from pathlib import Path

    try:
        import tomllib  # Python >= 3.11
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        import tomli as tomllib

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    scripts = tomllib.loads(pyproject.read_text())["project"]["scripts"]
    assert set(scripts) == {"skydiscover"}


def test_every_known_external_backend_is_selectable_by_flag():
    """`--search` must offer every backend the dispatcher knows about.

    `alphaevolve` was in KNOWN_EXTERNAL and had a working backend, but was
    missing from _SEARCH_CHOICES, so the flag rejected it and the only way in
    was `search.type:` in a config file.
    """
    from skydiscover.optimize.cli import _SEARCH_CHOICES
    from skydiscover.optimize.extras.external import KNOWN_EXTERNAL

    missing = sorted(KNOWN_EXTERNAL - set(_SEARCH_CHOICES))
    assert not missing, f"backends the --search flag cannot reach: {missing}"


def test_alphaevolve_is_accepted_by_the_parser():
    args = cli.parse_args(["evaluator.py", "--search", "alphaevolve"])
    assert args.search == "alphaevolve"
