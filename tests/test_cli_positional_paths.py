"""Tests for CLI positional path parsing."""

import sys

import pytest

from skydiscover.cli import parse_args


def test_parse_args_single_path_uses_evaluation_file(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["skydiscover-run", "evaluate.py"])

    args = parse_args()

    assert args.initial_program is None
    assert args.evaluation_file == "evaluate.py"


def test_parse_args_two_paths_use_initial_and_evaluation(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["skydiscover-run", "seed.py", "evaluate.py"])

    args = parse_args()

    assert args.initial_program == "seed.py"
    assert args.evaluation_file == "evaluate.py"


def test_parse_args_rejects_more_than_two_paths(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["skydiscover-run", "a.py", "b.py", "c.py"])

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2
