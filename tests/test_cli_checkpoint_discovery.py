"""Tests for checkpoint discovery helper in the CLI."""

from pathlib import Path

from skydiscover.cli import _find_latest_checkpoint


def test_find_latest_checkpoint_returns_highest_iteration(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    (checkpoint_dir / "checkpoint_2").mkdir()
    (checkpoint_dir / "checkpoint_10").mkdir()
    (checkpoint_dir / "checkpoint_1").mkdir()

    latest = _find_latest_checkpoint(str(checkpoint_dir))

    assert latest == str(checkpoint_dir / "checkpoint_10")


def test_find_latest_checkpoint_ignores_non_numeric_dirs(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    (checkpoint_dir / "latest").mkdir()
    (checkpoint_dir / "checkpoint_old").mkdir()
    (checkpoint_dir / "checkpoint_3").mkdir()

    latest = _find_latest_checkpoint(str(checkpoint_dir))

    assert latest == str(checkpoint_dir / "checkpoint_3")


def test_find_latest_checkpoint_returns_none_without_valid_checkpoints(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    (checkpoint_dir / "latest").mkdir()
    (checkpoint_dir / "checkpoint_old").mkdir()

    latest = _find_latest_checkpoint(str(checkpoint_dir))

    assert latest is None
