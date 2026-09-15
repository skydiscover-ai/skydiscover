"""validate_test.py: the one gate a test passes through. It runs `test.sh <file>` against the
reference and each mutant and reads exit codes; it knows nothing about the test's language."""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

SCRIPTS = (
    Path(__file__).resolve().parents[2] / "skydiscover" / "synthesize" / "workflow" / "scripts"
)


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_vt_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


validate_test = _load("validate_test")

# A test passes iff the implementation file holds the test's word.
TEST_SH = textwrap.dedent("""\
    #!/usr/bin/env bash
    cd "$(dirname "$0")"
    for t in "$@"; do grep -q "$(cat "$t")" "$SKYDISCOVER_IMPL" || exit 1; done
    """)


def _suite(tmp_path, word="durable"):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text(TEST_SH)
    test = tests / "durability.t"
    test.write_text(word)
    ref = tmp_path / "reference.txt"
    ref.write_text("durable fast")
    mutant = tmp_path / "mutant.txt"
    mutant.write_text("fast")
    return test, ref, mutant


def _run(*argv):
    return validate_test.main([str(a) for a in argv])


def test_a_test_that_passes_the_reference_and_catches_the_mutant_is_kept(tmp_path, capsys):
    test, ref, mutant = _suite(tmp_path)
    assert _run("--test", test, "--reference", ref, "--mutant", mutant) == 0
    assert "SOUND TEST" in capsys.readouterr().out


def test_a_test_that_catches_no_mutant_is_rejected(tmp_path, capsys):
    test, ref, mutant = _suite(tmp_path, word="fast")  # both hold "fast"
    assert _run("--test", test, "--reference", ref, "--mutant", mutant) == 1
    assert "catches no mutant" in capsys.readouterr().err


def test_a_test_the_reference_fails_cannot_be_validated(tmp_path, capsys):
    test, ref, mutant = _suite(tmp_path, word="absent")
    assert _run("--test", test, "--reference", ref, "--mutant", mutant) == 2
    assert "CANNOT VALIDATE" in capsys.readouterr().err


def test_seed_needs_the_reference_only(tmp_path, capsys):
    test, ref, mutant = _suite(tmp_path, word="fast")  # would catch no mutant
    assert _run("--test", test, "--reference", ref, "--seed") == 0
    assert "SEED TEST" in capsys.readouterr().out
    test.write_text("absent")
    assert _run("--test", test, "--reference", ref, "--seed") == 2


def test_seed_and_mutant_do_not_mix_and_one_of_them_is_needed(tmp_path, capsys):
    test, ref, mutant = _suite(tmp_path)
    assert _run("--test", test, "--reference", ref, "--seed", "--mutant", mutant) == 2
    assert "takes no --mutant" in capsys.readouterr().err
    assert _run("--test", test, "--reference", ref) == 2
    assert "need at least one --mutant" in capsys.readouterr().err


def test_a_mutant_that_only_times_out_is_not_caught(tmp_path, capsys):
    test, ref, mutant = _suite(tmp_path)
    (tmp_path / "tests" / "test.sh").write_text(
        '#!/usr/bin/env bash\ncase "$SKYDISCOVER_IMPL" in *mutant.txt) sleep 5;; esac\nexit 0\n'
    )
    assert _run("--test", test, "--reference", ref, "--mutant", mutant, "--timeout", 1) == 1
    assert "catches no mutant" in capsys.readouterr().err


def test_a_slow_test_is_rejected(tmp_path, capsys):
    test, ref, mutant = _suite(tmp_path)
    (tmp_path / "tests" / "test.sh").write_text("#!/usr/bin/env bash\nsleep 5\n")
    assert _run("--test", test, "--reference", ref, "--mutant", mutant, "--timeout", 1) == 1
    assert "took over 1s" in capsys.readouterr().err


def test_a_suite_without_test_sh_cannot_validate(tmp_path, capsys):
    test, ref, mutant = _suite(tmp_path)
    (tmp_path / "tests" / "test.sh").unlink()
    assert _run("--test", test, "--reference", ref, "--mutant", mutant) == 2
    assert "no test.sh" in capsys.readouterr().err


def test_test_sh_gets_the_file_name_the_impl_and_the_interface(tmp_path):
    test, ref, mutant = _suite(tmp_path)
    iface = tmp_path / "iface"
    iface.mkdir()
    (tmp_path / "tests" / "test.sh").write_text(
        '#!/usr/bin/env bash\necho "$1 $SKYDISCOVER_IMPL $SKYDISCOVER_INTERFACE" >> "$(dirname "$0")/seen"\n'
        'case "$SKYDISCOVER_IMPL" in *mutant*) exit 1;; esac\n'
    )
    assert _run("--test", test, "--reference", ref, "--mutant", mutant, "--interface", iface) == 0
    seen = (tmp_path / "tests" / "seen").read_text().splitlines()
    assert seen == [
        f"durability.t {ref.resolve()} {iface.resolve()}",
        f"durability.t {mutant.resolve()} {iface.resolve()}",
    ]
