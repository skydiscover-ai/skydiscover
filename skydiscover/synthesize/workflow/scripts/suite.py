"""Run a test suite against one implementation.

A suite is a directory with a `test.sh` at its top. The harness runs it and reads the exit code;
how the tests are written, built, and run is the script's business:

    bash test.sh              run every test
    bash test.sh <file>...    run only the named test files

with `SKYDISCOVER_IMPL` set to the implementation under test (a file or a directory),
`SKYDISCOVER_INTERFACE` to the interface directory when the run has one, and the suite directory as
the working directory. Exit 0 means every test passed; any other exit means the implementation
failed a test. A run that does not finish within its time budget is no verdict at all.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

SCRIPTS = Path(__file__).resolve().parent
SYNTHESIZE = SCRIPTS.parents[1]
for _p in (SCRIPTS, SYNTHESIZE):  # these scripts run standalone: siblings and spec/ must import
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import spec  # noqa: F401  (the sibling spec/ of a source checkout)
except ModuleNotFoundError:
    # A plugin install copies workflow/ on its own; spec/ then comes from the installed package.
    try:
        import skydiscover.synthesize.spec as spec  # noqa: F401
    except ModuleNotFoundError:
        sys.exit(
            "skysynth: cannot import spec/. Install the skydiscover package "
            "(uv pip install git+https://github.com/skydiscover-ai/skydiscover) or run from a "
            "source checkout."
        )
    sys.modules["spec"] = spec

TEST_SCRIPT = "test.sh"

# Time budget for one suite run: `run finish` and the release checks (default 600).
SLOW_SECS = int(os.environ.get("SKYDISCOVER_SLOW_SECS", "600") or "600")
# Budget for one test on the reference at validation. A kept test runs every iteration, so a test
# slower than this on a plain correct implementation is a soak test, not a test.
TEST_MAX_SECS = int(os.environ.get("SKYDISCOVER_TEST_MAX_SECS", "30") or "30")


@dataclass
class Result:
    exit_code: Optional[int]  # None: did not run (no test.sh) or did not finish (timed out)
    log: str = ""
    secs: float = 0.0
    timed_out: bool = False

    @property
    def passed(self) -> bool:
        return self.exit_code == 0

    @property
    def failed(self) -> bool:
        """A verdict against the implementation: the script ran to completion and said no."""
        return self.exit_code is not None and self.exit_code != 0

    def why(self) -> str:
        if self.timed_out:
            return "timed out"
        if self.exit_code is None:
            return "did not run"
        return f"exited {self.exit_code}"


def script(suite: Path) -> Path:
    return Path(suite) / TEST_SCRIPT


def run(
    suite: Path,
    impl: Path,
    interface: Optional[Path] = None,
    names: Sequence[str] = (),
    *,
    timeout: Optional[int] = None,
    env: Optional[Dict[str, str]] = None,
) -> Result:
    """`bash test.sh <names>` in `suite`, against `impl`, within `timeout` seconds (SLOW_SECS)."""
    if timeout is None:
        timeout = SLOW_SECS
    entry = script(suite)
    if not entry.is_file():
        return Result(None, f"no {TEST_SCRIPT} in {Path(suite).resolve()}")
    run_env = dict(os.environ)
    run_env["SKYDISCOVER_IMPL"] = str(Path(impl).resolve())
    if interface is not None:
        run_env["SKYDISCOVER_INTERFACE"] = str(Path(interface).resolve())
    run_env.update(env or {})
    started = time.monotonic()
    try:
        proc = subprocess.run(
            ["bash", TEST_SCRIPT, *names],
            cwd=str(suite),
            env=run_env,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        parts = [exc.stdout or "", exc.stderr or ""]
        log = "".join(p.decode(errors="replace") if isinstance(p, bytes) else p for p in parts)
        return Result(None, log, time.monotonic() - started, timed_out=True)
    return Result(proc.returncode, proc.stdout + proc.stderr, time.monotonic() - started)


def tail(text: str, n: int = 400) -> str:
    return text[-n:].strip()


def report(res: Result) -> None:
    """Relay the script's output to stderr, so the agent reading the verdict sees which test said no."""
    if res.log.strip():
        print(res.log.rstrip(), file=sys.stderr)
