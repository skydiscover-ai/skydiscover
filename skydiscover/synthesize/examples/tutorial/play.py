#!/usr/bin/env python3
"""The tutorial as a terminal walk-through: the same session tutorial.ipynb drives, without Jupyter.

Each step runs against a live Claude Code session and is scored by workload.replay(); clarifying
questions take their suggested defaults. Needs Claude Code installed and logged in; a run takes
a few hours.

    python skydiscover/synthesize/examples/tutorial/play.py
"""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Inside the checkout the package imports uninstalled; copied into a project dir (Step 1), it comes
# from the environment, as the README's `python3 -c 'import skydiscover'` check requires.
for _root in HERE.parents:
    if (_root / "skydiscover" / "__init__.py").is_file():
        sys.path.insert(0, str(_root))
        break
try:
    from skydiscover.synthesize.spec.paths import outputs, runs
    from skydiscover.synthesize.spec.render import passed_final_tests
except ModuleNotFoundError:
    sys.exit(
        "play.py: `import skydiscover` fails in this shell; activate the skydiscover venv first "
        "(source <path-to-skydiscover>/.venv/bin/activate)."
    )

RUNS = runs()  # where the run lives: .skydiscover unless config.toml or $SKYDISCOVER_RUNS moves it
sys.path.insert(0, str(HERE))
from workload import CAPACITY, HOT_KEYS, SCAN_OPS, replay  # noqa: E402

# palette
_ON = sys.stdout.isatty()


def _c(code):
    return code if _ON else ""


BOLD, DIM, GREEN, RED, CYAN, YELLOW, RESET = (
    _c("\033[1m"),
    _c("\033[2m"),
    _c("\033[32m"),
    _c("\033[31m"),
    _c("\033[36m"),
    _c("\033[33m"),
    _c("\033[0m"),
)
W = 72
STAGES = ["setup", "workload", "questions", "spec", "build", "result"]
SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"  # a little braille fidget so you can see it thinking
_ANSI = re.compile(r"\033\[[0-9;]*m")


def vlen(s):
    return len(_ANSI.sub("", s))


def wrap(s, width=W - 2):
    """Wrap a styled string at word boundaries, counting visible width only."""
    out, cur = [], ""
    for word in s.split():
        if cur and vlen(cur) + 1 + vlen(word) > width:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    if cur:
        out.append(cur)
    return out or [""]


def inline(s):
    """Render inline markdown: **bold**, code, drop stray emphasis marks."""
    s = re.sub(r"\*\*(.+?)\*\*", BOLD + r"\1" + RESET, s)
    s = re.sub(r"`(.+?)`", CYAN + r"\1" + RESET, s)
    s = re.sub(r"(?<!\*)\*(?!\*)", "", s)
    return s


def render_md(text):
    """Turn a Claude reply into clean, wrapped terminal text."""
    lines = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            lines.append("")
            continue
        if re.match(r"^\s*[-*=]{3,}\s*$", line):  # horizontal rule
            continue
        h = re.match(r"^(#{1,6})\s+(.*)", line)
        if h:
            for w in wrap(BOLD + inline(h.group(2)) + RESET):
                lines.append(f"  {w}")
            continue
        b = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)", line)
        if b:
            body = wrap(inline(b.group(3)), W - 6)
            lines.append(f"   {CYAN}•{RESET} {body[0]}")
            lines += [f"     {w}" for w in body[1:]]
            continue
        for w in wrap(inline(line)):
            lines.append(f"  {w}")
    return "\n".join(lines)


# chrome
def roadmap(stage):
    """A one-line 'you are here' across the six stages."""
    cur = STAGES.index(stage)
    cells = []
    for i, name in enumerate(STAGES):
        if i < cur:
            cells.append(f"{GREEN}✓ {name}{RESET}")
        elif i == cur:
            cells.append(f"{CYAN}{BOLD}▶ {name.upper()}{RESET}")
        else:
            cells.append(f"{DIM}· {name}{RESET}")
    return "   ".join(cells)


def head(title, stage, subtitle=""):
    if _ON:
        os.system("clear")  # the step log survives in the pane and in play.log
    print(f"{BOLD}{'═' * W}{RESET}")
    print(f"{BOLD} {title}{RESET}")
    if subtitle:
        print(f" {DIM}{subtitle}{RESET}")
    print(f" {roadmap(stage)}")
    print(f"{BOLD}{'═' * W}{RESET}\n")


def para(text):
    for line in wrap(inline(text)):
        print(f"  {line}")


def bullets(items):
    for it in items:
        body = wrap(inline(it), W - 6)
        print(f"   {CYAN}•{RESET} {body[0]}")
        for line in body[1:]:
            print(f"     {line}")


def buddy(msg):
    body = wrap(msg, W - 6)
    print(f"\n  {CYAN}🤖  {body[0]}{RESET}")
    for line in body[1:]:
        print(f"      {CYAN}{line}{RESET}")


def working(msg, note="This runs for real; each line below is a live step the agents take."):
    body = wrap(msg, W - 6)
    print(f"  {YELLOW}⏳  {body[0]}{RESET}")
    for line in body[1:]:
        print(f"      {YELLOW}{line}{RESET}")
    if note:
        print(f"  {DIM}{note}{RESET}")
    print()


def footer(cta="Press Enter for the next step"):
    print(f"\n{DIM}{'─' * W}{RESET}")
    print(f"  {CYAN}[Enter]{RESET} {cta}    {DIM}·   [q] quit{RESET}")


def pause(cta="Press Enter for the next step"):
    footer(cta)
    if input("  ").strip().lower() == "q":
        print("\n  stopped.")
        sys.exit(0)


def barchart(rows):
    width = 44
    label_w = max(len(name) for name, _, _ in rows)
    for name, score, color in rows:
        n = round(score * width)
        bar = color + "█" * n + DIM + "░" * (width - n) + RESET
        print(f"  {name:<{label_w}}  {bar}  {BOLD}{score:.3f}{RESET}")


# the live session (same helper as the notebook)
SESSION = None
LOGLINES = []  # every step of the whole run, across turns: what the pane scrolls
LOGFILE = Path(
    os.environ.get("SKYSYNTH_PLAY_LOG") or HERE / "play.log"
)  # plain-text copy of the log
MAX_ROUNDS = int(
    os.environ.get("SKYSYNTH_PLAY_ROUNDS", "60") or "60"
)  # rounds before the demo gives up


def _clock(seconds):
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


def _clip(s, width):
    return s if len(s) <= width else s[: max(1, width - 1)] + "…"


def _status(state, ch):
    """The pane's top line: buddy + spinner + timer + the live step + step count."""
    lead = f"  {CYAN}🤖 {ch}{RESET} {BOLD}{_clock(time.time() - state['t0'])}{RESET}  "
    tail = f"   {DIM}{state['n']} steps{RESET}"
    room = W - vlen(lead) - vlen(tail)
    return lead + _clip(state["desc"], max(12, room)) + tail


def _log(state, text, dim=True):
    """Append one line to the run-wide log: pane, memory, and play.log."""
    line = f"{DIM}{text}{RESET}" if dim else text
    LOGLINES.append(line)
    with open(LOGFILE, "a") as fh:
        fh.write(_ANSI.sub("", line) + "\n")


def _consume(proc, state, echo=False):
    """Drain the real stream-json; append each tool step to the run log."""
    for line in proc.stdout:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        state["session"] = event.get("session_id", state.get("session"))
        kind = event.get("type")
        if kind == "assistant":
            for block in event["message"]["content"]:
                if block.get("type") == "tool_use":
                    what = block["input"].get("description") or block["input"].get("summary")
                    if what:
                        state["n"] += 1
                        state["desc"] = what
                        stamp = _clock(time.time() - state["t0"])
                        text = f"  {stamp:>5}  {state['n']:>3}· {wrap(what, W - 14)[0]}"
                        _log(state, text)
                        if echo:
                            print(LOGLINES[-1])
        elif kind == "result":
            state["reply"] = event.get("result", "")
            # keep reading: background roles finish before the process exits
    proc.wait()
    state["done"] = True


def _spinner(state):
    """Fallback when the pane cannot run: just the live status line, animated."""
    frame = 0
    while not state["done"]:
        frame += 1
        sys.stdout.write("\r\033[K" + _status(state, SPIN[frame % len(SPIN)]))
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


def _logpane(state):
    """Scrollable log pane while the turn runs: arrows scroll, PgUp/PgDn page, f or End follows the tail."""
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    view = max(6, min(16, shutil.get_terminal_size().lines - 10))
    H = view + 4  # status, top rule, view log rows, bottom rule, controls
    top, follow = 0, True
    frame = 0
    sys.stdout.write("\n" * H)  # reserve the pane; cursor sits just below it
    sys.stdout.flush()
    try:
        tty.setcbreak(fd)
        while True:
            done = state["done"]  # read once: render one last frame after it flips
            frame += 1
            keys = b""
            if select.select([sys.stdin], [], [], 0)[0]:
                keys = os.read(fd, 32)
            max_top = max(0, len(LOGLINES) - view)
            if not follow:
                top = min(top, max_top)
            if b"\x1b[A" in keys:  # up: stop following, look back
                top, follow = max(0, (max_top if follow else top) - 1), False
            if b"\x1b[5~" in keys:  # PgUp
                top, follow = max(0, (max_top if follow else top) - view), False
            if b"\x1b[B" in keys and not follow:  # down
                top = min(max_top, top + 1)
                follow = top >= max_top
            if b"\x1b[6~" in keys and not follow:  # PgDn
                top = min(max_top, top + view)
                follow = top >= max_top
            if b"f" in keys or b"\x1b[F" in keys or b"\x1b[4~" in keys:  # follow again
                follow = True
            if follow:
                top = max_top

            rows = [_status(state, SPIN[frame % len(SPIN)])]
            rows.append(f"  {DIM}{'┄' * W}{RESET}")
            for i in range(top, top + view):
                rows.append(LOGLINES[i] if i < len(LOGLINES) else "")
            rows.append(f"  {DIM}{'┄' * W}{RESET}")
            where = (
                f"{GREEN}following{RESET}"
                if follow
                else f"{YELLOW}viewing {min(top + view, len(LOGLINES))}/{len(LOGLINES)}{RESET}"
            )
            rows.append(f"  {DIM}[↑/↓] scroll   [PgUp/PgDn] page   [f] follow{RESET}   {where}")

            sys.stdout.write(f"\033[{H}A")  # back to the top of the pane
            for ln in rows:
                sys.stdout.write("\r\033[K" + ln + "\n")
            sys.stdout.flush()
            if done:
                break
            time.sleep(0.08)
    except Exception:
        _spinner(state)  # never let the pane break a real run
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write(f"\033[{H}A\033[J")  # erase the pane, leave the frame clean
        sys.stdout.flush()


def talk(message, progress=True, render=True, verb="working", final=True):
    """Send one message to the Claude Code session, streaming each tool step into the log pane and play.log."""
    global SESSION
    cmd = [
        "claude",
        "-p",
        message,
        "--model",
        os.environ.get(
            "SKYSYNTH_PLAY_MODEL", "claude-sonnet-5"
        ),  # hours of agent calls at Sonnet pricing
        "--output-format",
        "stream-json",
        "--verbose",
        "--dangerously-skip-permissions",
    ]  # the agents run builds and benchmarks; sandbox accordingly
    if SESSION:
        cmd += ["--resume", SESSION]
    # wait for background roles instead of killing them after 10 min; keep our stdin out of the prompt
    env = {"CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS": "0", **os.environ}
    proc = subprocess.Popen(
        cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True, env=env
    )
    state = {
        "desc": f"{verb}...",
        "n": 0,
        "done": False,
        "reply": "",
        "session": SESSION,
        "t0": time.time(),
    }
    _log(state, f"  ── {verb} ──")
    if progress and _ON and sys.stdin.isatty():
        consumer = threading.Thread(target=_consume, args=(proc, state), daemon=True)
        consumer.start()
        _logpane(state)
        consumer.join()
    else:  # no keyboard or piped output: plain streaming lines instead of a pane
        _consume(proc, state, echo=progress)
    SESSION, reply, steps = state.get("session", SESSION), state.get("reply", ""), state["n"]
    if proc.returncode != 0:
        sys.exit(
            f"claude exited with {proc.returncode}; its message is above ({LOGFILE.name} has the steps)."
        )
    if final and progress:
        print(
            f"  {GREEN}✓{RESET} {BOLD}{_clock(time.time() - state['t0'])}{RESET}  "
            f"{verb} done   {DIM}{steps} steps · full log: {LOGFILE.name}{RESET}"
        )
    if render and reply:
        print()
        print(render_md(reply))
    return reply


# scoring helpers (same logic as the notebook)
def interface_operations(interface):
    """The operation signatures in a spec's `interface`, whichever shape the spec-builder wrote:
    a {"operations": [...]} block, a {name: signature} mapping (what real runs write), or a list."""
    if isinstance(interface, dict):
        ops = interface.get("operations")
        if isinstance(ops, list):
            return [str(op) for op in ops]
        return [str(name) for name in interface]
    return [str(op) for op in (interface or [])]


IMPL_MODULE = "impl_under_test"  # the name the tutorial's tests import the candidate as


def load_candidate(path, *search_paths):
    """Import an implementation the way the harness does, so what you score is what the tests saw.

    `path` is a file (one module), a directory with `__init__.py` (a package: every file in it,
    relative imports intact), or a directory holding exactly one `.py` file. It is imported under
    the harness's module name (`impl_under_test`), replacing any earlier candidate, so a package
    that says `from impl_under_test.x import y` works too. `search_paths` (the run's pinned
    interface, for instance) are made importable first.
    """
    path = Path(path).resolve()
    if path.is_dir() and not (path / "__init__.py").is_file():
        files = [p for p in path.rglob("*.py") if not p.name.startswith("__")]
        if len(files) != 1:
            raise ValueError(
                f"{path}: a directory candidate needs an __init__.py or exactly one .py file "
                f"(found {len(files)})"
            )
        path = files[0]
    for extra in search_paths:
        if extra.is_dir() and str(extra) not in sys.path:
            sys.path.insert(0, str(extra))
    for name in [m for m in sys.modules if m == IMPL_MODULE or m.startswith(IMPL_MODULE + ".")]:
        del sys.modules[name]
    if path.is_dir():
        spec = importlib.util.spec_from_file_location(
            IMPL_MODULE, path / "__init__.py", submodule_search_locations=[str(path)]
        )
    else:
        spec = importlib.util.spec_from_file_location(IMPL_MODULE, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[IMPL_MODULE] = mod
    spec.loader.exec_module(mod)
    return mod


def delivered(since=0):
    """The results whose final tests passed, oldest first (a saved candidate alone does not count)."""
    finished = []
    for path in outputs().glob("*/history.json"):
        try:
            rows = json.loads(path.read_text())
            score = json.loads((path.parent / "best/score.json").read_text())
            if path.stat().st_mtime >= since and any(
                passed_final_tests(row) and row.get("checkpoint") == score.get("checkpoint")
                for row in rows
            ):
                finished.append(path.parent / "best")
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return sorted(finished, key=lambda best: (best.parent / "history.json").stat().st_mtime)


def load_winner(best):
    """The `create_cache` of a published result: `best/artifact/` at the entry the run recorded."""
    artifact = Path(best) / "artifact"
    record = Path(best) / ".verification/record.json"
    entry = artifact
    if record.exists():
        entry = (artifact / json.loads(record.read_text()).get("entry", ".")).resolve()
        if not entry.is_relative_to(artifact.resolve()) or not entry.exists():
            raise ValueError("The recorded entry is missing or outside artifact/.")
    return load_candidate(entry, artifact).create_cache


def current_candidate():
    """The implementation the run is working on right now: the newest run's `synthesis/impl/`,
    at the entry its leaderboard last scored (or the sole file / package there). None before the
    first candidate exists."""
    from skydiscover.synthesize.spec.paths import Run

    for run_dir in sorted(RUNS.glob("*/synthesis/impl"), key=lambda p: -p.stat().st_mtime):
        run = Run(run_dir.parents[1])
        entry = run.entry_impl()
        if entry is not None:
            return run, entry
    return None


def best_so_far():
    """Score the run's current candidate on your own workload; None if it is not runnable yet."""
    found = current_candidate()
    if found is None:
        return None
    run, entry = found
    try:
        return replay(load_candidate(entry, run.interface).create_cache)["app_hit_rate"]
    except Exception:  # mid-edit, or not runnable yet
        return None


class FIFO:
    def __init__(self, capacity):
        self.capacity, self.data = capacity, OrderedDict()

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        if key not in self.data and len(self.data) >= self.capacity:
            self.data.popitem(last=False)
        self.data[key] = value

    def size(self):
        return len(self.data)


class LRU(FIFO):
    def get(self, key):
        if key in self.data:
            self.data.move_to_end(key)
        return self.data.get(key)


# the walk-through
def main():
    sys.stdout.reconfigure(line_buffering=True)
    os.chdir(HERE)
    started_at = time.time()
    if shutil.which("claude") is None:
        sys.exit("Claude Code is not installed. See https://docs.anthropic.com/en/docs/claude-code")
    LOGFILE.write_text("")  # one fresh log per demo run

    head(
        "Welcome: build a cache tailored to your workload",
        "setup",
        "a real run, one step at a time",
    )
    para(
        "A cache keeps recently used data close so your app does not fetch it again. "
        "Off-the-shelf caches (FIFO, LRU) are built for everyone, so they fit no one "
        "perfectly. Here an AI team builds one for **your exact traffic**."
    )
    print(f"\n  {BOLD}What will happen:{RESET}")
    bullets(
        [
            "You look at your workload, the traffic the cache must handle.",
            "You send **one sentence** describing what you want.",
            "The team asks a few short questions, **one at a time**. The demo takes the default it suggests for each.",
            "It builds, benchmarks, and audits for a while. **Every step lands in a log you can scroll (↑/↓) while it runs.**",
            "You see the result: your cache next to FIFO and LRU.",
        ]
    )
    buddy(
        "The demo answers the questions with the defaults; you sit back and watch. Press Enter to start."
    )
    pause("Start")

    head("Setting up", "setup", "installing the AI team into this folder (one time)")
    working("Wiring the SkySynth agents into this folder.", note="")
    r = subprocess.run(
        ["bash", str(HERE.parent.parent / "scripts/install.sh"), str(HERE)],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        sys.exit(r.stdout + r.stderr)
    para(next((ln for ln in r.stdout.splitlines() if "Linked" in ln), "installed").strip())
    buddy("Done. That was automatic. Press Enter to see the workload.")
    pause()

    head("Your workload", "workload", "the traffic your cache has to survive")
    para(
        f"Your app keeps asking for the same ~{HOT_KEYS} **hot keys**. But every so often a "
        f"batch job **scans** {SCAN_OPS} one-time keys it will never ask for again:"
    )
    print()
    print(f"    app wants these, again and again   {GREEN}●  ●  ●  ●  ●  ●  ●  ●{RESET}   hot keys")
    print(
        f"    then a scan floods one-time keys   {RED}▒  ▒  ▒  ▒  ▒  ▒  ▒  ▒{RESET}   never reused"
    )
    print(
        f"    the cache holds only {CAPACITY} slots  →  {RED}the scan shoves the hot keys OUT{RESET}"
    )
    print()
    para(
        "So right after every scan, the app cannot find its own keys and has to refetch "
        "them. The score we care about is **hit rate**: how often the app finds its key "
        "already in the cache. Higher is better; 1.0 is perfect."
    )
    fifo, lru = replay(FIFO)["app_hit_rate"], replay(LRU)["app_hit_rate"]
    buddy(
        f"Measured just now on your workload: FIFO {fifo:.3f}, LRU {lru:.3f}. Both miss "
        f"about {(1 - fifo) * 100:.0f}% of the time because every scan wipes them. "
        f"That is the bar to beat."
    )
    pause()

    head("You type: one sentence", "questions", "the demo answers its questions with the defaults")
    para("That is the entire brief. You send this one sentence:")
    print()
    print(f"  {GREEN}Build me a fast cache for the workload in workload.py.{RESET}")
    print()
    para(
        "The team studies your workload, then asks a few short questions **one at a "
        "time**, each with a sensible default. In a real run you answer in plain "
        "words; this demo accepts each default so it can run hands-off."
    )
    working("Sending it. The team studies your workload, then asks its first question.")
    reply = talk(
        "/skysynth Build me a fast cache for the workload in workload.py. "
        "Before writing anything, ask me your clarifying questions ONE AT A TIME: ask a "
        "single short question with a sensible default I can accept, wait for my answer, "
        "then ask the next. When you have no more questions, write the spec to disk and "
        "then reply with exactly: READY TO BUILD.",
        render=False,
        verb="studying your workload",
    )

    q = 0
    while "READY TO BUILD" not in reply.upper() and q < 8:
        q += 1
        head(f"Question {q}", "questions", "the demo takes the suggested default")
        print(render_md(reply))
        print()
        answer = "go with your suggested default"
        print(f"  {CYAN}answer (auto) ›{RESET} {answer}")
        print()
        working("Sending the default. Next question, or it starts building.", note="")
        reply = talk(answer, render=False, verb="thinking")

    head("All answered", "questions", "the answers are now pinned as a contract")
    buddy("That is everything the team needed. From here it is hands-off. Press Enter.")
    pause("See the plan")

    head("The plan", "spec", "the answers, pinned as a contract that is checked every round")
    specs = sorted(RUNS.glob("*/specification/spec.json"))
    if not specs:  # spec not on disk yet: let the turn that writes it finish
        working("Finalizing the spec (the contract the cache must satisfy).")
        talk("continue", render=False, verb="writing the spec")
        specs = sorted(RUNS.glob("*/specification/spec.json"))
    spec = json.loads(specs[0].read_text())
    para(f"**{spec['title'].replace(' — ', ': ').replace('—', '-')}**")
    print(f"\n  {DIM}The interface your cache will expose:{RESET}")
    bullets([op.split("  (")[0] for op in interface_operations(spec.get("interface"))])
    buddy("With the contract fixed, the real work starts: build, measure, repeat. Press Enter.")
    pause("Start the build")

    head(
        "Building your cache", "build", "the long part, a few hours. leave it running and check in."
    )
    working(
        "The team designs a cache, benchmarks it, tries to beat its own best, and audits it "
        "for reward hacks. After each round it is scored on YOUR workload, so you watch it improve."
    )
    turn = 0
    while not delivered(since=started_at):
        if turn >= MAX_ROUNDS:
            para(
                f"No result after {MAX_ROUNDS} rounds; the run's working files are under "
                f"{RUNS} and {LOGFILE} has every step. Raise SKYSYNTH_PLAY_ROUNDS to keep going."
            )
            return
        turn += 1
        talk("continue", render=False, verb=f"round {turn}", final=False)
        score = best_so_far()
        if score is not None:
            n = round(score * 40)
            print(
                f"  {DIM}round {turn:>2}{RESET}  {GREEN}{'█' * n}{DIM}{'░' * (40 - n)}{RESET}  {BOLD}{score:.3f}{RESET}"
            )
        else:
            print(f"  {DIM}round {turn:>2}  designing and testing, no scored cache yet{RESET}")
    best = delivered(since=started_at)[-1]
    print()
    para(
        "If the number stopped rising near the end, the team was not idle: it attacks its "
        "own best cache to find bugs before handing it over."
    )
    print(f"\n  {GREEN}delivered ›{RESET} {best}")
    pause("See your result")

    head("Your result", "result", "your cache next to the ones you would have used instead")
    try:
        create = load_winner(best)
    except (OSError, ValueError, ImportError, AttributeError) as exc:
        para(f"The cache entry could not be loaded: {exc}. Inspect {best}.")
        return
    yours = replay(create)["app_hit_rate"]
    fresh = replay(create, seed=123)["app_hit_rate"]
    fifo = replay(FIFO)["app_hit_rate"]
    print()
    barchart(
        [
            ("FIFO", fifo, DIM),
            ("LRU", replay(LRU)["app_hit_rate"], DIM),
            ("your cache", yours, GREEN),
        ]
    )
    print()
    para(
        f"Higher is better. Your cache finds the app's data **{yours * 100:.0f}% of the "
        f"time**, versus {fifo * 100:.0f}% for FIFO and LRU, about **{yours / fifo:.1f}x "
        f"fewer** trips to the backing store."
    )
    para(
        f"It also scores **{fresh:.3f}** on a trace it has never seen, so it learned the "
        f"pattern, not the test."
    )
    buddy(
        "Change the trace in workload.py and run again: a different workload gets a "
        "different cache. That is the Just-in-Time idea."
    )
    pause("Finish")

    head("Done", "result", "the evidence is on disk next to your cache")
    para(
        "You gave one sentence and a few answers; out came a cache built and proven for "
        "your workload. Next to it you will find **spec.md** (what was asked, what was "
        "decided, and the test that checks each answer), the tests, and its score "
        "against FIFO and LRU."
    )
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  stopped.")
