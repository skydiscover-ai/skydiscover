"""Where SkyDiscover-Synthesize keeps its files. Three places, one word each:

    outputs/synthesize/<slug>_<timestamp>/ the result: best/, checkpoints/, history.json (see checkpoint.py)
    .skydiscover/<slug>/              the run: the agents' working files, one folder per phase (Run)
    ~/.skydiscover/<domain>/          knowledge base: tests, decisions, and wiki pages kept across runs (Domain)

config.toml sets `runs` and `home`; $SKYDISCOVER_RUNS and $SKYDISCOVER_HOME override them. Every
other path derives from these two, here, so no script or brief spells a layout of its own.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def files_under(root: Path) -> List[Path]:
    """The regular files under a directory (or the file itself), skipping hidden paths and
    __pycache__. Any language: what counts is that the file is there."""
    root = Path(root)
    if root.is_file():
        return [root]
    out: List[Path] = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).parts
        if any(part.startswith(".") or part == "__pycache__" for part in rel):
            continue
        if p.is_file():
            out.append(p)
    return out


# A suite is a directory with a test.sh at its top (workflow/scripts/suite.py runs it). Every other
# regular file in the directory is a test, in whatever language test.sh knows how to run; a test's
# id is its file name without the extension.
TEST_SCRIPT = "test.sh"


def is_test_file(path: Path) -> bool:
    path = Path(path)
    return path.is_file() and path.name != TEST_SCRIPT and not path.name.startswith(".")


def test_files(suite: Path) -> List[Path]:
    """The tests in a suite directory, in name order."""
    return sorted((p for p in Path(suite).glob("*") if is_test_file(p)), key=lambda p: p.name)


def test_id(path: Path) -> str:
    """Stable id from a test file name: durability.cc -> durability. A legacy check_ prefix is
    dropped so an earlier run's check_durability.cc names the same property."""
    stem = Path(path).stem
    return stem[len("check_") :] if stem.startswith("check_") else stem


try:
    import tomllib  # Python 3.11+ stdlib
except ModuleNotFoundError:  # Python 3.10 (supported per requires-python) uses the backport
    try:
        import tomli as tomllib
    except ModuleNotFoundError:  # a bare interpreter running a hook: defaults apply
        tomllib = None

_DEFAULTS = {"runs": ".skydiscover", "home": "~/.skydiscover", "outputs": "outputs/synthesize"}


# ------------------------------------------------------------------------------------ the two roots


def _config_file(root: Optional[Path] = None) -> Optional[Path]:
    """<synthesize>/config.toml, or None if it was deleted (defaults then apply)."""
    kit = root or Path(__file__).resolve().parent.parent  # spec/ -> synthesize/
    cand = kit / "config.toml"
    return cand if cand.is_file() else None


def _load(root: Optional[Path] = None) -> Dict[str, Any]:
    """The declared settings. A missing, unreadable, or malformed file reads as no settings: a config
    typo must never take a pipeline down, and every key has a working default. Unknown keys and
    wrong types are dropped."""
    path = _config_file(root)
    if path is None or tomllib is None:
        return {}
    try:
        doc = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError):
        return {}
    return {k: v for k, v in doc.items() if k in _DEFAULTS and isinstance(v, str)}


def home(root: Optional[Path] = None) -> Path:
    """Knowledge base root: $SKYDISCOVER_HOME, else `home` in config.toml, else ~/.skydiscover."""
    raw = os.environ.get("SKYDISCOVER_HOME") or _load(root).get("home") or _DEFAULTS["home"]
    return Path(raw).expanduser()


def runs(root: Optional[Path] = None) -> Path:
    """Parent of every run directory: $SKYDISCOVER_RUNS, else `runs` in config.toml, else
    .skydiscover. A relative value stays relative to the cwd, so a project keeps its runs beside
    its code."""
    raw = os.environ.get("SKYDISCOVER_RUNS") or _load(root).get("runs") or _DEFAULTS["runs"]
    return Path(raw).expanduser()


def outputs(root: Optional[Path] = None) -> Path:
    """Parent of every published result (`<outputs>/<slug>_<timestamp>/`): $SKYDISCOVER_OUTPUTS,
    else `outputs` in config.toml, else outputs/synthesize, relative to the project like `runs`."""
    raw = (
        os.environ.get("SKYDISCOVER_OUTPUTS") or _load(root).get("outputs") or _DEFAULTS["outputs"]
    )
    return Path(raw).expanduser()


def project_of(run_dir: Path) -> Optional[Path]:
    """The project a run belongs to: the directory holding the runs folder (`<project>/.skydiscover/
    <slug>/`). None when runs live at an absolute path (SKYDISCOVER_RUNS) and belong to no project.
    """
    if runs().is_absolute():
        return None
    return Path(run_dir).resolve().parent.parent


# ------------------------------------------------------------------------------------ slugs


_SLUG_OK = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def check_slug(slug: str) -> str:
    """A run slug names a directory that is later removed and exported, so it is validated:
    lowercase alphanumerics, dot, underscore, hyphen; no leading dot."""
    s = (slug or "").strip()
    if not _SLUG_OK.match(s):
        raise ValueError(
            f"bad run slug {slug!r}: use a short kebab name (lowercase letters, digits, '-', '_', "
            f"'.'), no '/' and no leading '.'"
        )
    return s


def domain_slug(text: str) -> str:
    """The one spelling of a domain: `Job Scheduler`, `job-scheduler`, and `job  scheduler` are one
    knowledge base folder. Letters and digits of any script are kept, so two domains never share a
    folder by accident; a name with none of either is an error, not a default."""
    slug = re.sub(r"[^\w]+|_+", "-", (text or "").lower()).strip("-")
    if not slug:
        raise ValueError(
            f"domain {text!r} has no letters or digits to name its knowledge base folder"
        )
    if slug in _NOT_A_DOMAIN:
        raise ValueError(f"{slug!r} is reserved under {home()}; name the domain something else")
    return slug


# ------------------------------------------------------------------------------------ one run


_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)


def _scalar(raw: str) -> str:
    """A YAML-ish scalar: a quoted value is taken whole (a `#` inside it is not a comment); an
    unquoted one ends at the first ` #`."""
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] in "\"'" and raw[-1] == raw[0]:
        return raw[1:-1]
    return re.split(r"\s+#", raw, maxsplit=1)[0].strip()


def front_matter(path: Path) -> Dict[str, str]:
    """Scalar `key: value` front matter of a markdown file; {} when there is none."""
    try:
        m = _FRONT_MATTER.match(path.read_text(encoding="utf-8"))
    except OSError:
        return {}
    if not m:
        return {}
    out: Dict[str, str] = {}
    for line in m.group(1).splitlines():
        mm = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", line)
        if mm:
            out[mm.group(1)] = _scalar(mm.group(2))
    return out


class Run:
    """The working directory of one run. Four files a person reads at the top; below them one folder
    per phase, named like the phase and holding what that phase's roles write.

        <run>/
        ├── README.md            what this directory is and where the result went
        ├── task.md              what the user asked for (front matter: domain, checked_by)
        ├── decision_log.json    every question, its answer, who decided it, and every finding
        ├── report.md            the final report, for the lead's closing message
        ├── specification/       Phase 1: sources/ references/ questions→answers→spec.json cards/
        ├── synthesis/           Phase 2: plan.md impl/ evaluator/ tests/ bench/ audit/
        └── review/              Phase 3: security.md, reliability.md, attack.md
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def __fspath__(self) -> str:
        return str(self.path)

    def __truediv__(self, other: str) -> Path:
        return self.path / other

    @property
    def slug(self) -> str:
        return self.path.name

    # top level
    @property
    def readme(self) -> Path:
        return self.path / "README.md"

    @property
    def task(self) -> Path:
        return self.path / "task.md"

    @property
    def decision_log(self) -> Path:
        return self.path / "decision_log.json"

    @property
    def report(self) -> Path:
        return self.path / "report.md"

    @property
    def output_pointer(self) -> Path:
        """Where checkpoint.py published this run; written once so `run finish` is idempotent."""
        return self.path / ".output"

    def domain(self) -> Optional[str]:
        """The knowledge base folder this run reads and feeds: the `domain:` line of task.md's front matter,
        slugged. None when the task carries none."""
        raw = front_matter(self.task).get("domain")
        return domain_slug(raw) if raw else None

    def is_proof_run(self) -> bool:
        """A formal-proof-driven run: no benchmark; the task ships the proof check as its test suite."""
        return front_matter(self.task).get("checked_by") == "proof"

    # Phase 1: specification/
    @property
    def specification(self) -> Path:
        return self.path / "specification"

    @property
    def sources(self) -> Path:
        """The reference systems examined: a real `git clone` each, or a symlink into the shared
        clone cache."""
        return self.specification / "sources"

    @property
    def references(self) -> Path:
        """Everything learned from the reference systems: one folder per source, their real tests,
        and the cross-source aggregates (axes, skeleton, literature, acquisitions)."""
        return self.specification / "references"

    @property
    def acquisitions(self) -> Path:
        """Sources reused from the wiki instead of cloned, with the page that pins each."""
        return self.references / "acquisitions.json"

    @property
    def axes(self) -> Path:
        """The domain's property axes, each with the rationale for asking it."""
        return self.references / "axes.json"

    @property
    def literature(self) -> Path:
        """Papers, RFCs, and design docs consulted."""
        return self.references / "literature.json"

    @property
    def skeleton(self) -> Path:
        """The module structure the reference systems share."""
        return self.references / "skeleton.json"

    @property
    def reference_tests(self) -> Path:
        """The reference systems' real tests behind each property, copied with a note of where each came from."""
        return self.references / "tests"

    @property
    def reference_tests_index(self) -> Path:
        return self.references / "tests.json"

    @property
    def questions(self) -> Path:
        """The property questions a from-scratch build must answer."""
        return self.specification / "questions.json"

    @property
    def answers(self) -> Path:
        """The answers, and who gave each."""
        return self.specification / "answers.json"

    @property
    def spec(self) -> Path:
        """The consolidated specification the cards are built from."""
        return self.specification / "spec.json"

    @property
    def cards(self) -> Path:
        """The specification cards the synthesis loop reads."""
        return self.specification / "cards"

    @property
    def environment_card(self) -> Path:
        """The resource that bounds the score, its measured limit, and the ceiling."""
        return self.cards / "environment.json"

    @property
    def workload_card(self) -> Path:
        """The workload measured in the domain's own terms, and the scored configuration."""
        return self.cards / "workload.json"

    @property
    def requirements_card(self) -> Path:
        """The interface, the guarantees, the required properties, and the operating point."""
        return self.cards / "requirements.json"

    @property
    def properties_card(self) -> Path:
        """Each requirement as something a test can falsify: {id, property, probe, oracle}."""
        return self.cards / "properties.json"

    # Phase 2: synthesis/
    @property
    def synthesis(self) -> Path:
        return self.path / "synthesis"

    @property
    def plan(self) -> Path:
        """The planner's file: candidate designs, the ones ruled out and why, the current brief."""
        return self.synthesis / "plan.md"

    @property
    def proof_log(self) -> Path:
        """Formal-proof-driven runs only: every proof attempt and why it failed."""
        return self.synthesis / "proof-log.md"

    @property
    def impl(self) -> Path:
        """The current candidate: one source file, or a directory built whole."""
        return self.synthesis / "impl"

    def entry_impl(self, named: str = "") -> Optional[Path]:
        """The candidate the checks test: the one `named` (--impl or $SKYDISCOVER_IMPL) when given;
        else the latest scored candidate row's `impl` when it lies under impl/; else the sole file,
        or the whole directory. None when there is nothing there."""
        if named:
            p = Path(named)
            return p if p.exists() else None
        if not self.impl.is_dir():
            return None
        sources = files_under(self.impl)
        if not sources:
            return None
        try:
            board = json.loads(self.leaderboard.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            board = None
        latest = (
            next(
                (
                    row
                    for row in reversed(board)
                    if isinstance(row, dict)
                    and row.get("role", "candidate") == "candidate"
                    and row.get("draw", "scored") == "scored"
                ),
                {},
            )
            if isinstance(board, list)
            else {}
        )
        if latest:
            cand = latest.get("impl")
            if isinstance(cand, str) and cand:
                path = Path(cand) if os.path.isabs(cand) else self.path / cand
                try:
                    path.resolve().relative_to(self.impl.resolve())
                    if path.exists():
                        return path
                except ValueError:
                    pass
        return sources[0] if len(sources) == 1 else self.impl

    @property
    def evaluator(self) -> Path:
        return self.synthesis / "evaluator"

    @property
    def interface(self) -> Path:
        """The public contract the candidate implements and the tests compile against."""
        return self.evaluator / "interface"

    @property
    def benchmark(self) -> Path:
        """The scored benchmark harness."""
        return self.evaluator / "benchmark"

    @property
    def reference(self) -> Path:
        """The trusted reference implementation every test must pass."""
        return self.evaluator / "reference"

    @property
    def mutants(self) -> Path:
        """Deliberately broken implementations every test must catch."""
        return self.evaluator / "mutants"

    @property
    def tests(self) -> Path:
        """The kept test suite: test.sh and the tests it runs."""
        return self.synthesis / "tests"

    @property
    def test_script(self) -> Path:
        return self.tests / TEST_SCRIPT

    @property
    def bench(self) -> Path:
        return self.synthesis / "bench"

    @property
    def leaderboard(self) -> Path:
        return self.bench / "leaderboard.json"

    @property
    def bench_runs(self) -> Path:
        """Raw output of each scored benchmark run."""
        return self.bench / "runs"

    @property
    def profiles(self) -> Path:
        """The bottleneck analysis of each scored candidate."""
        return self.bench / "profiles"

    @property
    def audit(self) -> Path:
        """The reward-hacking audit: evidence per confirmed hack, and the coverage stamp."""
        return self.synthesis / "audit"

    @property
    def completeness(self) -> Path:
        return self.audit / "completeness.json"

    @property
    def severity(self) -> Path:
        """The severity each finding was first written with, kept by the findings CLI, so a defect
        cannot be silently downgraded before delivery. Hidden; nobody edits it."""
        return self.path / ".severity_snapshot.json"

    # Phase 3: review/
    @property
    def review(self) -> Path:
        return self.path / "review"

    @property
    def review_security(self) -> Path:
        """The auditor's security review (production lens mode)."""
        return self.review / "security.md"

    @property
    def review_reliability(self) -> Path:
        """The auditor's reliability review (production lens mode)."""
        return self.review / "reliability.md"

    @property
    def review_attack(self) -> Path:
        """The auditor's attack-mode review: one line per attack class, broken or not."""
        return self.review / "attack.md"

    def phases(self) -> tuple[Path, Path, Path]:
        return self.specification, self.synthesis, self.review

    def create(self) -> "Run":
        """Make the directory, its three phase folders, and the README. Idempotent."""
        for d in (self.path, *self.phases()):
            d.mkdir(parents=True, exist_ok=True)
        if not self.readme.exists():
            self.readme.write_text(run_readme(self.slug), encoding="utf-8")
        return self


def run_readme(slug: str) -> str:
    runs_dir, outputs_dir = runs().as_posix(), outputs().as_posix()
    return f"""# Run `{slug}`: working files

These are the working files of the run `{slug}`. The agents read and write here while they work;
**the result is written elsewhere**:

    {outputs_dir}/{slug}_<timestamp>/best/         the result: spec.md, artifact/, tests/, score.json
    {outputs_dir}/{slug}_<timestamp>/checkpoints/  one scored candidate per iteration: artifact/, score.json, tests.json
    {outputs_dir}/{slug}_<timestamp>/history.json  written at finish: one row per checkpoint, with the tests it fails today

Start with `best/spec.md`. `run finish` deletes this directory once the result is published
(`--keep-run` keeps it). Until then it holds everything needed to continue: open your coding agent
in the project and ask it to continue the skysynth run `{slug}` from `{runs_dir}/{slug}/`.

## Map

| Path | What it is |
|---|---|
| `task.md` | what was asked for; the front matter names the domain (and `checked_by: proof` for a proof-driven run) |
| `decision_log.json` | every question asked, who answered it, and every finding, in order |
| `report.md` | the agents' final report; the closing message you receive is drawn from it |
| `specification/` | **Initial Specification.** `sources/` the reference systems examined; `references/` what was learned from them (a folder per source with `file:line` citations, their real tests, the shared skeleton); `questions.json` → `answers.json` → `spec.json`; `cards/` the specification cards: `requirements.json`, `properties.json`, and when the run has them `workload.json`, `environment.json` |
| `synthesis/` | **Synthesis Loop.** `plan.md` the designs considered and the one being built; `impl/` the current candidate (the implementation under construction); `evaluator/` its interface, benchmark, and the simple trusted reference the tests are checked against; `tests/` the kept tests and the `test.sh` that runs them; `bench/` every score measured so far; `audit/` the reward hacks the Auditor found |
| `review/` | **Final Deliverables.** The Auditor's final reviews of the selected candidate: `security.md`, `reliability.md`, `attack.md` |

"""


def run_dir(slug: str, root: Optional[Path] = None) -> Path:
    """<runs>/<slug>, validated. Does not create it; see Run.create()."""
    return runs(root) / check_slug(slug)


def resolve_decision_log(arg: str) -> Path:
    """CLI form: accept the run dir (the normal case) or a direct path to the log file."""
    p = Path(arg)
    if p.is_file() and p.suffix == ".json":
        return p
    return Run(p).decision_log


# ------------------------------------------------------------------------------------ knowledge base


class Domain:
    """The knowledge base for one domain: what earlier runs in that domain learned, in one folder.

    ~/.skydiscover/<domain>/
    ├── tests/           kept tests from finished runs, with index.json
    ├── decisions.json   the user's answers and confirmed reward hacks, saved at run finish
    └── wiki/            optional pages kb-builder writes: sources, properties, hacks, designs
    """

    def __init__(self, slug: str, root: Optional[Path] = None):
        self.slug = domain_slug(slug)
        self.path = (root or home()) / self.slug

    @property
    def tests(self) -> Path:
        return self.path / "tests"

    @property
    def tests_index(self) -> Path:
        return self.tests / "index.json"

    @property
    def decisions(self) -> Path:
        path = self.path / "decisions.json"
        old = self.path / "findings.json"  # the first releases' name; taken over on first touch
        if not path.exists() and old.is_file():
            old.rename(path)
        return path

    @property
    def wiki(self) -> Path:
        return self.path / "wiki"


# Folders under home that are not a domain (dot-folders are never one either).
_NOT_A_DOMAIN = {"shared"}


def domains(root: Optional[Path] = None) -> list[Domain]:
    base = root or home()
    if not base.is_dir():
        return []
    return [
        Domain(d.name, base)
        for d in sorted(base.iterdir())
        if d.is_dir() and d.name not in _NOT_A_DOMAIN and not d.name.startswith(".")
    ]


def shared_wiki(root: Optional[Path] = None) -> Path:
    """Wiki pages that hold for every domain (for example reward hacks stated purely in terms of a
    scored objective)."""
    return (root or home()) / "shared" / "wiki"


def source_cache(root: Optional[Path] = None) -> Path:
    """Cloned reference systems, shared across runs so a 500 MB clone lands once per machine. A
    dot-folder: tool-managed and re-downloadable, never something to read."""
    return (root or home()) / ".cache" / "sources"


# ------------------------------------------------------------------------------------ CLI


def run_cli(main, name: str) -> int:
    """Run a tool's main() from `python3 -m`, turning a predictable input error (a missing or
    malformed file, a bad value, a refused write) into one line on stderr and exit code 2 instead
    of a traceback. Tests call main() directly and still see the exception."""
    try:
        return main()
    except KeyError as e:
        print(f"{name}: malformed input, missing key {e}", file=sys.stderr)
        return 2
    except (OSError, ValueError, RuntimeError) as e:
        print(f"{name}: {e}", file=sys.stderr)
        return 2


def main(argv: Optional[list] = None) -> int:
    """python3 -m skydiscover.synthesize.spec.paths
    (nothing)          print the three roots in effect and why
    runs | outputs | home   print just that root (for shell scripts)
    run <slug>         create the run directory (with README and phase folders) and print it
    domain <text>      print the knowledge base folder for a domain, creating nothing
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("-h", "--help"):
        print(main.__doc__)
        return 0
    if args and args[0] == "run":
        if len(args) != 2:
            print("usage: paths run <slug>", file=sys.stderr)
            return 2
        try:
            print(Run(run_dir(args[1])).create().path)
        except ValueError as e:
            print(f"paths: {e}", file=sys.stderr)
            return 2
        return 0
    if args and args[0] == "domain":
        if len(args) < 2:
            print("usage: paths domain <text>", file=sys.stderr)
            return 2
        print(Domain(" ".join(args[1:])).path)
        return 0
    roots = (
        ("runs", runs, "SKYDISCOVER_RUNS"),
        ("outputs", outputs, "SKYDISCOVER_OUTPUTS"),
        ("home", home, "SKYDISCOVER_HOME"),
    )
    if args and len(args) == 1 and args[0] in dict((n, f) for n, f, _ in roots):
        print(dict((n, f) for n, f, _ in roots)[args[0]]())
        return 0
    if args:
        print(
            f"paths: unknown argument {args[0]!r} (expected nothing, `runs`, `outputs`, `home`, "
            "`run <slug>`, or `domain <text>`)",
            file=sys.stderr,
        )
        return 2
    path = _config_file()
    print(f"config file : {path if path else '(deleted; using defaults)'}")
    declared = _load()
    for name, fn, env in roots:
        src = (
            f"${env}" if os.environ.get(env) else ("config.toml" if name in declared else "default")
        )
        print(f"{name:12}: {fn()}   (from {src})")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main, "paths"))
