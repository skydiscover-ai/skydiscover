"""agent-pipeline/memory.py -- Context-compaction for long-running codex sessions.

Rationale: codex 0.120.0 does not reliably auto-compact across
`codex exec resume <thread_id>` boundaries — session history accumulates
unboundedly and every resume replays it, eventually triggering TPM-quota
429s on the OpenAI account. See runs/.../iter_0NN/agent_log.txt for
the failure shape (4-line log ending in "exceeded retry limit, 429").

Mitigation (this module): when a session's last-turn `input_tokens` exceed
TOKEN_THRESHOLD, we one-shot the dying thread with a Claude-style compaction
prompt, save the resulting summary to iter_NNN/summary_<role>.md, then clear
the thread_id so the next call starts a fresh session. The summary is
prepended to the fresh session's next prompt so work continues seamlessly.

Design notes:
 * Schema is deliberately generic (not role-specific). The agent already
   knows what it does from its system prompt; the summary just preserves
   state across the session break.
 * Summary points to paths; it does NOT inline file contents. The fresh
   session can tool-read whatever it needs — that's codex's native mode
   and keeps the summary small.
 * Orchestrator-written ledger + best pointer (LEDGER.md / BEST.md, handled
   elsewhere) back up the summary — if the agent's summary is thin, the
   deterministic ledger still survives.
"""

import glob
import json
import os
import re
import subprocess

TOKEN_THRESHOLD = 200_000

SUMMARY_PROMPT = """\
Your session has grown large. Write a compact handoff summary so a fresh
session can continue this exact work without losing progress.

Cover, in Markdown:

1. **Primary task** — what you are trying to accomplish and against what
   target / baseline, with the key constraints.
2. **Key decisions and reasoning** — the concrete choices made so far and
   why. Cite specifics (values, file:line) where useful.
3. **Files and artifacts** — absolute paths to the current implementation,
   latest bench output, latest critique / audit artifacts, spec cards.
   Point; do not inline contents.
4. **Recent work and outcomes** — what the last several iterations tried
   and what happened, with enough detail that the successor can judge
   whether to continue a direction or pivot.
5. **Open hypothesis / live plan** — what you were about to do, and why.
6. **Next concrete action** — the single most valuable move for the
   successor to take first.

Target length: ~20-40 thousand tokens. Write as if your successor knows
nothing about the prior turns and must get oriented from this alone, but
CAN read any file you point at if they need more depth.
"""


def _codex_session_file(thread_id: str) -> str | None:
    """Locate the rollout JSONL for a given codex thread_id.

    Codex stores sessions under ~/.codex/sessions/YYYY/MM/DD/rollout-*<tid>*.jsonl.
    We glob across recent dates; the thread_id is globally unique.
    """
    if not thread_id:
        return None
    base = os.path.expanduser("~/.codex/sessions")
    matches = glob.glob(f"{base}/*/*/*/rollout-*{thread_id}*.jsonl")
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def _last_input_tokens(session_file: str) -> int | None:
    """Return a recent per-turn input_tokens value from a codex session.

    Codex emits multiple `input_tokens` values per logical turn: the main
    LLM call (e.g. 24M) plus small follow-up events (e.g. 180K response
    polling). Taking the LAST match underestimates true context. We take
    the MAX over the tail instead — any single "main" call that would
    trigger rotation will surface.

    The 2MB tail captures many recent turns, so even if the last few turns
    failed (empty usage), an earlier real turn still contributes.
    """
    try:
        size = os.path.getsize(session_file)
        with open(session_file, "rb") as f:
            f.seek(max(0, size - 2_000_000))
            tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    best = 0
    for m in re.finditer(r'"input_tokens":(\d+)', tail):
        val = int(m.group(1))
        if val > best:
            best = val
    return best if best > 0 else None


def should_rotate(thread_id: str | None, threshold: int = TOKEN_THRESHOLD) -> bool:
    """True iff the session's last-turn input_tokens exceed the threshold."""
    if not thread_id:
        return False
    sf = _codex_session_file(thread_id)
    if not sf:
        return False
    n = _last_input_tokens(sf)
    return n is not None and n > threshold


def _extract_agent_message(log_path: str) -> str:
    """Concatenate agent_message texts from a codex --json log."""
    parts = []
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") != "item.completed":
                    continue
                item = event.get("item") or {}
                if item.get("type") != "agent_message":
                    continue
                text = (item.get("text") or "").strip()
                if text:
                    parts.append(text)
    except OSError:
        pass
    return "\n\n".join(parts)


def rotate_codex(workspace: str, model: str, thread_id: str,
                 iter_dir: str, role: str) -> str | None:
    """One-shot the dying thread for a summary, save to iter_dir, return it.

    Returns the summary string on success, or None if the rotation call
    itself failed (e.g., the session is already too poisoned to respond).
    Caller is responsible for clearing its thread_id on success.
    """
    log_path = os.path.join(iter_dir, f"summary_{role}_log.txt")
    out_path = os.path.join(iter_dir, f"summary_{role}.md")
    cmd = ["codex", "exec",
           "--model", model,
           "--sandbox", "danger-full-access",
           "-C", workspace,
           "--skip-git-repo-check",
           "--json",
           "resume", thread_id, SUMMARY_PROMPT]
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, stdin=subprocess.DEVNULL,
                              stdout=lf, stderr=subprocess.STDOUT,
                              check=False)
    if proc.returncode != 0:
        return None
    text = _extract_agent_message(log_path)
    if not text:
        return None
    with open(out_path, "w") as f:
        f.write(text)
    return text


def maybe_rotate_codex(runner_obj, attr_name: str, workspace: str, model: str,
                       iter_dir: str, role: str) -> str | None:
    """Check + rotate in one call. Clears runner_obj.<attr_name> on success.

    Returns the summary text if rotation fired (caller should prepend to
    the next prompt), else None. Safe to call every iter.
    """
    tid = getattr(runner_obj, attr_name, None)
    if not should_rotate(tid):
        return None
    print(f"  [summary] rotating {role} session (>{TOKEN_THRESHOLD//1000}K tokens)...")
    memo = rotate_codex(workspace, model, tid, iter_dir, role)
    if memo is None:
        print(f"  [summary] rotation call failed for {role}; keeping old thread")
        return None
    setattr(runner_obj, attr_name, None)
    print(f"  [summary] {role} rotated; memo={len(memo)} chars at "
          f"{os.path.relpath(os.path.join(iter_dir, f'summary_{role}.md'), iter_dir)}")
    return memo
