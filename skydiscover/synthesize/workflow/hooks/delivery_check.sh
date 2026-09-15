#!/usr/bin/env bash
# Delivery hook: run every kept test against the delivered candidate before the task can close.
#
# Wired as Claude Code's TaskCompleted hook (SubagentStop on Codex, an extension on pi). The task
# payload arrives on stdin; for a delivery task this runs run_tests.py and exits 2 on failure, which
# blocks completion. Other tasks exit 0.
#
# Environment the lead sets:
#   SKYDISCOVER_RUN          the run directory; every path derives from its layout (spec/paths.py)
#   SKYDISCOVER_PRODREADY=1  also run the release checks (operating point, open defects, audit)
#   SKYDISCOVER_IMPL         the delivered candidate (a file or a directory), only when synthesis/impl/ holds several

set -euo pipefail
payload="$(cat || true)"

# Block on an UNEXPECTED abort. The non-blocking exit code is 1, which is also what set -e
# uses for most failures, so any unhandled error once we know this is a checked delivery would tell
# the harness "don't block" and let an UNCHECKED delivery complete. The trap converts every such abort
# into 2 = "cannot check".
is_delivery=0
trap 'st=$?; if [ "$is_delivery" = "1" ]; then
        echo "delivery check: aborted unexpectedly (exit $st) before the tests could run; refusing to deliver unchecked." >&2
        exit 2
      fi
      exit "$st"' ERR

# A completion is a checked delivery iff the lead configured any SKYDISCOVER_* delivery variable for
# it OR the task title names a delivery. The env test comes first and cannot fail, so the reliable
# signal is latched before the fallible title parse runs.
[ -n "${SKYDISCOVER_RUN:-}${SKYDISCOVER_PRODREADY:-}${SKYDISCOVER_IMPL:-}" ] && is_delivery=1

# Title fallback. Claude Code's TaskCompleted payload names the task in `task_subject` (and its
# body in `task_description`). Without jq we match the raw payload, a superset of the title: it
# can only ever over-match, and an over-match only means an extra check.
payload_cwd=""
event=""
if command -v jq >/dev/null 2>&1; then
  # Codex's SubagentStop has no title; its last_assistant_message names the delivery instead.
  title="$(printf '%s' "$payload" | jq -r '.task_subject // .title // .task.title // .task_description // .description // .last_assistant_message // ""' 2>/dev/null)" || title="$payload"
  payload_cwd="$(printf '%s' "$payload" | jq -r '.cwd // ""' 2>/dev/null)" || payload_cwd=""
  event="$(printf '%s' "$payload" | jq -r '.hook_event_name // ""' 2>/dev/null)" || event=""
else
  echo "delivery check: jq not on PATH; matching the raw payload instead of the task title (an over-match only means an extra check)." >&2
  title="$payload"
fi
title="$(printf '%s' "$title" | tr '[:upper:]' '[:lower:]')"
# A delivery verb (deliver/ship/finaliz/submit) near a domain-neutral artifact noun, or the
# skill's own Phase 3 task title ("Phase 3: Final Deliverables").
case "$title" in
  *deliver*impl*|*deliver*implementation*|*"impl delivery"*|*"final impl"*|\
  *deliver*build*|*deliver*candidate*|*deliver*artifact*|*deliver*system*|*deliver*module*|\
  *ship*impl*|*ship*implementation*|*ship*build*|*ship*artifact*|*ship*system*|*ship*module*|\
  *finaliz*impl*|*finaliz*build*|*finaliz*artifact*|*finaliz*system*|\
  *submit*impl*|*submit*implementation*|*submit*build*|*submit*artifact*|\
  *final*deliverable*) is_delivery=1 ;;
esac
[ "$is_delivery" = "1" ] || exit 0   # an ordinary task: nothing to check

# The lead's `export SKYDISCOVER_RUN` happens in its own shell, which a hook process does not
# inherit. When the variable is absent, find the run directory the way spec/paths.py lays it out:
# <project>/<runs>/<run>/synthesis/impl (runs = $SKYDISCOVER_RUNS, else config.toml, else
# .skydiscover), looking under the payload's cwd, then the project Claude Code reports, then the
# current directory. Several candidates: the most recently modified.
runs="${SKYDISCOVER_RUNS:-$(python3 -m skydiscover.synthesize.spec.paths runs 2>/dev/null || echo .skydiscover)}"
if [ -z "${SKYDISCOVER_RUN:-}" ]; then
  for base in "$payload_cwd" "${CLAUDE_PROJECT_DIR:-}" "$PWD"; do
    case "$runs" in /*) root="$runs" ;; *) root="$base/$runs" ;; esac
    [ -n "$base" ] && [ -d "$root" ] || continue
    found="$(ls -1dt "$root"/*/synthesis/impl 2>/dev/null | head -n 1 || true)"
    if [ -n "$found" ]; then
      SKYDISCOVER_RUN="${found%/synthesis/impl}"
      export SKYDISCOVER_RUN
      echo "delivery check: SKYDISCOVER_RUN not set; checking the run at $SKYDISCOVER_RUN" >&2
      break
    fi
  done
fi

# Without the run directory nothing can be checked, so BLOCK (exit 2) rather than wave through.
# (Not ${VAR:?}: under set -u an unset var exits 1, non-blocking, the opposite of what we want.)
if [ -z "${SKYDISCOVER_RUN:-}" ]; then
  echo "delivery check: set SKYDISCOVER_RUN (the run directory) to enforce; the candidate, interface, and" >&2
  echo "      tests all derive from it. Nothing can be checked, so refusing to deliver unchecked." >&2
  exit 2
fi

# Stay under the hook's own timeout (3600 s as installed): a hook the harness kills is not a
# block, so the check must give up first, with exit 2.
args=(--run "$SKYDISCOVER_RUN" --wall-secs "${SKYDISCOVER_DELIVERY_SECS:-3500}")
[ -n "${SKYDISCOVER_PRODREADY:-}" ] && args+=(--production-ready)
[ -n "${SKYDISCOVER_IMPL:-}" ] && args+=(--impl "$SKYDISCOVER_IMPL")
# run_tests.py lives beside this hook (workflow/scripts/) and finds everything else from its own
# location, so this works from a source checkout and from an installed plugin alike.
scripts_dir="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../scripts" && pwd)"
if out="$(python3 "$scripts_dir/run_tests.py" "${args[@]}" 2>&1)"; then
  if [ "$event" = "SubagentStop" ]; then
    # Codex reads JSON here; plain text is rejected.
    printf '%s' "$out" | jq -Rs '{systemMessage: .}'
  else
    echo "$out"
  fi
  exit 0
fi
echo "$out" >&2          # exit 2: block the task and return the failures to the agent
exit 2
