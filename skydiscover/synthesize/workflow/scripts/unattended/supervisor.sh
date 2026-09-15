#!/usr/bin/env bash
# Keep one tmux-hosted claude run driving itself: approve permission prompts, resume a dead run
# from disk, and nudge a stalled one (no new on-disk artifact for a while) to checkpoint and continue.
# Never restarts from scratch and never acts while a human is in the pane. Detection keys off the
# pane's own process tree, not a global process match or the footer text. Exits once `run
# finish` leaves .skydiscover/<slug>.done, or after MAX_RESTARTS restarts without progress. Provider env, if any, comes
# from "<runs>/env.sh" (the runs folder: .skydiscover by default).
#
# Usage: supervisor.sh <tmux-session> <run-dir> [tmux-socket]
#   <run-dir>      the root that contains .skydiscover/<slug>/
#   <tmux-socket>  default ${TMPDIR:-/tmp}/skydiscover.<uid>.sock
set -uo pipefail

SESS="${1:?usage: supervisor.sh <session> <run-dir> [socket]}"
RUNDIR="${2:?usage: supervisor.sh <session> <run-dir> [socket]}"
SOCK="${3:-${TMPDIR:-/tmp}/skydiscover.$(id -u).sock}"

POLL="${SKYDISCOVER_SUP_POLL:-60}"   # seconds between checks
STALL_STRIKES=25     # checks at an idle prompt with no new artifact before the run is nudged
GEN_STALL_STRIKES=45 # checks while generating with no new artifact before the generation is cut;
                     # a coding agent legitimately generates for many minutes, so the number is larger
SAT_STALL_STRIKES=12 # checks for a lead whose context is full ("100% context used") before it is compacted
DEAD_STRIKES=2       # checks with no claude process before the run is resumed
FRESH_LEAD_AFTER=2   # compacts with no new artifact between them before a fresh lead is started;
                     # a lead that compaction cannot recover refills its context at once
CODING_AGENT_MAX_MIN=30   # a single coding agent running this long with no new code artifact is treated like a
                     # saturated lead: compacted, then cut and re-driven by a fresh lead
CPU_BUSY=20          # a non-claude process above this %CPU in the run's tree counts as heavy compute
MAX_RESTARTS="${SKYDISCOVER_SUP_MAX_RESTARTS:-3}"
                     # consecutive restarts with no on-disk progress between them before the supervisor
                     # stops and exits with a diagnostic; any new artifact resets the count

RUNS_REL="${SKYDISCOVER_RUNS:-$(python3 -m skydiscover.synthesize.spec.paths runs 2>/dev/null || echo .skydiscover)}"
case "$RUNS_REL" in /*) RUNS="$RUNS_REL" ;; *) RUNS="$RUNDIR/$RUNS_REL" ;; esac   # where the runs live (spec/paths.py runs())
LOG="$RUNS/supervisor.$SESS.log"
STOP="$RUNS/supervisor.$SESS.stop"   # terminal marker; keepalive's watchdog honors it
mkdir -p "$(dirname "$LOG")" 2>/dev/null || true
say(){ echo "[$(date +%F' '%T)] $*" >> "$LOG"; }

# Terminal condition. A finished run looks the same as a stuck one (no new artifacts, an idle prompt),
# so completion is the .skydiscover/<slug>.done marker `run finish` leaves after deleting the run
# directory. Only a marker newer than this session's start counts, so an earlier run's marker in the
# same project does not end a new one. finish also drops a stop marker so the keepalive watchdog
# does not relaunch the supervisor.
START="$RUNS/supervisor.$SESS.start"
[ -f "$START" ] || : > "$START"
run_done(){ [ -n "$(find "$RUNS" -maxdepth 1 -name '*.done' -newer "$START" 2>/dev/null | head -1)" ]; }
finish(){   # $1 = log line. Always exit 0: a nonzero exit would make systemd Restart= relaunch us.
  say "$1"
  : > "$STOP" 2>/dev/null || true
  rm -f "$START"
  exit 0
}

# Generic resume/continue prompt: point the agent at its OWN on-disk state. No domain text. The last
# sentence is what lets the supervisor ever stop; it is the only completion signal it can trust.
CONT="Continue the in-progress skysynth run from disk. Read the newest $RUNS_REL/<slug>/ directory (its README.md, task.md, decision log, specification/, and synthesis/bench/leaderboard.json) and resume EXACTLY where it left off; do NOT restart from scratch or re-create the layout. Work in small tested steps and checkpoint progress to the decision log as you go; fix+test every real defect; prod-ready is blocked while any defect is open; honest verified numbers only. The run is complete when the report is written and \`run finish <run> --export-to .\` exits 0: it publishes the result, deletes the run directory, and leaves $RUNS_REL/<slug>.done, which is the completion signal. If no $RUNS_REL/<slug>/ exists but a $RUNS_REL/<slug>.done does, the run is finished; do nothing. Hands-off."

tm(){ tmux -S "$SOCK" "$@"; }  # every tmux call on the run's socket

send_prompt(){ # send a multi-line prompt then submit (double Enter clears the paste-collapse)
  tm send-keys -t "$SESS" -l "$1"; sleep 1
  tm send-keys -t "$SESS" Enter;  sleep 1
  tm send-keys -t "$SESS" Enter
}

start_agent(){ # ensure a session exists and (re)launch claude, sourcing optional deployment env
  tm has-session -t "$SESS" 2>/dev/null || tm new-session -d -s "$SESS" -c "$RUNDIR"
  local rd_q runs_q; rd_q="$(printf '%q' "$RUNDIR")"; runs_q="$(printf '%q' "$RUNS")"   # shell-escape (a quote/space can't inject)
  tm send-keys -t "$SESS" \
    "cd $rd_q; [ -f $runs_q/env.sh ] && . $runs_q/env.sh; claude --dangerously-skip-permissions" Enter
  sleep 15
}

# The pane's process tree: the pane_pid and every descendant, one PID per line.
# This is what makes death/stall detection run-scoped instead of box-wide.
proc_tree(){
  local p="$1" c
  [ -n "$p" ] || return 0
  printf '%s\n' "$p"
  for c in $(pgrep -P "$p" 2>/dev/null); do proc_tree "$c"; done
}

# Compact the lead's context, then tell it to continue from disk. Shared by the idle-stall and
# hung-generation paths, so a lead whose context is full is actually freed, not re-prompted onto it.
compact_and_continue(){
  tm send-keys -t "$SESS" Escape; sleep 1; tm send-keys -t "$SESS" Escape; sleep 2
  tm send-keys -t "$SESS" -l "/compact"; sleep 1; tm send-keys -t "$SESS" Enter
  for _i in $(seq 1 30); do sleep 10; tm capture-pane -p -t "$SESS" 2>/dev/null | grep -q "Compacting" || break; done
  send_prompt "$CONT"
}

# Start a fresh lead: kill every claude in this run's tree so the death-recovery path below relaunches a
# clean agent that continues from disk. This is the escalation for a lead that /compact cannot free,
# because a compacted context refills at once and a clean one does not. Never a restart from scratch.
fresh_lead_restart(){
  local pid
  for pid in $tree; do case "$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)" in *claude*) kill    "$pid" 2>/dev/null;; esac; done
  sleep 3
  for pid in $tree; do case "$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)" in *claude*) kill -9 "$pid" 2>/dev/null;; esac; done
}

# Free a stalled or saturated lead: compact in place, and escalate to a fresh lead once repeated
# compacts stop producing on-disk progress. The compact count resets whenever a real artifact appears.
unchoke(){
  compacts=$((compacts+1))
  if [ "$compacts" -ge "$FRESH_LEAD_AFTER" ]; then
    restarts=$((restarts+1))
    if [ "$restarts" -gt "$MAX_RESTARTS" ]; then
      finish "GIVING UP: $MAX_RESTARTS consecutive restarts produced no new on-disk artifact; the run is \
finished or stuck, and re-driving it would only churn. Inspect $RUNS and this log, then delete \
$STOP and relaunch the supervisor to resume."
    fi
    say "CHURN: ${compacts} compacts, still no new artifact -> FRESH-LEAD restart (clean context, continue-from-disk; restart $restarts/$MAX_RESTARTS)"
    fresh_lead_restart; compacts=0
  else
    compact_and_continue
  fi
}

say "supervisor start: session=$SESS rundir=$RUNDIR socket=$SOCK"
dead=0; stall=0; genstall=0; compacts=0; restarts=0
while true; do
  # Terminal: the run said it is done. Stop cleanly instead of re-driving a finished run forever.
  run_done && finish "run finished ($RUNS_REL/<slug>.done present); supervisor exiting"

  # A human interacting (copy-mode / scrollback / selection) -> never touch the session.
  if [ "$(tm display-message -p -t "$SESS" '#{pane_in_mode}' 2>/dev/null || echo 0)" = "1" ]; then
    dead=0; stall=0; genstall=0; sleep "$POLL"; continue
  fi

  pane="$(tm capture-pane -p -t "$SESS" 2>/dev/null || true)"
  pane_pid="$(tm display-message -p -t "$SESS" '#{pane_pid}' 2>/dev/null || true)"
  tree="$(proc_tree "$pane_pid" | sort -u | tr '\n' ' ')"

  # 1) auto-approve a permission prompt; ONLY when the prompt box is actually at the BOTTOM of the pane
  #    (last few non-empty lines) AND shows both the question line and the "1. Yes" option. Matching the
  #    full signature at the tail (not anywhere on screen) stops a rendered diff or quoted text from
  #    triggering a stray "1<Enter>" into the running agent.
  tail_pane="$(printf '%s\n' "$pane" | grep -v '^[[:space:]]*$' | tail -8)"
  # (a) a normal permission prompt, or (b) the first-launch "trust this folder" dialog that a fresh
  #     resume (after a death) hits; both are answered with option 1 ("Yes"). Handling the trust
  #     dialog is REQUIRED so death-recovery on a fresh checkout isn't blocked forever at it.
  if { printf '%s' "$tail_pane" | grep -q 'Do you want to proceed' \
       || printf '%s' "$tail_pane" | grep -qiE 'trust (this|the) folder|created or one you trust'; } \
     && printf '%s' "$tail_pane" | grep -qE '(❯[[:space:]]*)?1\.[[:space:]]*Yes'; then
    tm send-keys -t "$SESS" 1; sleep 1; tm send-keys -t "$SESS" Enter; sleep 1
    say "approved a prompt (permission or trust-folder)"; continue
  fi

  # 2) agent gone from THIS session's tree -> resume from disk (after DEAD_STRIKES).
  #    Screen-independent: match the live process by its cmdline in the pane's own tree, not the
  #    footer text and not a global pgrep. Auto-compaction keeps the process alive, so it won't trip.
  agent_live=""
  for pid in $tree; do
    if tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q 'claude'; then agent_live=1; break; fi
  done
  if [ -z "$agent_live" ]; then
    dead=$((dead+1)); say "agent absent from session tree (strike $dead/$DEAD_STRIKES)"
    if [ "$dead" -ge "$DEAD_STRIKES" ]; then
      restarts=$((restarts+1))
      if [ "$restarts" -gt "$MAX_RESTARTS" ]; then
        finish "GIVING UP: $MAX_RESTARTS consecutive restarts produced no new on-disk artifact; the run is \
finished or stuck, and re-driving it would only churn. Inspect $RUNS and this log, then delete \
$STOP and relaunch the supervisor to resume."
      fi
      say "RESUME (death): fresh claude + continue-from-disk (restart $restarts/$MAX_RESTARTS)"
      start_agent; send_prompt "$CONT"; dead=0; stall=0
    fi
    sleep "$POLL"; continue
  fi
  dead=0

  # 3) stall: no new artifact under the run dir and no heavy compute in this run's process tree.
  #    Scoped to the run, so a neighbor's job on a shared machine cannot mask a stall and the run's
  #    own long build or benchmark (CPU-bound, logging only at phase boundaries) does not look like
  #    one. "Heavy compute" is any non-claude process in the tree above the CPU threshold.
  #    Only real progress counts: the supervisor's own log, heartbeat files, and benchmark logs are
  #    ignored, since a lead that re-runs the same benchmark or only emits heartbeats is not progressing.
  fresh_artifact="$(find "$RUNS" -type f -newermt "-2 min" \
      ! -name 'supervisor.*' ! -iname '*heartbeat*' ! -iname '*liveness*' ! -iname '*keepalive*' \
      ! -path '*/logs/*' ! -name '*.log' 2>/dev/null | head -1)"
  # Heavy compute: any process in the tree other than the agent itself above the CPU threshold. The
  # agent is excluded by its command line, not its comm (which is "node"), so a spinning agent is not
  # mistaken for compute.
  heavy_compute=""
  for pid in $tree; do
    case "$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)" in *claude*) continue;; esac
    pc="$(LC_ALL=C ps -o pcpu= -p "$pid" 2>/dev/null | tr -d ' ')"   # LC_ALL=C -> dot decimal (not "23,4")
    [ -n "$pc" ] || continue
    if [ "${pc%[.,]*}" -gt "$CPU_BUSY" ] 2>/dev/null; then heavy_compute="$pid:$pc"; break; fi
  done
  # Is the agent generating? Claude Code shows "esc to interrupt" in the footer only while a turn is
  # generating. A coding-agent subagent can generate for many minutes with no new artifact and almost no
  # local CPU, so generating counts as progress; only a very long generation with no artifact is cut.
  generating=""; printf '%s' "$pane" | grep -q 'esc to interrupt' && generating=1
  # Is a SUBAGENT actively running? The agent list shows a running teammate as "◯ <name> … <Xm Ys>" with
  # an elapsed timer. A delegated coding agent can run a long bench that is disk-I/O-bound at ~0% CPU (so
  # heavy_compute misses it) and leaves the footer NOT "generating"; without this it is false-nudged on
  # the idle path. An active subagent is progress; treat it like generating (bounded by the same ceiling).
  subagent_active=""; printf '%s' "$pane" | grep -qE '◯[[:space:]].+[0-9]+m[[:space:]]+[0-9]+s' && subagent_active=1
  # SATURATED = the lead's context is ~full ("100% context used"): a /compact can't create lasting headroom,
  # so for it, burning CPU on the same binary is NOT progress; only a real new code/decision-log artifact is.
  saturated=""; printf '%s' "$pane" | grep -qE '(100|9[0-9])% context used' && saturated=1
  # PER-CODING-AGENT TIME-BOX: the longest-running delegated sub-agent's elapsed minutes (from "◯ <name> … <Xm Ys>").
  # A single coding agent past CODING_AGENT_MAX_MIN with NO new code artifact is monopolizing the cycle -> treat like saturation.
  # Match the elapsed timer "<Xh >Ym Zs" on subagent lines (handles hour-plus + names containing digits), convert
  # to total minutes, take the max across subagents.
  sub_min="$(printf '%s' "$pane" | grep '◯' | grep -oE '([0-9]+h[[:space:]]+)?[0-9]+m[[:space:]]+[0-9]+s' \
      | awk '{h=0;m=0; for(i=1;i<=NF;i++){ if($i ~ /h$/) h=$i+0; else if($i ~ /m$/) m=$i+0 } print h*60+m }' \
      | sort -n | tail -1)"
  long_coding_agent=""; [ -n "$sub_min" ] && [ "$sub_min" -ge "$CODING_AGENT_MAX_MIN" ] 2>/dev/null && [ -z "$fresh_artifact" ] && long_coding_agent=1

  # RESET the churn only on REAL progress: a new source/decision-log/test artifact, OR local heavy compute when the lead
  # is NOT saturated AND no coding agent is over its time-box (a fresh coding agent legitimately builds/benches). A SATURATED
  # lead; or an over-time-box coding agent; burning CPU with no new artifact must NOT reset, else escalation never fires.
  if [ -n "$fresh_artifact" ] || { [ -n "$heavy_compute" ] && [ -z "$saturated" ] && [ -z "$long_coding_agent" ]; }; then
    stall=0; genstall=0; compacts=0
    [ -n "$fresh_artifact" ] && restarts=0   # a real artifact = the last restart worked; re-arm the budget
  elif [ -n "$generating" ] || [ -n "$subagent_active" ] || [ -n "$heavy_compute" ]; then
    genstall=$((genstall+1)); stall=0
    # A saturated lead, or a coding agent past its time-box, generating with no new code artifact gets the
    # shorter ceiling; repeated compacts without progress escalate to a fresh lead (see unchoke()).
    # A coding agent within its time-box legitimately generates for many minutes and gets the full ceiling.
    ceil="$GEN_STALL_STRIKES"; { [ -n "$saturated" ] || [ -n "$long_coding_agent" ]; } && ceil="$SAT_STALL_STRIKES"
    say "generating/benching, no new code artifact (gen $genstall/$ceil${saturated:+ · saturated}${long_coding_agent:+ · coding agent ${sub_min}m})"
    if [ "$genstall" -ge "$ceil" ]; then
      say "GEN-STALL (ceil=$ceil): compact; fresh lead if it keeps churning"
      unchoke; genstall=0; stall=0
    fi
  else
    # idle at the prompt, no artifact, no compute -> a genuine stall.
    stall=$((stall+1)); say "idle, no new artifact (strike $stall/$STALL_STRIKES)"
    if [ "$stall" -ge "$STALL_STRIKES" ]; then
      say "STALL: compact; fresh lead if it keeps churning"
      unchoke; stall=0; genstall=0
    fi
  fi
  sleep "$POLL"
done
