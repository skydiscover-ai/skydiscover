#!/usr/bin/env bash
# Keep the supervisor itself alive for fully unattended runs: a systemd --user service where
# available (cron cannot hold a long-lived daemon on systemd hosts), else a cron watchdog every 2 min
# and at reboot. Both stand down once the supervisor writes its stop marker.
#
# Usage: keepalive.sh <tmux-session> <run-dir> [tmux-socket]
set -uo pipefail

SESS="${1:?usage: keepalive.sh <session> <run-dir> [socket]}"
RUNDIR="${2:?usage: keepalive.sh <session> <run-dir> [socket]}"
SOCK="${3:-${TMPDIR:-/tmp}/skydiscover.$(id -u).sock}"
SUP="$(cd "$(dirname "$0")" && pwd)/supervisor.sh"

# Every one of these values is pasted into a systemd unit, a crontab line and a pgrep pattern; i.e.
# into text that a shell and systemd will parse. A space, newline, quote, ';' or backtick in any of
# them injects a command instead of naming a session. Reject anything outside a conservative charset
# rather than trying to escape it for three different parsers.
for v in SESS RUNDIR SOCK SUP; do
  case "${!v}" in
    *[!A-Za-z0-9._/-]*|"")
      echo "keepalive: refusing to install; $v must match ^[A-Za-z0-9._/-]+$ (no spaces or shell" >&2
      echo "  metacharacters); it is embedded in a systemd unit and a crontab line. Got: ${!v}" >&2
      exit 2 ;;
  esac
done
[ -f "$SUP" ] || { echo "keepalive: no supervisor.sh at $SUP" >&2; exit 2; }
# Resolve the run dir NOW: cron and systemd start from a different cwd, so a relative "." would point
# the supervisor at $HOME instead of the run.
RUNDIR="$(cd "$RUNDIR" 2>/dev/null && pwd)" || { echo "keepalive: no such run dir: ${2}" >&2; exit 2; }

LOG="${TMPDIR:-/tmp}/sup.$(id -u).${SESS}.log"
UNIT="skydiscover-sup-${SESS}.service"
RUNS_REL="${SKYDISCOVER_RUNS:-$(python3 -m skydiscover.synthesize.spec.paths runs 2>/dev/null || echo .skydiscover)}"
case "$RUNS_REL" in /*) RUNS="$RUNS_REL" ;; *) RUNS="$RUNDIR/$RUNS_REL" ;; esac
STOP="${RUNS}/supervisor.${SESS}.stop"   # written by supervisor.sh when it is done

# ---- preferred: a systemd --user service (auto-restart on death + reboot) --------------------------------
install_systemd_user(){
  command -v systemctl >/dev/null 2>&1 || return 1
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  systemctl --user show-environment >/dev/null 2>&1 || return 1     # is the user bus reachable?
  loginctl enable-linger "$(id -un)" >/dev/null 2>&1 || true         # survive logout + reboot (best-effort)
  mkdir -p "$HOME/.config/systemd/user"
  # on-failure (not always): the supervisor exits 0 when the run is complete or it has given up, and
  # that decision must stick.
  cat > "$HOME/.config/systemd/user/$UNIT" <<UNIT
[Unit]
Description=skydiscover supervisor (session ${SESS})
[Service]
ExecStart=/bin/bash "${SUP}" "${SESS}" "${RUNDIR}" "${SOCK}"
Restart=on-failure
RestartSec=10
[Install]
WantedBy=default.target
UNIT
  systemctl --user daemon-reload || return 1
  systemctl --user enable --now "$UNIT" >/dev/null 2>&1 || return 1
  sleep 2
  systemctl --user is-active "$UNIT" >/dev/null 2>&1
}

# ---- fallback: a cron watchdog (every 2 min + @reboot) --------------------------------------------------
install_cron(){
  command -v crontab >/dev/null 2>&1 || return 1
  local MARK="skydiscover-keepalive:${SESS}:${RUNDIR}"
  local LAUNCH="setsid bash \"${SUP}\" \"${SESS}\" \"${RUNDIR}\" \"${SOCK}\" </dev/null >>\"${LOG}\" 2>&1"
  local GUARD="[ -e \"${STOP}\" ] || pgrep -f 'supervisor.sh ${SESS} ${RUNDIR}' >/dev/null 2>&1 || ${LAUNCH}"
  local existing; existing="$(crontab -l 2>/dev/null | grep -vF "${MARK}" || true)"
  {
    [ -n "${existing}" ] && printf '%s\n' "${existing}"
    printf '*/2 * * * * %s # %s\n' "${GUARD}" "${MARK}"
    printf '@reboot sleep 30; %s # %s\n' "${GUARD}" "${MARK}"
  } | crontab -
  start_supervisor
}

# Start the supervisor now if it is not already running. Direct argv; no eval and no string that a
# shell has to re-parse, so nothing in SESS/RUNDIR/SOCK can be read as a command.
start_supervisor(){
  pgrep -f "supervisor.sh ${SESS} ${RUNDIR}" >/dev/null 2>&1 && return 0
  setsid bash "${SUP}" "${SESS}" "${RUNDIR}" "${SOCK}" </dev/null >>"${LOG}" 2>&1 || true
}

if [ -e "$STOP" ]; then
  echo "keepalive: this run is already marked finished ($STOP)."
  echo "  Delete that file first if you want to resume supervising it."
  exit 1
fi

if install_systemd_user; then
  echo "keepalive: systemd --user service ${UNIT} ACTIVE (Restart=on-failure, reboot-enabled); preferred path"
  pgrep -af "supervisor.sh ${SESS} ${RUNDIR}" | head -1
elif install_cron; then
  echo "keepalive: systemd --user unavailable; cron watchdog installed (every 2 min + @reboot)"
  echo "  NOTE: on a systemd host where cron reaps the launched daemon, prefer the systemd --user path."
  pgrep -af "supervisor.sh ${SESS} ${RUNDIR}" | head -1
else
  echo "keepalive: no systemd --user and no cron available; starting supervisor once via setsid (NOT self-healing)."
  start_supervisor
fi
echo "keepalive: supervisor log -> ${LOG} and ${RUNS}/supervisor.${SESS}.log"
