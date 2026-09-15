#!/usr/bin/env bash
# Wire the /skysynth skill, its agent roles, and the hook that runs the tests before anything is
# delivered into a coding agent's project config.
#
# Usage: install.sh [--agent claude|cursor|codex|pi|all] [--no-hook] [target_project_dir]
#   --agent      which coding agent to wire (default: claude)
#   --no-hook    skip the hook configuration
#   target dir   where to write the agent config (default: this repo's root)
#
# skydiscover init runs this. Everything is a symlink into skydiscover/synthesize/workflow/, so
# editing the workflow updates every installed agent, and re-running is safe. Per agent:
#
#   claude   .claude/skills/skysynth  .claude/agents/*.md    .claude/settings.local.json (TaskCompleted + PreToolUse)
#   cursor   .cursor/skills/skysynth  .cursor/agents/*.md    .cursor/hooks.json (beforeShellExecution)
#   codex    .agents/skills/skysynth  .codex/agents/*.toml   .codex/hooks.json (SubagentStop + PreToolUse)
#   pi       .agents/skills/skysynth  .pi/agents/*.md        .pi/extensions/skydiscover-hooks.ts
#
# Cursor has no task-completion event, so there the delivery check runs through run_tests.py,
# which the lead invokes each cycle; only the clone-reuse guard is hook-wired.

set -euo pipefail
scripts_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
here="$(cd "$scripts_dir/.." && pwd)"          # .../skydiscover/synthesize
repo_root="$(cd "$here/../.." && pwd)"
workflow="$here/workflow"
hooks="$workflow/hooks"

agent="claude" nohook=0 target="$repo_root"
while [ $# -gt 0 ]; do
  case "$1" in
    --agent)
      shift
      [ $# -gt 0 ] || { echo "install: --agent requires a value" >&2; exit 2; }
      agent="$1" ;;
    --no-hook) nohook=1 ;;
    *)
      [ -d "$1" ] || { echo "install: target directory does not exist: $1" >&2; exit 1; }
      target="$(cd "$1" && pwd)" ;;
  esac
  shift
done
case "$agent" in
  claude|cursor|codex|pi|all) ;;
  *) echo "install: unknown agent '$agent' (expected claude|cursor|codex|pi|all)" >&2; exit 2 ;;
esac
cd "$target"

# ---------------------------------------------------------------------------------------- helpers

# keep_aside <path>: a file or directory the user owns (not our symlink) is moved to <path>.bak
# instead of deleted; the user decides what to do with it.
keep_aside() {
  [ -e "$1" ] && [ ! -L "$1" ] || return 0
  bak="$1.bak"
  [ -e "$bak" ] && bak="$1.bak.$(date +%Y%m%d%H%M%S)"
  mv "$1" "$bak"
  echo "install: $1 was not a skysynth symlink; kept as $bak" >&2
}

# link_dir <src> <dst>: replace a stale copy with a symlink.
link_dir() {
  keep_aside "$2"
  ln -sfn "$1" "$2"
}

# link_agents <dst dir>: one symlink per role brief. Briefs live in agents/<phase>/<role>.md; the
# coding agent's directory is flat, keyed by basename. agents/README.md documents them and is not one.
link_agents() {
  mkdir -p "$1"
  for f in "$workflow"/agents/*/*.md; do
    b="$(basename "$f")"
    keep_aside "$1/$b"
    ln -sfn "$f" "$1/$b"
  done
}

# prune_broken <dir>...: remove symlinks whose target no longer exists (renamed or deleted briefs).
prune_broken() {
  find -L "$@" -maxdepth 1 -type l -delete 2>/dev/null || true
}

# Claude Code: merge the hook and env config into .claude/settings.local.json without touching
# anything else in it. Backs the file up once and writes atomically.
write_claude_settings() {
python3 - "$1" "$2" "$3" "$4" <<'PY'
import json, os, shutil, sys, uuid
path, hook, guard, edit_guard = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
try:
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    assert isinstance(cfg, dict)
except Exception:
    cfg = {}
before = json.dumps(cfg, sort_keys=True)
# One entry per hook: update the command in place if the script moved, append if absent. The
# delivery check runs the whole test suite (sanitizer builds included), so it gets a long timeout.
HOOK_TIMEOUT, GUARD_TIMEOUT = 3600, 60
tc = cfg.setdefault("hooks", {}).setdefault("TaskCompleted", [])
found = False
for e in tc:
    for h in (e.get("hooks", []) if isinstance(e, dict) else []):
        if isinstance(h, dict) and "delivery_check.sh" in str(h.get("command", "")):
            found = True
            if h.get("command") != hook:
                h["command"] = hook
            # A shorter timeout than the check's own wall cap would let a slow check pass unchecked.
            if not isinstance(h.get("timeout"), int) or h["timeout"] < HOOK_TIMEOUT:
                h["timeout"] = HOOK_TIMEOUT
if not found:
    tc.append({"hooks": [{"type": "command", "command": hook, "timeout": HOOK_TIMEOUT}]})
pt = cfg.setdefault("hooks", {}).setdefault("PreToolUse", [])
gfound = False
for e in pt:
    for h in (e.get("hooks", []) if isinstance(e, dict) else []):
        if isinstance(h, dict) and "clone_reuse_guard.py" in str(h.get("command", "")):
            gfound = True
            if h.get("command") != guard:
                h["command"] = guard
            h.setdefault("timeout", GUARD_TIMEOUT)
if not gfound:
    pt.append(
        {"matcher": "Bash", "hooks": [{"type": "command", "command": guard, "timeout": GUARD_TIMEOUT}]}
    )
# The framework-edit guard: the agents may not patch the installed skill.
EDIT_MATCHER = "Edit|Write|MultiEdit|NotebookEdit"
efound = False
for e in pt:
    for h in (e.get("hooks", []) if isinstance(e, dict) else []):
        if isinstance(h, dict) and "framework_edit_guard.py" in str(h.get("command", "")):
            efound = True
            if h.get("command") != edit_guard:
                h["command"] = edit_guard
            h.setdefault("timeout", GUARD_TIMEOUT)
if not efound:
    pt.append(
        {
            "matcher": EDIT_MATCHER,
            "hooks": [{"type": "command", "command": edit_guard, "timeout": GUARD_TIMEOUT}],
        }
    )
if json.dumps(cfg, sort_keys=True) != before:
    bak = f"{path}.bak.{os.getpid()}"
    if os.path.exists(path) and not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"backed up existing settings to {bak}")
    tmp = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
PY
}

# Cursor: add the clone-reuse guard to .cursor/hooks.json, same merge discipline as above.
write_cursor_hooks() {
python3 - "$1" "$2" <<'PYCURSOR'
import json, os, shutil, sys, uuid
path, guard = sys.argv[1], sys.argv[2]
try:
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    assert isinstance(cfg, dict)
except Exception:
    cfg = {}
before = json.dumps(cfg, sort_keys=True)
cfg.setdefault("version", 1)
entries = cfg.setdefault("hooks", {}).setdefault("beforeShellExecution", [])
found = False
for e in entries:
    if isinstance(e, dict) and "clone_reuse_guard.py" in str(e.get("command", "")):
        found = True
        if e.get("command") != guard:
            e["command"] = guard
if not found:
    entries.append({"command": guard})
if json.dumps(cfg, sort_keys=True) != before:
    bak = f"{path}.bak.{os.getpid()}"
    if os.path.exists(path) and not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"backed up existing hooks to {bak}")
    tmp = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
PYCURSOR
}

# ---------------------------------------------------------------------------------------- agents

install_claude() {
  mkdir -p .claude/skills
  link_dir "$workflow" .claude/skills/skysynth
  link_agents .claude/agents
  prune_broken .claude/agents .claude/skills
  echo "Linked the /skysynth skill and its agent roles into .claude/."

  for t in python3 "${CXX:-g++}" jq; do
    command -v "$t" >/dev/null 2>&1 || echo "WARNING: '$t' not found on PATH (python3 runs the scripts, a C++ compiler is needed only for C++ tests, jq parses hook payloads)."
  done

  [ "$nohook" -eq 1 ] && { echo "(--no-hook: skipped .claude/settings.local.json)"; return; }
  # Inside this repo, route through $CLAUDE_PROJECT_DIR so the written path survives a clone or
  # move; fall back to the absolute path when the variable is unset. Elsewhere, absolute paths.
  if [ "$target" = "$repo_root" ]; then
    hook_cmd='p="$CLAUDE_PROJECT_DIR/skydiscover/synthesize/workflow/hooks/delivery_check.sh"; [ -f "$p" ] || p="'"$hooks"'/delivery_check.sh"; exec bash "$p"'
    guard_cmd='p="$CLAUDE_PROJECT_DIR/skydiscover/synthesize/workflow/hooks/clone_reuse_guard.py"; [ -f "$p" ] || p="'"$hooks"'/clone_reuse_guard.py"; [ -f "$p" ] && exec python3 "$p" || exit 0'
    edit_guard_cmd='p="$CLAUDE_PROJECT_DIR/skydiscover/synthesize/workflow/hooks/framework_edit_guard.py"; [ -f "$p" ] || p="'"$hooks"'/framework_edit_guard.py"; [ -f "$p" ] && exec python3 "$p" || exit 0'
  else
    # Absolute paths, guarded the same way: a checkout that moved must not turn every Bash call
    # into a blocked PreToolUse (exit 2) or every task completion into a failed hook.
    hook_cmd='p="'"$hooks"'/delivery_check.sh"; [ -f "$p" ] && exec bash "$p"; echo "delivery check: $p is missing (was the skydiscover checkout moved?); re-run skydiscover init" >&2; exit 0'
    guard_cmd='p="'"$hooks"'/clone_reuse_guard.py"; [ -f "$p" ] && exec python3 "$p" || exit 0'
    edit_guard_cmd='p="'"$hooks"'/framework_edit_guard.py"; [ -f "$p" ] && exec python3 "$p" || exit 0'
  fi
  settings="$target/.claude/settings.local.json"
  if write_claude_settings "$settings" "$hook_cmd" "$guard_cmd" "$edit_guard_cmd"; then
    echo "Configured $settings (the hook that runs the tests before anything is delivered, plus the PreToolUse guards); restart Claude Code to load it."
    echo "Optional: set CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1 in that file's env to run the roles as an agent team."
  else
    echo "Could not write $settings. Add manually a TaskCompleted hook running:  $hook_cmd"
  fi
}

install_cursor() {
  mkdir -p .cursor/skills
  link_dir "$workflow" .cursor/skills/skysynth
  link_agents .cursor/agents
  prune_broken .cursor/skills .cursor/agents
  echo "Linked the /skysynth skill and its agent roles into .cursor/."
  [ "$nohook" -eq 1 ] && { echo "(--no-hook: skipped .cursor/hooks.json)"; return; }
  write_cursor_hooks "$target/.cursor/hooks.json" "python3 $hooks/clone_reuse_guard.py"
  echo "Configured .cursor/hooks.json (clone-reuse guard on beforeShellExecution)."
}

install_codex() {
  mkdir -p .agents/skills .codex/agents
  link_dir "$workflow" .agents/skills/skysynth
  # Codex agents are TOML, generated from the briefs of this checkout (the adapter is stdlib-only;
  # importing it as a module could pick up another installed copy of skydiscover).
  python3 "$workflow/adapters/codex/agents.py" .codex --quiet
  prune_broken .agents/skills
  echo "Linked the /skysynth skill into .agents/ and generated its agent roles into .codex/."
  [ "$nohook" -eq 1 ] && { echo "(--no-hook: skipped .codex/hooks.json)"; return; }
  write_codex_hooks "$target/.codex/hooks.json" "$workflow"
  echo "Configured .codex/hooks.json (the hook that runs the tests before anything is delivered, plus the PreToolUse guard)."
}

# Codex: merge the plugin's hook manifest into .codex/hooks.json, with the plugin root resolved to
# this checkout, without touching anything else in it. Backs the file up once and writes atomically.
write_codex_hooks() {
python3 - "$1" "$2" <<'PYCODEX'
import json, os, shutil, sys, uuid
path, workflow = sys.argv[1], sys.argv[2]
with open(os.path.join(workflow, "adapters", "codex", "hooks.json"), encoding="utf-8") as fh:
    shipped = json.load(fh)["hooks"]
try:
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    assert isinstance(cfg, dict)
except Exception:
    cfg = {}
before = json.dumps(cfg, sort_keys=True)
for event, entries in shipped.items():
    mine = cfg.setdefault("hooks", {}).setdefault(event, [])
    for entry in entries:
        for h in entry["hooks"]:
            h["command"] = h["command"].replace("${PLUGIN_ROOT:-${CLAUDE_PLUGIN_ROOT:-.}}", workflow)
        script = os.path.basename(entry["hooks"][0]["command"].split('"')[1])
        found = False
        for e in mine:
            for h in (e.get("hooks", []) if isinstance(e, dict) else []):
                if isinstance(h, dict) and script in str(h.get("command", "")):
                    found = True
                    h["command"] = entry["hooks"][0]["command"]
                    h.setdefault("timeout", entry["hooks"][0]["timeout"])
        if not found:
            mine.append(entry)
if json.dumps(cfg, sort_keys=True) != before:
    bak = f"{path}.bak.{os.getpid()}"
    if os.path.exists(path) and not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f"backed up existing hooks to {bak}")
    tmp = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
PYCODEX
}

install_pi() {
  mkdir -p .agents/skills .pi/extensions
  link_dir "$workflow" .agents/skills/skysynth
  link_agents .pi/agents
  prune_broken .agents/skills .pi/agents
  echo "Linked the /skysynth skill and its agent roles into .agents/ and .pi/."
  [ "$nohook" -eq 1 ] && { echo "(--no-hook: skipped .pi/extensions/skydiscover-hooks.ts)"; return; }
  # pi has no hooks.json; the extension calls the same two hook scripts.
  sed "s#__SKYDISCOVER_HOOKS_DIR__#$hooks#g" "$workflow/adapters/pi/skydiscover-hooks.ts" \
    > .pi/extensions/skydiscover-hooks.ts
  echo "Configured .pi/extensions/skydiscover-hooks.ts."
}

# The workflow's helpers run as `python3 -m skydiscover.synthesize...` from the target directory, so
# the package has to be importable there. The symlinks alone do not make it so.
check_package() {
  (cd "$target" && python3 -c "import skydiscover" >/dev/null 2>&1) && return
  echo "WARNING: 'import skydiscover' fails for $(command -v python3) in $target."
  echo "         The agents' helper commands (python3 -m skydiscover.synthesize...) will not work until it does."
  echo "         Fix: (cd \"$repo_root\" && uv sync && source .venv/bin/activate), then re-run in the same shell."
}

case "$agent" in
  claude) install_claude ;;
  cursor) install_cursor ;;
  codex)  install_codex ;;
  pi)     install_pi ;;
  all)    install_claude; install_cursor; install_codex; install_pi ;;
esac
check_package
echo "Installed /skysynth for $agent in $target. Open your coding agent there and run: /skysynth <what to build>"
