# Coding Agent Adapters

The workflow itself is agent-neutral: `SKILL.md`, `agents/`, `hooks/`, `scripts/`, `references/`.
This directory holds the small pieces each coding agent needs to load it.

| Agent | File | Why it exists |
|---|---|---|
| Claude Code | `claude/commands/skysynth.md` | the `/skysynth` slash command; reads the root `SKILL.md` and follows it |
| Codex | `codex/skills/skysynth/SKILL.md` | same entry point in Codex's skill layout |
| Codex | `codex/hooks.json` | Codex's hook events, pointing at the shared scripts in `hooks/` |
| Codex | `codex/agents.py` | turns the briefs in `agents/<phase>/*.md` into the TOML files Codex needs; run by `skydiscover init --agent codex` |
| pi | `pi/skydiscover-hooks.ts` | pi has no hooks file, so this extension runs the shared hook scripts |

Cursor needs no adapter: `skydiscover init --agent cursor` symlinks the workflow into
`.cursor/skills/` and writes `.cursor/hooks.json` directly.

The plugin manifests, `../.claude-plugin/plugin.json` and `../.codex-plugin/plugin.json`, are not in
this directory because Claude Code and Codex require them at the plugin root — the directory that
holds the components the plugin ships, namely `workflow/`. Paths inside a manifest are relative to
that root and cannot leave it. The manifests point at the files in this directory.
