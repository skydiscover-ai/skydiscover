---
name: skysynth
description: >
  Build a system specialized for your workload from a prompt or a formal spec: study reference
  systems, settle the requirements with the user, then build it behind tests (or a machine-checked
  proof) while an auditor turns every reward hack it finds into a new test. Trigger on
  requests to build, synthesize, or specialize a system (e.g. "skysynth build me a key-value
  store", or a path to a task.md spec).
---

# SkySynth: Codex Entry Point

This file is a thin entry point. The workflow lives at the plugin root, four directories up
(`../../../../SKILL.md`, beside `agents/`, `references/`, and `scripts/`).

1. Read the plugin root's `SKILL.md`: the lead workflow, its rules, and its references.
2. Follow it as the lead, with the user's request as the system to synthesize.

Role agents: the briefs are in the plugin root's `agents/<phase>/`. If this project has
`.codex/agents/*.toml` installed (`skydiscover init --agent codex`), run roles as subagents by
name; otherwise adopt each role's brief from `agents/<phase>/<role>.md` when its phase begins.
