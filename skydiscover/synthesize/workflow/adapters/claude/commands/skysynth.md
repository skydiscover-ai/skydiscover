---
description: >
  Build a system specialized for your workload from a prompt or a formal spec: study reference
  systems, settle the requirements with the user, then build it behind tests (or a machine-checked
  proof) while an auditor turns every reward hack it finds into a new test.
argument-hint: <what to build, or a path to a task.md spec>
---

This file is a thin entry point. The workflow lives at the plugin root, three directories above
this file's `commands/` directory, beside `agents/`, `references/`, and `scripts/`.

1. Read `${CLAUDE_PLUGIN_ROOT}/SKILL.md`: the lead workflow, its rules, and its references.
2. Follow it as the lead, with the user's request as the system to synthesize:

$ARGUMENTS
