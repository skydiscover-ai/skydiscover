# Agent Roles

One instruction file per agent in the SkySynth figure, grouped by phase. The lead (`../SKILL.md`)
runs a role by giving it its file as the prompt.

```
agents/
├── 1-specification/
│   ├── spec-builder.md     Spec Builder: studies real systems, writes the specification
│   └── kb-builder.md       writes the knowledge base wiki for a new domain (optional, background)
└── 2-synthesis-loop/
    ├── planner.md          Planner: the design for the next iteration
    ├── coding-agent.md     Coding Agent: one well-scoped, tested change
    ├── dsa.md              Coding Agent, proof-driven: one step of code and proof
    ├── isa.md              Coding Agent, proof-driven: a new design when the DSA stalls
    ├── evaluator.md        Evaluator: the tests and the benchmark
    ├── auditor.md          Auditor: turns every reward hack into a test; reviews the final candidate
    └── critic.md           Critic: guidance for the next iteration
```

A role with more than one job takes a **mode** from the lead's prompt; modes are sections of its
file, not separate roles.

A brief is written for the role, which sees its brief, the lead's prompt (mode, run path,
constraints), and the run directory, and nothing of the lead's own procedure. So a brief says what
the role reads and writes and when the lead runs it, in the role's terms; the sequence of steps,
and their numbering, belong to `../SKILL.md`. Each brief ends by pointing at the one section of
`SKILL.md` every role must follow, "Rules That Hold Everywhere".
