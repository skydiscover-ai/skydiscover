# EvoxContextBuilder

Extends `DefaultContextBuilder` with LLM-generated summaries and statistical insights injected into every prompt. Used by the EvoX co-evolutionary search algorithm.

## What it adds

Before building the prompt, `EvoxContextBuilder` makes parallel LLM calls to a cheaper guide model to generate:

- Population statistics insights (score trends, diversity, improvement rates)
- Problem context summary (description of the task derived from the evaluator)
- Batch summaries of context programs

These are assembled into the prompt alongside the standard sections (metrics, previous attempts, current program).

## Files

| File | What it does |
|------|-------------|
| `builder.py` | `EvoxContextBuilder` class, parallel LLM calls, template assembly |
| `formatters.py` | Pure formatting functions (population state, execution trace, DB stats diff) |
| `templates/` | Prompt templates (override default ones with the same filename) |

## Templates

`EvoxContextBuilder` loads both `default/templates/` and `evox/templates/`. Files with the same name in `evox/templates/` override the defaults.

| File | Role | Purpose |
|------|------|---------|
| `system_message.txt` | system | Overrides the default system prompt for search evolution |
| `search_evolution_user_message.txt` | user | Main user prompt for search algorithm discovery |
| `problem_template.txt` | fragment | Problem description fed to the guide LLM |
| `problem_context_summary_system_message.txt` | system (guide LLM) | Instructs guide LLM to summarize the problem context |
| `stats_insight_system_message.txt` | system (guide LLM) | Instructs guide LLM to generate population statistics insights |
| `batch_summary_prompt.txt` | system + user (guide LLM) | Instructs guide LLM to summarize context programs in batch |
