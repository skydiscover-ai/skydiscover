# Context Builder

The context builder turns program state into LLM prompts. It is called once per iteration and returns `{"system": ..., "user": ...}`.

For most tasks you do not need to touch this. Just set the system prompt in `config.yaml`:

```yaml
prompt:
  system_message: |-
    You are an expert at optimizing load balancing algorithms.
```

Only write a custom builder if your algorithm has search-state data (tree path, island ID, rejection history) that the LLM should see.

---

## Structure

```
context_builder/
  base.py              ContextBuilder ABC (one method: build_prompt)
  utils.py             TemplateManager (loads .txt templates from directories)
  human_feedback.py    File-based human feedback injection
  default/
    builder.py         DefaultContextBuilder (handles diff / rewrite / image / prompt modes)
    templates/         .txt prompt templates
  evox/
    builder.py         EvoxContextBuilder (extends Default, adds LLM-generated summaries)
    templates/         .txt templates (override default ones with the same filename)
```

Each builder owns its own `TemplateManager`. Later directories passed to `TemplateManager` override earlier ones on filename conflicts.

---

## Default templates

| File | Role | When used |
|------|------|-----------|
| `system_message.txt` | system | Default system prompt (overridden by config) |
| `diff_user_message.txt` | user | Diff-based generation (default mode) |
| `full_rewrite_user_message.txt` | user | Full rewrite mode |
| `full_rewrite_prompt_opt_user_message.txt` | user | Prompt optimization tasks |
| `image_user_message.txt` | user | Image generation mode |
| `evaluator_system_message.txt` | system | LLM judge (only with llm_as_judge) |
| `evaluator_user_message.txt` | user | LLM judge user message |

---

## Writing a custom builder

The most common pattern is extending `DefaultContextBuilder` and injecting extra guidance via the `{search_guidance}` placeholder. The default templates already include this slot; an empty string makes it disappear cleanly.

```python
from pathlib import Path
from skydiscover.optimize.context_builder.default import DefaultContextBuilder
from skydiscover.optimize.context_builder.utils import TemplateManager

class MyContextBuilder(DefaultContextBuilder):

    def __init__(self, config):
        super().__init__(config)
        # load your templates on top of the defaults
        default_templates = str(Path(__file__).parent.parent / "default" / "templates")
        my_templates = str(Path(__file__).parent / "templates")
        self.template_manager = TemplateManager(default_templates, my_templates)

    def build_prompt(self, current_program, context=None, **kwargs):
        context = context or {}
        # format whatever the manager put into the context dict
        guidance = self._format_guidance(context.get("my_key"))
        return super().build_prompt(current_program, context, search_guidance=guidance, **kwargs)

    def _format_guidance(self, data):
        if not data:
            return ""
        return f"## CONTEXT\n{data}"
```

The manager populates `context["my_key"]` before calling `build_prompt()`, and sets the builder in its `__init__`:

```python
self.context_builder = MyContextBuilder(self.config)
```

Example to copy: `adaevolve/builder.py` (adds evaluator feedback, paradigm guidance, and sibling context).

---

## Registration

To make a builder available via config instead of hardcoding it in a manager, add it to `_init_context_builder()` in `search/default_discovery_controller.py`:

```python
elif template == "my_builder":
    self.context_builder = MyContextBuilder(self.config)
```

Then activate with:

```yaml
prompt:
  template: my_builder
```
