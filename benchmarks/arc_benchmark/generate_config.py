import json
import os

import yaml


def load_task_as_prompt(task_json, task_num):
    with open(task_json, "r") as f:
        tasks = json.load(f)

    task_id = list(tasks.keys())[int(task_num)]
    task = tasks[task_id]
    train_inputs = [inp["input"] for inp in task["train"]]
    train_outputs = [gt["output"] for gt in task["train"]]

    train_pairs = ""
    for i, (inp, out) in enumerate(zip(train_inputs, train_outputs)):
        train_pairs += f"In {i} - {inp}\nOut {i} - {out}\n"

    prompt = f"""You are participating in a puzzle solving competition. You are an expert at solving puzzles.
Find the common pattern that transforms each input grid into its corresponding output grid.

Your task is to write python functions that implement the MOST GENERAL transformation rule. The rule must:
- Apply consistently to ALL training examples
- Generalize to unseen inputs (critical for success)
- Be based on structural patterns, not memorized examples
- Use relative/spatial rules rather than absolute coordinates

Generalization rules (THIS IS CRITICAL):
- Infer the transformation ONLY from the training input-output pairs
- If multiple rules fit the training data, choose the SIMPLEST and MOST GENERAL one
- Prefer structural/relational rules (shapes, adjacency, symmetry, patterns) over coordinate-based rules
- Do NOT hardcode any values, coordinates, or specific grid sizes that appear in training examples
- Think: "What is the underlying principle?" not "What fits these specific examples?"
- Use numpy only (no external libraries)

Common failure modes to avoid:
- Overfitting to specific grid sizes or positions in training examples
- Hardcoding colors, coordinates, or counts from training data
- Assuming global properties (like separator colors) without verifying across ALL examples
- Using absolute positions when relative/structural rules would generalize better

Solution approach:
- Analyze the training examples to identify the CORE transformation principle
- Prefer block-wise, object-wise, or pattern-based rules that work locally
- If the grid has distinct regions, solve each region independently
- Build flexible rules that adapt to different input sizes and structures

Training examples:
{train_pairs}

Your task: Write 2 different Python functions that implement the general transformation rule.
- Each function takes a 2D numpy array as input and returns the transformed 2D numpy array
- The two attempts should use genuinely different strategies (e.g., different algorithmic approaches)
- Focus on generalization - your solution will be evaluated on BOTH training examples AND unseen test cases

CRITICAL: Write general transformations that discover the underlying rule, not memorize the training examples.

Remember to only output the modified python functions as your solution."""

    return prompt


def generate_config(task_num, task_file, dataset_root=None, base_config=None):
    if dataset_root is None:
        dataset_root = os.getenv("DATA_ROOT")
        if not dataset_root:
            dataset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    task_json = os.path.join(dataset_root, f"arc-agi_{task_file}_challenges.json")
    prompt = load_task_as_prompt(task_json, task_num)

    if base_config is None:
        default_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        base_config = os.getenv("BASE_CONFIG", default_base)
    with open(base_config, "r") as file:
        config = yaml.safe_load(file)

    config["prompt"]["system_message"] = prompt
    # Use OPENAI_API_KEY at runtime if set (keeps real key out of committed config)
    api_key_env = os.getenv("OPENAI_API_KEY")
    if api_key_env and api_key_env.strip() and api_key_env != "your-gemini-api-key":
        config["llm"]["api_key"] = api_key_env.strip()
    # Override max_iterations from env if set (e.g. by run_discovery.sh)
    max_iter_env = os.getenv("MAX_ITERATIONS")
    if max_iter_env is not None and str(max_iter_env).strip() != "":
        try:
            config["max_iterations"] = int(max_iter_env)
        except ValueError:
            pass

    # Write to a per-task config file so parallel runs don't conflict
    out_path = os.getenv("CONFIG_OUT", f"./config_task_{task_num}.yaml")
    with open(out_path, "w") as file:
        yaml.dump(config, file)
    return out_path


if __name__ == "__main__":
    TASK_FILE = os.getenv("ARC_TASK_FILE", "training")
    TASK_NUM = os.getenv("TASK_NUM", 0)

    path = generate_config(TASK_NUM, TASK_FILE)
    print(path)
