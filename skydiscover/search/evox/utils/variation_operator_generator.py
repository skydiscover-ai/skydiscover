#!/usr/bin/env python3
"""
Script to generate problem-specific variation operators (e.g. structural variation or local refinement).
Reads config.yaml and evaluator.py, then uses an LLM to generate both variation operators.

- Structural variation operator: Different algorithmic approaches, structural changes
- Local refinement operator: Intensify search within current approach, parameter tuning
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
from typing import Optional, Tuple

import yaml

from skydiscover.llm.llm_pool import LLMPool

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# EXPLORATION PROMPT - for diverge_operator (structural/algorithmic diversity)
# ------------------------------------------------------------------
EXPLORE_SYSTEM_PROMPT = """\
You are an expert at analyzing problems and suggesting diverse algorithmic approaches.

Given a problem description and the evaluator setup and code, generate a concise "different approaches" guidance block that will help an LLM generate diverse solutions.

## STEP 1: ANALYZE THE PROBLEM TYPE (do this internally, don't output)
Before suggesting anything, you MUST determine: 
- What kinds of problems are you dealing with? 
    - **Optimization problem**: continuous vs discrete, constrained vs unconstrained, what variables and constraints, and what optimization family, ...
    - **Algorithm/heuristic design**: what decisions, what tradeoffs
    - **System design**: what resources, what objectives
    - And anything else that is related to the problem type and context.
- What is the objective? 
    - Read the evaluator's scoring function (e.g., `combined_score`) to understand ALL components of the objective.
    - If the objective combines multiple sub-scores, what are the weights and tradeoffs between them?
- What are the constraints or requirements? 
    - E.g., bounds, accuracy, time limits, correctness criteria, etc.
- Does this problem have multiple decision stages? 
- What tradeoffs exist? 
    - E.g. size vs weight, cost vs quality, speed vs accuracy, ...
- For optimization problems specifically:
    - Are the decisions discrete or continuous (real-valued parameters)?
    - What variables are being optimized and what are the constraint types?

## STEP 2: PROVIDE SUGGESTIONS FOR DIFFERENT APPROACHES 

### STEP 2a: IDENTIFY STANDARD TOOLKIT
Before suggesting anything, think like a DOMAIN EXPERT. Ask yourself:
- "What would a textbook or survey paper list as the main approaches?"
- "What is the standard toolkit, or major libraries and functions for this problem domain?" (not just what context programs use)
- For optimization problems specifically
    - What type? (constrained/unconstrained, local/global, gradient/derivative-free)
    - What methods handle the constraint types? (bounds, equality, inequality)
    - How are constraints ENFORCED? Penalty methods, projection/repair, barrier methods, and exact constraint solvers are fundamentally different paradigms. Treat constraint handling as a distinct design axis, separate from the choice of optimizer.
    - What are ALL the decision variables? Not just the final output — also intermediate/structural variables (positions, configurations, orderings) that the output depends on. The FULL joint optimization over all these variables is the real problem.
    - Does the problem DECOMPOSE? Often the full problem has a HARD core (nonlinear, jointly over many variables with constraints) + EASIER sub-problems (one set of variables given another fixed). Identify the hardest sub-problem — the solver for that is the MAIN solver. Simpler methods (LP, closed-form) can serve as sub-routines for the easier parts.
    - What metaheuristic/evolutionary algorithms are available?

### STEP 2b: GENERATE SUGGESTIONS (based on problem & evaluator)
Your suggestions should be driven primarily by analyzing the PROBLEM and EVALUATOR from Step 1. Enumerate the landscape of known, proven techniques for this problem type. 
- In the Libraries bullet: include the canonical/standard library for this problem domain, with its key functions. Do not skip it.
- For multi-objective problems, include at least ONE category named after a KEY QUALITY DIMENSION from the evaluator's `combined_score`. Name the category after the objective, then list diverse techniques that target it.
- Consider MULTIPLE PARADIGMS: Your suggestions should span at least two distinct approach families. Don't anchor on one paradigm.
- Consider MULTI-PHASE / PIPELINE approaches: Chaining methods sequentially (e.g., one for an initial solution, another to refine it) can help.
- Consider MULTI-START / ENSEMBLE strategies: Running the same method from many different starting points and keeping the best result can be a simple but powerful technique.
- Do NOT dismiss simple approaches just because advanced ones exist. Simple methods with good tuning often beat complex methods.

## STEP 3: OUTPUT FORMAT (follow exactly)
```

EXAMPLES OF DIFFERENT approaches (NOT LIMITED TO, PROPOSE YOUR OWN):
- **[Category 1]**: [describe 3-5 alternative specific techniques], e.g. A ↔ B ↔ C, D ↔ E, etc.
- **[Category 2]**: [describe 3-5 alternative specific techniques], e.g. A ↔ B ↔ C, D ↔ E, etc.
- ... (3-6 bullet points total)

```

Your goal is to come up with the "DIFFERENT APPROACHES" sections, with the problem specific guidance, e.g. 
different structural components (e.g. initialization, objective design, search strategy to balance multiple factors, anything else that 
is related to the problem type and context).

EXAMPLE CATEGORIES (do not blindly copy, tailor to problem):
```
EXAMPLES OF DIFFERENT approaches (NOT LIMITED TO, PROPOSE YOUR OWN):
- **Libraries / solvers**: 
- **Multi-phase pipeline**: 
- **Algorithm family / strategy**: 
- **Runtime optimization (if applicable to the problem objective)** 
- **Construction / search method**: 
- **Constraint handling**: 
... Or anything else you can think of. 
```
Do not just blindly copy from the example; tailor the categories and ideas specifically to the problem type and context.

## RULES
1. LIBRARIES / TOOLS - Include for problems where libraries provide algorithmic solutions. 
    - **CRITICAL**: ONLY suggest libraries that appear in the "Available Packages in Environment" section. Do NOT suggest libraries that are not installed.
    - Always list it in the first category if applicable.
    - **USE THIS EXACT FORMAT**: `library.submodule: func1, func2 ↔ another_lib: funcA, funcB`. Use → to wire dependent functions: `setup_func → apply_func`.
        - **CRITICAL** For wrapper functions that accept a `method` parameter: you MUST specify the exact method names. 
            - The LLM needs to know WHICH specific solver to use. Format: `wrapper(method='X'), wrapper(method='Y')`.
    - **BE FOCUSED, NOT EXHAUSTIVE**: Do not list every possible methods, but DO include those from DIFFERENT library ecosystems when they offer distinct algorithms for the problem.
        - **CRITICAL**: ONLY list functions that directly SOLVE the problem . Do NOT list low-level building blocks (linear algebra routines, random generators, data structures, geometry computations, plotting, etc.).
        - For optimization problems specifically:
            - First identify the constraints on the actual decision variables (not evaluation metrics like combined_score), then include solvers for each constraint level present (bounds, general). Do not list solvers that cannot handle the problem's constraints.
            - **KEY**: Match the complexity of the tool to the problem structure. If the problem has complex interactions or constraints, the PRIMARY tool must be capable of handling that full complexity — simpler/restricted methods should only be listed as complementary sub-routines for easier sub-problems. When a lightweight specialised variant exists for a simpler sub-task (e.g., a final refinement step with fewer constraints), include it too — it can be significantly faster.
            - If the problem decomposes (e.g., fixing one set of variables makes the remaining problem simpler), list the methods for those sub-problems too.
    - **INCLUDE COMPLEMENTARY VARIANTS**: Standard libraries often have PAIRS for the same operation (e.g., local vs global, forward-only vs bidirectional, online vs batch). ALWAYS list **both** sides of each pair — never omit the complement.

2. HIGH-LEVEL IDEAS ONLY - short, abstract concepts (stay within 15 words).
    -  Within each category just list 3-5 alternative ideas. 
3. Tailor categories to the problem; try to focus on a few different categories, and within each category focus on actionable and simple approaches not parameter tweaks
4. Only assume constraints explicitly stated in the problem.
5. 3-6 bullet points, order them based on ease of implementation and relevance to the problem.
6. Use ↔ to separate alternatives 
7. **BALANCE**: Include libraries with SPECIFIC algorithms when they provide solutions, but EQUALLY emphasize domain-specific algorithmic strategies. Both are important.
8. If the evaluator passes input data for processing, add a **separate bullet** after Libraries: state that the full input is available and library functions can be applied directly to it.
"""

# ------------------------------------------------------------------
# EXPLOITATION PROMPT - for refine_operator (intensify within current approach)
# ------------------------------------------------------------------
EXPLOIT_SYSTEM_PROMPT = """\
You are an expert at analyzing problems and suggesting ways to INTENSIFY and REFINE within an existing approach.

Given a problem description and evaluator, generate a concise "refinement strategies" guidance block that will help an LLM squeeze more performance from the CURRENT approach without changing its fundamental structure.

## GOAL: EXPLOITATION, NOT EXPLORATION
The LLM receiving this guidance already has a working solution. Your job is to help it, for example:
- Tune hyperparameters more aggressively (iterations, batch size, population, etc.)
- Improve the quality of inputs (seeds, initializations, candidates)
- Improve the quality of the solution (post-processing, validation, polish)
- Refine tradeoff parameters: If the approach uses parameters to balance competing factors, tuning strategies in different ways (e.g., search ranges, convergence criteria, adaptive schedules)
- Explore composite metrics: If the approach combines multiple factors, varying different ways in how they're combined (e.g., weights, normalization, thresholds) 

## STEP 1: ANALYZE THE PROBLEM TYPE (do this internally, don't output)
Before suggesting anything, you MUST determine:
- What type of problem is this? 
- What are the "knobs" that can be tuned without changing the core algorithm? For example,
  - Tradeoff parameters: If the approach balances multiple factors, what parameters control the tradeoff? 
  - Composite metrics: If the approach combines multiple factors, how are they combined? 
  - Search strategies: If the approach searches for parameter values, what can be refined? 
- What does "better" mean for this problem? Read the evaluator's scoring function (e.g., `combined_score`) and understand ALL components.
  - **PRINCIPLE**: Only include suggestions that DIRECTLY improve a component of `combined_score`. 
    If something is NOT measured in the score (e.g., runtime when only quality matters), do NOT suggest directions that do not improve the score.
- What post-processing, validation, or polish stages could help? 
- If the problem objective is only to improve the quality of the solution, where can more computation budget be spent to improve the quality?

## STEP 2: GENERATE REFINEMENT STRATEGIES (based on problem & evaluator)
Your suggestions should be driven primarily by analyzing the PROBLEM and EVALUATOR from Step 1.
Think about what "knobs" exist in typical solutions for this problem type:
- Iteration counts, population sizes, step sizes, convergence thresholds
- Input quality (initializations, seeds, candidates)
- Post-processing and polish stages
- Tradeoff parameters and composite metric weights

## STEP 3: OUTPUT FORMAT (follow exactly)
```

EXAMPLES OF REFINEMENT strategies (NOT LIMITED TO, PROPOSE YOUR OWN):
- **[Category 1]**: [describe 3-5 specific tuning techniques]
- **[Category 2]**: [describe 3-5 specific tuning techniques]
- ... (3-5 bullet points total, only include categories that actually apply)
... Or anything else you can think of. 

```

EXAMPLE CATEGORIES (do not blindly copy, tailor to problem):
```
REFINEMENT strategies examples:
- **Libraries / tools**: (`library.submodule: func1, func2 ↔ another_lib: funcA`; for wrapper functions that accept a `method` parameter: you MUST specify the exact method names. Format: `wrapper(method='X'), wrapper(method='Y')`).
- **Computational budget**: 
- **Hyperparameter tuning**: 
- **Dynamic scheduling**: 
- **Multi-objective weights / tradeoff balancing**: 
- **Input quality / initialization**: 
- **Solver tolerances / convergence**: 
- **Post-processing / polish**: 
```
Do not just blindly copy from the example; tailor the categories and ideas specifically to the problem type and context.

## RULES:
1. **STRICT**: DO NOT suggest changing the core algorithm or problem formulation. You CAN suggest library tools that REFINE/POLISH within the same approach.
2. **TAILOR**: Only suggest categories that actually exist in this problem type - do NOT invent knobs that don't apply
    - Do NOT invent hyperparameters that don't exist in the code
3. **EVALUATOR ALIGNMENT**: Only suggest strategies that DIRECTLY improve metrics in the evaluator's scoring function
4. Keep each bullet concise (max 15 words per alternative)
5. 3-5 bullet points only
6. Use ↔ to separate alternatives
7. Only give high-level directions, not specific numbers
"""

# ------------------------------------------------------------------
# COMBINED PROMPT - generates both exploration and exploitation in one call
# ------------------------------------------------------------------
COMBINED_SYSTEM_PROMPT = f"""\
You must generate TWO separate guidance blocks in a single response. Output them as two clearly separated sections.

## OUTPUT STRUCTURE
Your response must have exactly two sections:
1. First section: "### EXPLORATION (diverge_label)" followed by the exploration guidance
2. Second section: "### EXPLOITATION (refine_label)" followed by the exploitation guidance

---

## INSTRUCTIONS FOR SECTION 1 (EXPLORATION):

{EXPLORE_SYSTEM_PROMPT}

---

## INSTRUCTIONS FOR SECTION 2 (EXPLOITATION):

{EXPLOIT_SYSTEM_PROMPT}

---

## CROSS-REFERENCE RULE
The EXPLOITATION section MUST be consistent with the EXPLORATION section AND must be self-sufficient (usable even if the user never sees the exploration section).
- If the EXPLORATION section suggests optimization solvers or libraries, it is recommended for the EXPLOITATION section to include: 
    - (1) a "Libraries / tools for refinement" category that is the same list and order as the one in the EXPLORATION section
          - ALSO INCLUDE ADDITIONAL optimizers, libraries, and/or polish methods for polishing and refining existing solutions.
    - (2) solver-specific refinement knobs: for example, maxiter, convergence tolerances, different initializations and polish passes. 
- The EXPLOITATION section should contain enough concrete, actionable ideas that the LLM can make meaningful improvements even without seeing the EXPLORATION section.
- These can be general knobs, not specific numbers.

Generate BOTH sections now, clearly separated with the headers above.
"""

from skydiscover.search.evox.utils.template import DIVERGE_TEMPLATE, REFINE_TEMPLATE

# Backwards-compatible aliases
DEFAULT_DIVERGE_TEMPLATE = DIVERGE_TEMPLATE
DEFAULT_REFINE_TEMPLATE = REFINE_TEMPLATE


def load_config(config_path: str) -> dict:
    """Load and return the config.yaml contents."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_evaluator(evaluator_path: str) -> str:
    """Load and return the evaluator.py contents."""
    with open(evaluator_path, "r") as f:
        return f.read()


def load_initial_program(initial_program_path: str) -> str:
    """Load and return the initial_program.py contents."""
    with open(initial_program_path, "r") as f:
        return f.read()


def get_available_packages(problem_dir=None) -> list:
    """Get list of available packages from requirements.txt or pyproject.toml (direct dependencies only)."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[4]

    # Priority 1: requirements.txt in problem directory
    if problem_dir is not None:
        problem_dir = Path(problem_dir)
        requirements_path = problem_dir / "requirements.txt"
        if requirements_path.exists():
            try:
                with open(requirements_path, "r") as f:
                    lines = f.readlines()
                packages = []
                for line in lines:
                    line = line.strip()
                    if (
                        not line
                        or line.startswith("#")
                        or line.startswith("-e")
                        or line.startswith("--")
                    ):
                        continue
                    packages.append(line)
                if packages:
                    logger.info(f"Read {len(packages)} packages from {requirements_path}")
                    return packages
            except Exception as e:
                logger.warning(f"Could not read {requirements_path} ({e}), trying repo root")

    # Priority 2: requirements.txt at repo root
    requirements_path = repo_root / "requirements.txt"
    if requirements_path.exists():
        try:
            with open(requirements_path, "r") as f:
                lines = f.readlines()
            packages = []
            for line in lines:
                line = line.strip()
                if (
                    not line
                    or line.startswith("#")
                    or line.startswith("-e")
                    or line.startswith("--")
                ):
                    continue
                packages.append(line)
            if packages:
                logger.info(f"Read {len(packages)} packages from {requirements_path}")
                return packages
        except Exception as e:
            logger.warning(f"Could not read requirements.txt ({e}), trying pyproject.toml")

    # Priority 3: pyproject.toml
    try:
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                raise ImportError("tomllib/tomli not available")

        pyproject_path = repo_root / "pyproject.toml"
        if not pyproject_path.exists():
            raise FileNotFoundError(f"pyproject.toml not found at {pyproject_path}")

        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        dependencies = data.get("project", {}).get("dependencies", [])
        return dependencies
    except (ImportError, FileNotFoundError, KeyError) as e:
        logger.warning(f"Could not read pyproject.toml ({e}), falling back to uv pip list")
        try:
            result = subprocess.run(
                ["uv", "pip", "list", "--format", "json"],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            packages = json.loads(result.stdout)
            return [f"{pkg['name']}=={pkg['version']}" for pkg in packages]
        except Exception as e2:
            logger.warning(f"Could not fetch package list: {e2}")
            return []


def _parse_combined_response(response: str) -> Tuple[str, str]:
    """Parse the combined LLM response to extract structural variation (explore broadly) and local refinement (exploit within current approach) sections."""
    exploration = ""
    exploitation = ""

    lines = response.split("\n")
    current_section = None
    current_lines = []

    for line in lines:
        line_upper = line.upper().strip()
        if "### EXPLORATION" in line_upper or "EXPLORATION (DIVERGE" in line_upper:
            if current_section == "exploitation":
                exploitation = "\n".join(current_lines)
            current_section = "exploration"
            current_lines = []
        elif "### EXPLOITATION" in line_upper or "EXPLOITATION (REFINE" in line_upper:
            if current_section == "exploration":
                exploration = "\n".join(current_lines)
            current_section = "exploitation"
            current_lines = []
        elif current_section:
            current_lines.append(line)

    if current_section == "exploration":
        exploration = "\n".join(current_lines)
    elif current_section == "exploitation":
        exploitation = "\n".join(current_lines)

    exploration = _extract_examples(exploration, is_diverge=True)
    exploitation = _extract_examples(exploitation, is_diverge=False)

    return exploration, exploitation


def _extract_examples(response: str, is_diverge: bool = True) -> str:
    """Extract the examples section from LLM response."""
    lines = response.strip().split("\n")
    examples_lines = []
    in_examples = False

    for line in lines:
        if is_diverge and "EXAMPLES OF DIFFERENT" in line.upper():
            in_examples = True
            examples_lines.append(line)
        elif not is_diverge and "EXAMPLES OF REFINEMENT" in line.upper():
            in_examples = True
            examples_lines.append(line)
        elif in_examples:
            if line.strip().startswith("Format:") or line.strip().startswith("Your solution"):
                break
            if line.strip() in ("```", "```\n"):
                continue
            examples_lines.append(line)

    if examples_lines:
        while examples_lines and not examples_lines[-1].strip():
            examples_lines.pop()
        return "\n".join(examples_lines)

    return response.strip()


# ------------------------------------------------------------------
# PUBLIC API — generate_variation_operators (programmatic use)
# ------------------------------------------------------------------


def _build_operator_prompt(
    system_message: str,
    evaluator_code: str,
    problem_dir: Optional[str] = None,
    initial_program_solution: Optional[str] = None,
) -> str:
    """Build the user prompt for variation operator generation."""
    available_packages = get_available_packages(problem_dir=problem_dir)
    packages_list = "\n".join(available_packages) if available_packages else "No packages found"

    initial_program_section = ""
    if initial_program_solution:
        initial_program_section = f"""

## Initial Program (Reference Implementation)
The following is a very simple reference implementation program that will be evolved:
```python
{initial_program_solution}
```
This shows the current approach and structure. Use this to understand what exists but do not over-rely on the structure of the reference implementation."""

    context = f"""## Problem Description:
```
{system_message}
```

## Available Packages in Environment
The following packages are available in the current uv environment:
```
{packages_list}
```

## Evaluator Code:
```python
{evaluator_code}
```{initial_program_section}"""

    return f"""Please analyze this problem and generate BOTH guidance blocks.

{context}

Generate BOTH the EXPLORATION (different approaches) and EXPLOITATION (refinement/intensification) guidance blocks now.

For EXPLORATION guidance block, focus on DIFFERENT algorithmic approaches and structural changes.
For EXPLOITATION guidance block, focus on INTENSIFYING within existing approaches - e.g., computational budget (e.g., increase max iterations), better seeds, tighter tolerances, local polish stages.
"""


def _operators_from_response(combined_response: str) -> Tuple[str, str]:
    """Parse LLM response and build diverge/refine variation operators."""
    explore_examples, refine_examples = _parse_combined_response(combined_response)
    diverge_operator = DIVERGE_TEMPLATE.replace("{GENERATED_EXAMPLES}", explore_examples)
    refine_operator = REFINE_TEMPLATE.replace("{GENERATED_EXAMPLES}", refine_examples)
    return diverge_operator, refine_operator


async def generate_variation_operators(
    system_message: str,
    evaluator_code: str,
    problem_dir: Optional[str] = None,
    initial_program_solution: Optional[str] = None,
    llm_pool: Optional[LLMPool] = None,
) -> Tuple[str, str]:
    """Generate problem-specific variation operators (e.g. structural variation or local refinement).

    Args:
        system_message: Problem description (from config).
        evaluator_code: The evaluator.py source code.
        problem_dir: Optional path to problem directory (for requirements.txt).
        initial_program_solution: Optional initial program source code for additional context.
        llm_pool: LLMPool to use for generation.

    Returns:
        (structural_variation_label, local_refinement_label)
    """
    user_prompt = _build_operator_prompt(
        system_message,
        evaluator_code,
        problem_dir,
        initial_program_solution,
    )

    result = await llm_pool.generate(
        system_message=COMBINED_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    return _operators_from_response(result.text)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
DEFAULT_CLI_MODEL = "gpt-5-mini"
DEFAULT_CLI_MAX_TOKENS = 8000
DEFAULT_CLI_TIMEOUT = 300


def main():
    parser = argparse.ArgumentParser(
        description="Generate problem-specific variation operators (e.g. structural variation operator and local refinement operator)"
    )
    parser.add_argument(
        "problem_dir",
        type=str,
        help="Path to the problem directory containing config.yaml and evaluator.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (e.g. variation_operators.txt)",
    )
    parser.add_argument(
        "--provide-initial",
        action="store_true",
        default=False,
        help="Include initial_program.py as additional context for variation operator generation",
    )
    args = parser.parse_args()

    # Build paths
    config_path = os.path.join(args.problem_dir, "config.yaml")
    evaluator_path = os.path.join(args.problem_dir, "evaluator.py")

    # Validate
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        return 1
    if not os.path.exists(evaluator_path):
        print(f"Error: Evaluator file not found: {evaluator_path}")
        return 1

    # Load config and evaluator
    print(f"Loading config from: {config_path}")
    config_content = load_config(config_path)
    system_message = config_content.get("prompt", {}).get("system_message", "")

    print(f"Loading evaluator from: {evaluator_path}")
    evaluator_code = load_evaluator(evaluator_path)

    # Optionally load initial program
    initial_program_solution = None
    if args.provide_initial:
        initial_program_path = os.path.join(args.problem_dir, "initial_program.py")
        if os.path.exists(initial_program_path):
            print(f"Loading initial program from: {initial_program_path}")
            initial_program_solution = load_initial_program(initial_program_path)
        else:
            print(f"Warning: --provide-initial set but {initial_program_path} not found, skipping")

    # Build LLMPool for CLI usage
    from skydiscover.config import LLMModelConfig
    from skydiscover.llm.llm_pool import LLMPool

    model_cfg = LLMModelConfig(
        name=DEFAULT_CLI_MODEL,
        api_base="https://api.openai.com/v1",
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        max_tokens=DEFAULT_CLI_MAX_TOKENS,
        timeout=DEFAULT_CLI_TIMEOUT,
        retries=3,
        retry_delay=5,
    )
    llm = LLMPool([model_cfg])

    # Generate variation operator labels
    print(f"Generating variation operators with model={DEFAULT_CLI_MODEL}...")
    diverge_operator, refine_operator = asyncio.run(
        generate_variation_operators(
            system_message=system_message,
            evaluator_code=evaluator_code,
            problem_dir=args.problem_dir,
            initial_program_solution=initial_program_solution,
            llm_pool=llm,
        )
    )

    # Output
    output_text = f"### STRUCTURAL VARIATION OPERATOR \n{diverge_operator}\n\n### LOCAL REFINEMENT OPERATOR \n{refine_operator}"
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_text)
        print(f"Saved variation operators to: {args.output}")
    else:
        print("\n" + "=" * 80)
        print(output_text)
        print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
