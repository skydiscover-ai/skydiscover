"""Default templates for variation operators (structural variation and local refinement)."""

DIVERGE_TEMPLATE = """\
## IMPORTANT: YOU MUST FOLLOW THE FOLLOWING IN YOUR GENERATION, OTHERWISE THE SOLUTION WILL BE REJECTED.

### Goal
You MUST USE a FUNDAMENTALLY DIFFERENT APPROACH to the following code, NOT JUST parameter tweaks or local changes.
This may mean using different solver, function, or library, a different algorithm, or a different method entirely; or adding new components that have never been used before.
You can replace partial or the entire code if needed (i.e., rewrite a simpler solution from scratch), or build upon / complement it.
Think through what is needed to achieve a BREAKTHROUGH SCORE.

**Tools Available:**
You have access to a list of libraries. Use these tools to your advantage.
- Choose library functions appropriate to the problem's structure and constraints.
- Feel free to break from the existing code entirely — focus on correct library usage for the problem, not on preserving the current code's structure.
- **Prioritize libraries and functions not already used by the parent program.** Adding new components or changing how existing ones are combined is fine, but prefer approaches that introduce something the parent has never tried.

**CRITICAL: IF A LIBRARY EXISTS FOR AN ALGORITHM, USE IT. DO NOT REIMPLEMENT MANUALLY. SEE THE EXAMPLES BELOW FOR RELEVANT LIBRARIES.**

### Examples
{GENERATED_EXAMPLES}

### Output
Your solution MUST still be VALID / CORRECT for the same problem. A broken "creative" solution is worthless.

Format:
1. APPROACH: Describe your different approach.
2. CODE: Implementation.
"""

REFINE_TEMPLATE = """\
## IMPORTANT: YOU MUST FOLLOW THE FOLLOWING IN YOUR GENERATION, OTHERWISE THE SOLUTION WILL BE REJECTED.

### Goal
You should keep the same fundamental approach and algorithm structure.
DO NOT switch to a different problem formulation.
Focus on REFINING / EXPLOITING the current approach: better inputs, additional polish, etc.
Think through what is needed to SQUEEZE the last few percent of performance.
Use sufficient computational budget for refinement.

**Tools Available:**
You have access to a list of libraries. Use these tools to your advantage.
- Choose library functions appropriate to the problem's structure and constraints.
- Feel free to break from the existing code entirely — focus on correct library usage for the problem, not on preserving the current code's structure.

**CRITICAL: IF A LIBRARY EXISTS FOR AN ALGORITHM, USE IT. DO NOT REIMPLEMENT MANUALLY. SEE THE EXAMPLES BELOW FOR RELEVANT LIBRARIES.**

### Examples
{GENERATED_EXAMPLES}

### Output
Your solution MUST still be VALID / CORRECT for the same problem. A broken solution is worthless.

Format:
1. REFINEMENT: Describe what you are refining.
2. CODE: Implementation with refinement.
"""

# ------------------------------------------------------------------
# General, Default Variation Operator Templates
# ------------------------------------------------------------------
DEFAULT_DIVERGE_TEMPLATE = """\
## IMPORTANT: YOU MUST FOLLOW THE FOLLOWING IN YOUR GENERATION, OTHERWISE THE SOLUTION WILL BE REJECTED.

### Goal
Produce a **fundamentally different** solution than the current one.
This must be a real **strategy shift**, not minor edits or small tweaks.

Allowed: new structure, new strategy, different generation plan, different style/format, new components.
DO NOT do: superficial rewrites, polishing, or the same idea with only small changes.

### Constraints
- Keep the output **valid** for the same task and constraints.
- Prefer changes that introduce something **not present** in the following current solution.
- If reliable tools or shortcuts exist, use them instead of manual re-creation.

### Output
1. APPROACH: One short paragraph describing what is different and why.
2. OUTPUT: The full generated solution.
"""

DEFAULT_REFINE_TEMPLATE = """\
## IMPORTANT: YOU MUST FOLLOW THE FOLLOWING IN YOUR GENERATION, OTHERWISE THE SOLUTION WILL BE REJECTED.

### Goal
Improve the current solution **within the same core structure**.
Do **not** change the fundamental structure of the following solution; instead make it stronger, cleaner, and more reliable.

Allowed: rewriting for clarity, fixing weaknesses, improving completeness, better edge-case handling, and added polish.
DO NOT switch to a fundamentally different approach.

### Constraints
- Keep the output **valid** for the same task and constraints.
- Spend effort on higher-quality execution and fewer mistakes.

### Output
1. REFINEMENT: One short paragraph describing what you improved and how.
2. OUTPUT: The full refined solution.
"""
