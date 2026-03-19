"""
ParadigmGenerator - LLM-based breakthrough idea generation.

Generates structured paradigm ideas using a 6-step analysis framework.
Takes problem context, evaluator code, and current best program to
produce actionable breakthrough ideas.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from skydiscover.llm.llm_pool import LLMPool

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1.0
BACKOFF_MULTIPLIER = 2.0


class ParadigmGenerator:
    """
    Generates breakthrough paradigms using LLM analysis.

    Uses a structured 6-step analysis framework:
    1. Understand the task
    2. Analyze the evaluator code
    3. Identify metrics
    4. Identify constraints
    5. Identify problem structure
    6. Identify improvement opportunities

    Output format per paradigm:
    {
        "idea": "Short description of the breakthrough idea",
        "description": "Detailed implementation guide",
        "what_to_optimize": "Target metric from evaluator",
        "cautions": "Important implementation details",
        "approach_type": "library.function format"
    }
    """

    def __init__(
        self,
        llm_pool: LLMPool,
        system_message: str = "",
        evaluator_code: str = "",
        num_paradigms: int = 3,
        eval_timeout: int = 300,
        language: str = "python",
        objective_names: Optional[List[str]] = None,
        higher_is_better: Optional[Dict[str, bool]] = None,
        fitness_key: Optional[str] = None,
    ):
        """
        Initialize the paradigm generator.

        Args:
            llm_pool: LLM pool for generation
            system_message: Problem description from config
            evaluator_code: Evaluator source code
            num_paradigms: Number of paradigms to generate per call
            eval_timeout: Evaluation timeout in seconds
            language: Language of the solution being evolved ("python" for code, "image" for images, etc.)
        """
        self.llm_pool = llm_pool
        self.system_message = system_message
        self.evaluator_code = evaluator_code
        self.num_paradigms = num_paradigms
        self.eval_timeout = eval_timeout
        self.language = language
        self._is_image_mode = language.lower() == "image"
        self._is_prompt_optimization = language.lower() in ("text", "prompt", "image")
        self.objective_names = list(objective_names or [])
        self.higher_is_better = dict(higher_is_better or {})
        self.fitness_key = fitness_key

    def _is_multiobjective(self) -> bool:
        """Return True when explicit Pareto objectives are configured."""
        return bool(self.objective_names)

    def _score_label(self) -> str:
        """Label for the numeric score shown in prompts."""
        return "proxy score" if self._is_multiobjective() else "score"

    def _optimization_targets_text(self) -> str:
        """Describe what the paradigms should optimize."""
        if not self._is_multiobjective():
            return "Optimize the primary scalar score defined by the evaluator."

        parts = []
        for objective in self.objective_names:
            direction = "maximize" if self.higher_is_better.get(objective, True) else "minimize"
            parts.append(f"{objective} ({direction})")

        text = "Optimize the Pareto trade-offs across: " + ", ".join(parts) + "."
        if self.fitness_key:
            text += (
                f" Use `{self.fitness_key}` only as a scalar proxy when one score is needed for"
                " ranking, stagnation detection, or tie-breaking."
            )
        return text

    async def generate(
        self,
        current_program_solution: str,
        current_best_score: float,
        previously_tried_ideas: Optional[List[str]] = None,
        evaluator_feedback: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate breakthrough paradigms with retry logic.

        Args:
            current_program_solution: Current best program solution
            current_best_score: Current best score
            previously_tried_ideas: List of previously tried approaches
            evaluator_feedback: Optional diagnostic feedback from evaluator artifacts

        Returns:
            List of paradigm dicts with keys:
            idea, description, what_to_optimize, cautions, approach_type
        """
        prompt = self._build_prompt(
            current_program_solution,
            current_best_score,
            previously_tried_ideas or [],
            evaluator_feedback=evaluator_feedback,
        )

        last_error = None
        backoff = INITIAL_BACKOFF_SECONDS

        for attempt in range(MAX_RETRIES):
            try:
                result = await self.llm_pool.generate(
                    system_message=self._get_system_message(),
                    messages=[{"role": "user", "content": prompt}],
                )
                response = result.text

                if not response:
                    logger.warning(f"Empty response from LLM (attempt {attempt + 1}/{MAX_RETRIES})")
                    last_error = "Empty response"
                    # Don't retry for empty response - likely a parsing issue
                    break

                paradigms = self._parse_response(response)

                if not paradigms:
                    logger.warning(
                        f"Failed to parse paradigms (attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    last_error = "Parse failure"
                    # Don't retry parse failures - the prompt needs fixing
                    break

                logger.info(f"Generated {len(paradigms)} paradigms:")
                for i, p in enumerate(paradigms):
                    logger.info(
                        f"  [{i+1}] {p.get('idea', 'N/A')} (approach: {p.get('approach_type', 'N/A')})"
                    )
                return paradigms

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Paradigm generation failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}"
                )

                if attempt < MAX_RETRIES - 1:
                    logger.info(f"Retrying in {backoff:.1f}s...")
                    await asyncio.sleep(backoff)
                    backoff *= BACKOFF_MULTIPLIER

        logger.error(f"Paradigm generation failed after {MAX_RETRIES} attempts: {last_error}")
        return []

    def _get_system_message(self) -> str:
        """Get system message for paradigm generation."""
        if self._is_image_mode:
            return (
                "You are an expert visual artist and image prompt engineer. "
                "Think carefully and deeply about visual composition, spatial layout, "
                "and how image generation models interpret text prompts. "
                "Analyze the evaluation rubric thoroughly and suggest breakthrough "
                "prompt strategies that will actually improve the generated images. "
                "Focus on strategies that are fundamentally different from what has been tried."
            )
        elif self._is_prompt_optimization:
            return (
                "You are an expert prompt engineer and LLM researcher. Think carefully "
                "and deeply. Analyze the current prompt, understand the evaluation "
                "pipeline by reading the evaluator code, and suggest breakthrough "
                "prompt strategies that are actionable and will improve accuracy. "
                "Focus on strategies that are fundamentally different from what has "
                "been tried."
            )
        return (
            "You are an expert algorithm researcher. Think carefully and deeply. "
            "Analyze the problem thoroughly, understand the evaluation metric "
            "by reading the evaluator code, and suggest breakthrough ideas that are "
            "correct, actionable, and will actually help improve the solution. "
            "Focus on ideas that are fundamentally different from what has been tried."
        )

    def _build_prompt(
        self,
        program_solution: str,
        best_score: float,
        previously_tried: List[str],
        evaluator_feedback: Optional[str] = None,
    ) -> str:
        """Build the full prompt for paradigm generation."""
        if self._is_prompt_optimization:
            sections = [
                self._build_prompt_opt_context(program_solution, best_score),
                self._build_prompt_opt_analysis(best_score),
                self._build_prompt_opt_analysis_framework(best_score),
                self._build_previously_tried_section(previously_tried),
                self._build_prompt_opt_techniques_section(),
                self._build_prompt_opt_output_format_section(),
            ]
        else:
            sections = [
                self._build_problem_context(program_solution, best_score),
                self._build_current_program_analysis(best_score),
                self._build_analysis_framework(best_score),
                self._build_previously_tried_section(previously_tried),
                self._build_techniques_section(),
                self._build_output_format_section(),
            ]

        # Inject evaluator feedback so paradigm ideas are informed by
        # specific failure modes identified by the evaluator.
        if evaluator_feedback:
            max_len = 2000
            if len(evaluator_feedback) > max_len:
                evaluator_feedback = evaluator_feedback[:max_len] + "\n... (truncated)"
            sections.insert(
                -1,  # before the output format section
                f"## Evaluator Feedback on Current Best Program\n"
                f"The evaluator analyzed cases where the current program fails. "
                f"Use this to inform your breakthrough ideas:\n\n"
                f"{evaluator_feedback}",
            )

        return "\n\n".join(sections)

    def _build_current_program_analysis(self, best_score: float) -> str:
        """Build the current program analysis directive."""
        return f"""**CRITICAL: ANALYZE THE CURRENT PROGRAM FIRST**
Before suggesting new ideas, carefully analyze the Current Program above:
- What algorithm/approach does it use? (This is what's WORKING - {self._score_label()} {best_score:.6f})
- What are its strengths? (Why does it achieve this {self._score_label()}?)
- What are its weaknesses? (What limits further improvement?)
- How can you improve it? (How to beat it?)

**IMPORTANT:** The program above is the CURRENT program that needs to be improved. Start by understanding what works, then suggest breakthrough ideas that build on or improve it."""

    def _build_problem_context(self, program_solution: str, best_score: float) -> str:
        """Build the problem context section."""
        if self._is_image_mode:
            return f"""## Problem Objective

{self.system_message}

## Optimization Targets

{self._optimization_targets_text()}

## Evaluator Code (shows how images are scored)

```python
{self.evaluator_code if self.evaluator_code else "N/A - evaluator code not provided"}
```

## Current Best Image Prompt ({self._score_label()}: {best_score:.6f})

{program_solution if program_solution else "N/A"}

**CRITICAL:** Analyze the current prompt first. What visual elements does it describe?
What details are present vs missing? Which rubric categories does it address well vs poorly?
How can the prompt be restructured to produce a better image?"""

        return f"""## Problem Objective

{self.system_message}

## Optimization Targets

{self._optimization_targets_text()}

## Evaluator Code (shows how solutions are scored)

```python
{self.evaluator_code if self.evaluator_code else "N/A - evaluator code not provided"}
```

## Current Best Program ({self._score_label()}: {best_score:.6f})

```python
{program_solution if program_solution else "N/A"}
```

**CRITICAL:** Analyze the current program first. What algorithm does it use?
What are its strengths and weaknesses? How can you improve upon it?"""

    def _build_analysis_framework(self, best_score: float) -> str:
        """Build the 6-step analysis framework section."""
        if self._is_image_mode:
            return self._build_image_analysis_framework(best_score)
        return f"""## Analysis Framework - Complete Before Generating Ideas

**STEP 0: Understand the TASK (MOST IMPORTANT - DO THIS FIRST)**
- What is the problem asking you to do?
- What is the goal or objective? (maximize, minimize, optimize)
- What are the inputs and outputs?
- What needs to be improved? (variables/decisions that affect the goal)
- What constraints exist?

**STEP 1: Analyze the Evaluator Code**
- How are solutions scored?
- What metrics are computed?
- What causes failures or penalties?

**STEP 2: Identify Metrics**
- What is the primary metric or Pareto objective set?
- How is it calculated?
- What secondary metrics exist?
- If variance/std is penalized, the program needs consistency across scenarios

**STEP 3: Identify Constraints**
- What conditions must be satisfied?
- What validation happens?
- What causes score penalties?

**STEP 4: Identify Problem Structure**
- Is processing sequential or global?
- Are decision variables discrete or continuous?
- What dependencies exist between decisions?
- **CRITICAL:** What data does your program receive vs what the evaluator uses?
- **CRITICAL:** How are metrics computed across components? Independently then aggregated, or jointly?

**STEP 5: Determine Appropriate Approach**
- Match approach to problem structure
- Consider what has worked vs failed before
- Identify promising library/technique combinations

**STEP 6: Identify Improvement Opportunities**
- What would increase each metric?
- What would satisfy constraints better?
- What fundamentally different approaches could work?

Current best {self._score_label()} is {best_score:.6f}. Your ideas must improve the configured optimization targets and, in multiobjective mode, explicitly reason about objective trade-offs."""

    def _build_image_analysis_framework(self, best_score: float) -> str:
        """Build image-specific analysis framework."""
        return f"""## Analysis Framework (Image Mode) - Complete Before Generating Ideas

**STEP 0: Understand the VISUAL TASK (MOST IMPORTANT)**
- What scene/image is being requested?
- What specific visual elements are required? (objects, counts, arrangements)
- What text/labels must appear in the image?
- What spatial relationships are specified?

**STEP 1: Analyze the Evaluation Rubric**
- How is the image scored? (read the evaluator code)
- What categories/dimensions are evaluated?
- What specific counts, colors, positions does the judge look for?
- What causes low scores in each category?

**STEP 2: Identify Weakest Categories**
- Which rubric categories score lowest in the current best?
- Which elements are consistently missing or wrong?
- Which details does the image model struggle with most?

**STEP 3: Understand Image Model Limitations**
- Image models struggle with: exact counts, legible text, precise spatial layouts
- Image models are good at: mood, color, style, general composition, prominent objects
- Which required elements hit model limitations?

**STEP 4: Analyze Current Prompt Structure**
- How is the prompt organized? (single block vs sections vs enumerated)
- Does it emphasize the right things? (models weight early text more heavily)
- Are critical details buried or prominent?
- Is the prompt too long or too short?

**STEP 5: Identify Prompt Engineering Strategies**
- What structural changes could help? (reordering, sectioning, emphasis)
- What description strategies could improve counts? (explicit enumeration, spatial anchoring)
- What style/medium changes might help with specific elements?

**STEP 6: Prioritize Improvements**
- Which categories have the most room for improvement?
- Which improvements are achievable given model limitations?
- What prompt changes would have the highest impact?

Current best score is {best_score:.6f}. Your strategies must be capable of improving this."""

    def _build_previously_tried_section(self, previously_tried: List[str]) -> str:
        """Build the previously tried ideas section."""
        if not previously_tried:
            return """## Previously Tried Ideas

No previous paradigms have been tried yet. You have freedom to explore any approach."""

        formatted = "\n".join(f"- {idea}" for idea in previously_tried)
        return f"""## Previously Tried Ideas - CHECK THIS FIRST

**CRITICAL:** Review what was already tried. Do NOT suggest ideas that use
the same libraries, functions, or approaches as FAILED attempts.

{formatted}

**STRICT PROHIBITION:** Do NOT keep suggesting approaches that have already failed.
If an approach failed, understand WHY before suggesting similar techniques.
Prioritize approaches that are fundamentally different from failed attempts.

**Learning from Failures - Understand Root Causes:**
When a technique fails badly (score decreased significantly), understand WHY before suggesting alternatives:
- **Fundamental mismatch:** Wrong problem type (e.g., continuous optimizer on discrete problem) -> avoid that entire class of approaches
- **Structural mismatch:** Wrong approach for problem structure (e.g., linear proxy for non-linear objective) -> use approaches that match the actual structure
- **Implementation issues:** If the same library failed multiple times or very badly (>10% decrease), it likely indicates a fundamental mismatch - suggest a different class of approaches"""

    def _build_techniques_section(self) -> str:
        """Build the techniques guidance section."""
        if self._is_image_mode:
            return self._build_image_techniques_section()
        return """## Technique Guidance

**Note:** Standard scientific libraries (scipy, numpy, etc.) are available. PyTorch and TensorFlow are not available.

**For Continuous Optimization with Constraints:**
- scipy.optimize.minimize with constraint handling (SLSQP, trust-constr)
- Multiple initial guesses for global optimization
- Geometric approaches (Voronoi, convex hull)

**For Discrete/Combinatorial Problems:**
- Greedy heuristics with good ordering
- Local search (swaps, moves)
- scipy.optimize.linear_sum_assignment for assignment problems
- scipy.optimize.linprog for linear constraints

**For Graph/Network Problems:**
- NetworkX algorithms (shortest path, min spanning tree, flow)
- Spectral methods (eigenvalue-based ordering)

**For Repair/Reconstruction:**
- Heuristic-based detection and correction
- Structural constraint exploitation
- Averaging/interpolation for consistency

**For Robust Filtering/Noise Reduction:**
- scipy.signal (medfilt, savgol_filter, wiener) for direct filtering
- Use methods that handle outliers better than mean-based (median, percentile)
- Do NOT use scipy.optimize.minimize to tune filter parameters
- Use filtering functions directly, not multi-stage optimization

**General Principles:**
- Prefer single-function library calls over multi-stage pipelines
- Match algorithm to problem structure
- Simple approaches with good heuristics often beat complex methods

## ANTI-PATTERNS - Critical rules about what NOT to do

1. **Do NOT use multi-stage optimization**: Do NOT call one function then optimize its output. Deterministic setup code followed by a single optimization call is allowed.

2. **Do NOT use scipy.optimize.minimize for hyperparameter tuning**: Use minimize to solve the problem directly, NOT to tune parameters for another function.

3. **Do NOT use scipy.optimize.minimize for discrete problems**: Continuous optimizers cannot handle discrete constraint violations properly.

4. **Each idea MUST be a single-function library call**: Do NOT suggest multi-stage processing (e.g., "call A then call B").

**AVOID:** DEAP, genetic algorithm libraries, domain-specific complex libraries, custom research algorithms, or any library requiring additional `pip install`

**Learning from Success:**
When an approach succeeds, think: what principle made it work? Learn and think of better ideas, don't just add complexity. If breakthrough patterns are known, prioritize approaches that match them.

## DIVERSITY REQUIREMENTS

Before generating ideas, explicitly think:
- Idea 1: [Type A - e.g., algorithmic refinement or library-based approach]
- Idea 2: [Type B - e.g., structural change or processing pattern - DIFFERENT from A]
- Idea 3: [Type C - e.g., different technique or optimization method - DIFFERENT from A and B]

**Verify:** Are these DIFFERENT types? NOT variations of the same approach.

Each idea must:
- Use DIFFERENT libraries/techniques than failed attempts
- Target DIFFERENT metrics/aspects from the evaluator
- Be independently implementable
- Prefer clear implementations (different != more complex)

### Be Specific and Actionable

Not vague: "Try optimization"
Specific: "Use scipy.optimize.minimize with SLSQP method"

- Include exact library names, function names, methods, parameters
- Provide step-by-step implementation guide
- Focus on core logic that implements the idea correctly
- Handle edge cases and avoid errors/warnings
- For optimization: use multiple initializations, appropriate iteration counts and convergence criteria (evaluation timeout: {self.eval_timeout}s)"""

    def _build_image_techniques_section(self) -> str:
        """Build image-specific techniques guidance section."""
        return """## Prompt Engineering Techniques for Image Generation

**For Improving Object Counts & Specificity:**
- Explicit enumeration: number and describe each instance ("the first balloon is red with stripes, the second balloon is blue with dots, the third...")
- Spatial anchoring: place objects at specific locations ("in the top-left corner", "at the center")
- Grid/layout descriptions: describe the scene as zones or a grid
- Repetition emphasis: mention critical counts multiple times

**For Improving Spatial Arrangement:**
- Layered composition: describe background, midground, foreground separately
- Directional flow: describe the scene left-to-right or top-to-bottom
- Relative positioning: define objects in relation to each other ("to the right of X, below Y")
- Scene sectioning: divide the image into named regions and populate each

**For Improving Text/Labels in Images:**
- Prominent placement: make text elements the primary focus of a region
- Sign/banner framing: describe text on clear, high-contrast surfaces
- Simplify text: shorter text is more reliably rendered
- Style emphasis: "clearly legible text reading exactly..."

**For Improving Detail Accuracy:**
- Category isolation: dedicate a paragraph to each evaluation category
- Attribute chaining: attach all required attributes directly to each object
- Checklist-style: explicitly list each required detail as a bullet

**For Changing Overall Approach:**
- Art style shifts: try different mediums (digital painting, 3D render, illustration, watercolor)
- Perspective changes: bird's eye view, isometric, close-up vs wide shot
- Simplification: reduce scene complexity to improve accuracy on key elements
- Narrative framing: describe the scene as a story moment for better coherence

**ANTI-PATTERNS - What NOT to suggest:**
1. Do NOT suggest code/algorithmic approaches (scipy, numpy, ML training) - this is prompt engineering
2. Do NOT suggest using different image models - work with the current model
3. Do NOT suggest post-processing or image editing - only prompt changes
4. Do NOT suggest vague ideas like "make it better" - be specific about prompt structure changes

**General Principles:**
- Image models weight text at the beginning of prompts more heavily
- Fewer, more specific constraints are better than many vague ones
- Concrete visual descriptions beat abstract concepts
- Structural prompt changes (reordering, sectioning) often help more than adding words"""

    def _build_output_format_section(self) -> str:
        """Build the output format section."""
        if self._is_image_mode:
            return self._build_image_output_format_section()
        return f"""## Output Format

Generate {self.num_paradigms} breakthrough ideas of DIFFERENT types.

Each idea must be a JSON object with these fields:
- "idea": Clear, direct description with library/technique name
- "description": Detailed implementation guide (5-10 sentences)
- "what_to_optimize": What metrics/areas to focus on
- "cautions": Important implementation details to watch for
- "approach_type": Exact "library.function" format (e.g., "scipy.optimize.minimize")

**Diversity Requirement:** Each idea must use a DIFFERENT approach type.
Do not generate variations of the same technique.

Return ONLY a JSON array with {self.num_paradigms} paradigm objects. No other text.

Example:
```json
[
  {{
    "idea": "Use scipy.optimize.minimize with SLSQP",
    "description": "Apply scipy.optimize.minimize directly to optimize all variables together...",
    "what_to_optimize": "{', '.join(self.objective_names) if self.objective_names else 'primary evaluator score'}",
    "cautions": "Ensure constraints are properly formulated, use multiple starting points",
    "approach_type": "scipy.optimize.minimize"
  }}
]
```"""

    def _build_image_output_format_section(self) -> str:
        """Build image-specific output format section."""
        return f"""## Output Format

Generate {self.num_paradigms} breakthrough prompt strategies of DIFFERENT types.

Each strategy must be a JSON object with these fields:
- "idea": Clear description of the prompt engineering strategy
- "description": Detailed guide on how to restructure/rewrite the prompt (5-10 sentences)
- "what_to_optimize": Which rubric categories/visual elements to focus on
- "cautions": What to watch out for (e.g., don't lose existing good elements)
- "approach_type": Strategy category in "prompt.strategy_name" format

**Diversity Requirement:** Each strategy must use a FUNDAMENTALLY DIFFERENT approach.
Do not generate variations of the same technique.

Return ONLY a JSON array with {self.num_paradigms} strategy objects. No other text.

Example:
```json
[
  {{
    "idea": "Use explicit spatial anchoring with grid-based layout",
    "description": "Divide the scene into a 3x3 grid and assign each required element to a specific grid cell. Describe the contents of each cell in order (top-left to bottom-right). This helps the image model place objects precisely. For example: top-left contains 3 shaped clouds, top-center contains the banner, top-right contains 3 more clouds...",
    "what_to_optimize": "cloud_shapes, floating_island, spatial arrangement",
    "cautions": "Grid descriptions can feel rigid - add natural transitions between cells to maintain visual coherence",
    "approach_type": "prompt.spatial_grid"
  }}
]
```"""

    # =========================================================================
    # Prompt-Optimization-Specific Paradigm Methods
    # =========================================================================

    def _build_prompt_opt_context(self, prompt_text: str, best_score: float) -> str:
        """Build problem context for prompt optimization."""
        return f"""## Problem Objective

{self.system_message}

## Optimization Targets

{self._optimization_targets_text()}

## Evaluator Code (shows how prompts are scored)

```python
{self.evaluator_code if self.evaluator_code else "N/A - evaluator code not provided"}
```

## Current Best Prompt ({self._score_label()}: {best_score:.6f})

```text
{prompt_text if prompt_text else "N/A"}
```

**CRITICAL:** Analyze the current prompt first. What instruction strategy does it use?
What are its strengths and weaknesses? How can you improve upon it?"""

    def _build_prompt_opt_analysis(self, best_score: float) -> str:
        """Build analysis directive for prompt optimization."""
        return f"""**CRITICAL: ANALYZE THE CURRENT PROMPT FIRST**
Before suggesting new strategies, carefully analyze the current prompt above:
- What instruction approach does it use? (This is what's WORKING - {self._score_label()} {best_score:.6f})
- What are its strengths? (Clarity? Structure? Examples? Reasoning guidance?)
- What are its weaknesses? (Vague? Missing constraints? No examples? Poor format spec?)
- What would make the LLM perform better on this task?

**IMPORTANT:** The prompt above is the CURRENT prompt that needs improvement.
Understand what works, then suggest fundamentally different prompt strategies."""

    def _build_prompt_opt_analysis_framework(self, best_score: float) -> str:
        """Build analysis framework for prompt optimization."""
        return f"""## Analysis Framework - Complete Before Generating Ideas

**STEP 0: Understand the TASK (MOST IMPORTANT)**
- What task is the LLM being asked to perform?
- What inputs does the LLM receive? What output is expected?
- What makes this task hard? (reasoning steps, ambiguity, retrieval quality)

**STEP 1: Analyze the Evaluator Pipeline**
- How are prompts evaluated? (dataset, metric, scoring)
- What types of errors cause score loss? (wrong answer, wrong format, hallucination)
- How does retrieval interact with the prompt?

**STEP 2: Analyze Current Prompt Weaknesses**
- Is the instruction clear and unambiguous?
- Does it guide the LLM's reasoning process?
- Does it specify output format precisely?
- Does it handle edge cases (ambiguous questions, missing info)?

**STEP 3: Identify Improvement Dimensions**
- Instruction clarity and specificity
- Reasoning chain guidance (step-by-step, decomposition)
- Output format constraints
- Error prevention (hallucination guards, hedging strategies)
- Example inclusion (few-shot demonstrations)

**STEP 4: Design Breakthrough Strategies**
- What fundamentally different instruction approaches could work?
- What prompt engineering techniques haven't been tried?
- How can we better exploit the retrieval context?

Current best {self._score_label()} is {best_score:.6f}. Your ideas must improve the configured optimization targets and, in multiobjective mode, explicitly address evaluator trade-offs."""

    def _build_prompt_opt_techniques_section(self) -> str:
        """Build techniques guidance for prompt optimization."""
        return """## Prompt Engineering Techniques

**Reasoning & Decomposition:**
- Chain-of-thought: "Think step by step before answering"
- Multi-hop decomposition: "First identify relevant facts, then reason across them"
- Self-verification: "Check your answer against the passages before responding"
- Contrastive reasoning: "Consider why other answers might be wrong"

**Instruction Structure:**
- Role assignment: Give the LLM a specific expert persona
- Task decomposition: Break complex tasks into sub-steps within the prompt
- Explicit constraints: "Answer ONLY based on provided passages"
- Output format specification: "Respond with just the answer, no explanation"

**Few-Shot & Examples:**
- Include 1-3 worked examples showing input->reasoning->output
- Show examples of common error patterns and correct handling
- Demonstrate edge cases (unanswerable, ambiguous)

**Retrieval-Augmented Strategies:**
- Passage prioritization: "Focus on passages most relevant to the question"
- Evidence extraction: "Quote the specific evidence before answering"
- Multi-passage synthesis: "Combine information from multiple passages"

**Error Prevention:**
- Hallucination guards: "Only use information from the given passages"
- Confidence calibration: "If unsure, state the most likely answer"
- Format enforcement: "Your answer must be a short phrase, not a sentence"

**General Principles:**
- Be specific over generic — vague prompts lead to vague answers
- Structure matters — numbered steps outperform wall-of-text instructions
- Constraints prevent errors — explicit "do not" rules reduce hallucination
- Less can be more — overly long prompts can confuse the LLM

## ANTI-PATTERNS for Prompt Optimization

1. **Do NOT just rephrase the same instruction** — changing words without changing strategy is not a breakthrough
2. **Do NOT add irrelevant constraints** — constraints should target observed failure modes
3. **Do NOT make the prompt excessively long** — diminishing returns after a certain length
4. **Do NOT add examples that don't match the task** — misleading examples hurt performance

## DIVERSITY REQUIREMENTS

Before generating ideas, explicitly think:
- Idea 1: [Strategy A - e.g., reasoning structure change]
- Idea 2: [Strategy B - e.g., output format / constraint change - DIFFERENT from A]
- Idea 3: [Strategy C - e.g., few-shot / example-based - DIFFERENT from A and B]

**Verify:** Are these DIFFERENT strategy types? NOT variations of the same approach.

Each idea must:
- Use a DIFFERENT prompt engineering technique
- Target DIFFERENT aspects of LLM behavior
- Be independently implementable as a complete prompt
- Be specific and actionable (not just "improve clarity")"""

    def _build_prompt_opt_output_format_section(self) -> str:
        """Build output format for prompt optimization paradigms."""
        return f"""## Output Format

Generate {self.num_paradigms} breakthrough prompt strategies of DIFFERENT types.

Each idea must be a JSON object with these fields:
- "idea": Clear description of the prompt strategy
- "description": Detailed guide on how to write the new prompt (5-10 sentences)
- "what_to_optimize": What aspect of LLM behavior this targets
- "cautions": What to watch out for when implementing this strategy
- "approach_type": Category of the technique (e.g., "chain-of-thought", "few-shot", "format-constraint")

**Diversity Requirement:** Each idea must use a DIFFERENT strategy type.
Do not generate variations of the same technique.

Return ONLY a JSON array with {self.num_paradigms} paradigm objects. No other text.

Example:
```json
[
  {{
    "idea": "Add step-by-step multi-hop reasoning instructions",
    "description": "Restructure the prompt to explicitly guide the LLM through multi-hop reasoning. First identify key entities in the question, then find relevant facts about each entity in the passages, then chain the facts together to arrive at the answer. Include explicit instructions like: Step 1: Identify what the question is asking. Step 2: Find passages mentioning the key entities. Step 3: Extract relevant facts. Step 4: Combine facts to answer.",
    "what_to_optimize": "multi-hop reasoning accuracy",
    "cautions": "Keep steps concise. Too many steps can confuse the model. Ensure the steps match the actual reasoning pattern needed.",
    "approach_type": "chain-of-thought"
  }}
]
```"""

    def _parse_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse LLM response to extract paradigms.

        Args:
            response: Raw LLM response

        Returns:
            List of validated paradigm dicts
        """
        # Extract JSON from markdown code blocks if present
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            # Try to find JSON array in any code block
            parts = text.split("```")
            for part in parts[1::2]:  # Odd indices are inside code blocks
                part = part.strip()
                if part.startswith("["):
                    text = part
                    break

        try:
            paradigms = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse paradigm JSON: {e}")
            logger.debug(f"Response text: {text[:500]}")
            return []

        if not isinstance(paradigms, list):
            logger.warning(f"Expected list, got {type(paradigms)}")
            return []

        # Validate each paradigm
        validated = []
        required_keys = ["idea", "description", "approach_type"]

        for p in paradigms:
            if not isinstance(p, dict):
                continue

            # Check required keys
            if not all(k in p for k in required_keys):
                logger.debug(f"Paradigm missing required keys: {p}")
                continue

            # Ensure all expected keys exist with defaults
            validated.append(
                {
                    "idea": p.get("idea", ""),
                    "description": p.get("description", ""),
                    "what_to_optimize": p.get("what_to_optimize", "score"),
                    "cautions": p.get("cautions", ""),
                    "approach_type": p.get("approach_type", "unknown"),
                }
            )

        return validated
