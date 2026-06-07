"""
Qualitative analyzer — orchestrates 6 LLM judge steps (A–F) that run after
the deterministic quantitative pass. Receives a QuantitativeBundle and returns
a QualitativeBundle.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from skydiscover.extras.evolve_analyzer.llm.cache import cache_call
from skydiscover.extras.evolve_analyzer.llm.client import LLMClient, get_llm_client
from skydiscover.extras.evolve_analyzer.llm.parallel import ParallelResult, run_parallel
from skydiscover.extras.evolve_analyzer.quantitative.bundle import (
    IterationSummary,
    QuantitativeBundle,
    StagnationPeriod,
)

logger = logging.getLogger(__name__)

# ── QualitativeBundle ─────────────────────────────────────────────────────────


@dataclass
class QualitativeBundle:
    stagnation_analyses: List[dict]   # Step A: one dict per stagnation period
    artifact_clusters: List[dict]     # Step B: list of pattern dicts
    mutation_quality: List[dict]      # Step C: per-iteration quality assessments
    semantic_compliance: List[dict]   # Step D: per-iteration compliance checks
    exploration_structure: dict       # Step E: overall exploration assessment
    meta_quality: dict                # Step F: reasoning trace quality


# ── Constants ─────────────────────────────────────────────────────────────────

_STAGNATION_CATEGORIES = {
    "LOCAL_OPTIMUM",
    "INSTRUCTION_CONFUSION",
    "APPROACH_EXHAUSTION",
    "FORMAT_ISSUE",
    "EVALUATOR_NOISE",
    "OTHER",
}

_MAX_PROMPT_CHARS = 2000
_MAX_SAMPLE_SIZE = 50

SKIP_REASON_FULL_REWRITE = "full_code_rewrite_no_diff"
SKIP_REASON_NO_TRACE = "no_reasoning_trace"


def _truncate(text: str, max_chars: int = _MAX_PROMPT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "… [truncated]"


def _extract_json(text: str) -> Optional[dict]:
    """Extract JSON from a response that may contain markdown code fences."""
    # Try stripping markdown code block
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        text = match.group(1)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


# ── Main class ────────────────────────────────────────────────────────────────


class QualitativeAnalyzer:
    def __init__(
        self,
        llm_client: LLMClient,
        config: dict,
        cache_dir: str = ".evolve_cache",
    ) -> None:
        self.llm_client = llm_client
        self.config = config
        self.cache_dir = cache_dir
        self._step_clients: dict[str, LLMClient] = {}

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, quant: QuantitativeBundle) -> QualitativeBundle:
        """Orchestrate all judge steps. Respects config['judges'] enable/disable flags."""
        judges_cfg: dict = self.config.get("judges", {})
        df = quant.df

        stagnation_analyses: List[dict] = []
        if judges_cfg.get("stagnation_root_cause", True):
            stagnation_analyses = self._eval_stagnation_periods(
                quant.stagnation_periods, df
            )

        artifact_clusters: List[dict] = []
        if judges_cfg.get("artifact_clustering", True):
            artifact_clusters = self._eval_artifact_clusters(df)

        mutation_quality: List[dict] = []
        if judges_cfg.get("mutation_quality", True):
            mutation_quality = self._eval_mutation_quality(df)

        semantic_compliance: List[dict] = []
        if judges_cfg.get("semantic_compliance", True):
            semantic_compliance = self._eval_semantic_compliance(df)

        exploration_structure: dict = {}
        if judges_cfg.get("exploration_structure", True):
            exploration_structure = self._eval_exploration_structure(df)

        meta_quality: dict = {}
        if judges_cfg.get("meta_quality", True):
            meta_quality = self._eval_meta_quality(df)

        return QualitativeBundle(
            stagnation_analyses=stagnation_analyses,
            artifact_clusters=artifact_clusters,
            mutation_quality=mutation_quality,
            semantic_compliance=semantic_compliance,
            exploration_structure=exploration_structure,
            meta_quality=meta_quality,
        )

    # ── Step A: Stagnation root-cause analysis ────────────────────────────────

    def _eval_stagnation_periods(
        self, periods: List[StagnationPeriod], df: pd.DataFrame
    ) -> List[dict]:
        alert_periods = [p for p in periods if p.is_alert]
        if not alert_periods:
            return []

        def _analyze_one(period: StagnationPeriod) -> dict:
            mask = (df.get("streak_id", pd.Series(dtype=str)) == period.streak_id)
            period_df = df[mask] if mask.any() else df.iloc[0:0]

            # Build sequence table
            sequence_rows = []
            for it_sum in period.failure_sequence:
                sequence_rows.append(
                    f"  iter={it_sum.iteration} type={it_sum.mutation_type} "
                    f"failure={it_sum.failure_type} delta={it_sum.score_delta:.4f}"
                )
            sequence_table = "\n".join(sequence_rows) if sequence_rows else "(none)"

            # Gather artifacts
            artifacts_parts: List[str] = []
            for col in ("evaluator_artifacts", "build_warnings", "llm_feedback"):
                if col in period_df.columns:
                    vals = period_df[col].dropna().astype(str).tolist()
                    if vals:
                        artifacts_parts.append(f"[{col}]\n" + "\n".join(vals))
            artifacts_text = _truncate("\n\n".join(artifacts_parts) or "(none)")

            # Parent code
            parent_code = "(unavailable)"
            for col in ("parent_code", "parent", "code"):
                if col in period_df.columns and not period_df[col].dropna().empty:
                    parent_code = _truncate(str(period_df[col].dropna().iloc[0]))
                    break

            # System message
            system_message = "(unavailable)"
            for col in ("system_message", "system_prompt"):
                if col in period_df.columns and not period_df[col].dropna().empty:
                    system_message = _truncate(str(period_df[col].dropna().iloc[0]))
                    break

            prompt = (
                "You are analyzing a stagnation period in an evolutionary code optimization experiment.\n"
                f"The algorithm was stuck for {period.length} consecutive iterations with no progress.\n"
                f"System message given to the LLM: {system_message}\n"
                f"Parent code being mutated (unchanged across all iterations): {parent_code}\n"
                f"Sequence of what was attempted:\n{sequence_table}\n"
                f"Evaluator artifacts from this period: {artifacts_text}\n\n"
                "Answer:\n"
                "1. What approach was the LLM repeatedly trying?\n"
                "2. What constraint prevented progress?\n"
                "3. Was there a pattern in the failure types?\n"
                "4. What broke the stagnation (if recovery shown)?\n"
                "5. Categorize: LOCAL_OPTIMUM | INSTRUCTION_CONFUSION | APPROACH_EXHAUSTION | "
                "FORMAT_ISSUE | EVALUATOR_NOISE | OTHER\n"
                "6. One concrete recommendation."
            )

            try:
                raw = self._call_llm_cached(prompt, step_name="stagnation")
            except Exception as exc:
                logger.warning("Step A LLM call failed for streak %s: %s", period.streak_id, exc)
                return {
                    "streak_id": period.streak_id,
                    "category": "OTHER",
                    "approach_used": "",
                    "structural_constraint": "",
                    "failure_pattern": "",
                    "recovery_explanation": "",
                    "recommendation": "",
                    "raw_response": "",
                }

            # Parse numbered answers
            lines = raw.splitlines()

            def _extract_answer(num: int) -> str:
                prefix = f"{num}."
                collecting = False
                parts: List[str] = []
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith(prefix):
                        collecting = True
                        parts.append(stripped[len(prefix):].strip())
                    elif collecting and re.match(r"^\d+\.", stripped):
                        break
                    elif collecting:
                        parts.append(stripped)
                return " ".join(p for p in parts if p).strip()

            # Extract category from answer 5
            category = "OTHER"
            ans5 = _extract_answer(5)
            for cat in _STAGNATION_CATEGORIES:
                if cat in ans5.upper():
                    category = cat
                    break

            return {
                "streak_id": period.streak_id,
                "category": category,
                "approach_used": _extract_answer(1),
                "structural_constraint": _extract_answer(2),
                "failure_pattern": _extract_answer(3),
                "recovery_explanation": _extract_answer(4),
                "recommendation": _extract_answer(6),
                "raw_response": raw,
            }

        parallel_inputs = [(p,) for p in alert_periods]
        results: List[ParallelResult] = run_parallel(
            _analyze_one,
            parallel_inputs,
            use_async=False,
            progress_desc="Step A: stagnation analysis",
        )

        output: List[dict] = []
        for res in results:
            if res is not None and res.is_success and res.result is not None:
                output.append(res.result)
        return output

    # ── Step B: Evaluator artifact clustering ─────────────────────────────────

    def _eval_artifact_clusters(self, df: pd.DataFrame) -> List[dict]:
        # Collect failed iterations
        if "failure_mode" in df.columns:
            failed_df = df[df["failure_mode"] != "success"]
        else:
            failed_df = df

        if failed_df.empty:
            return []

        parts: List[str] = []
        for col in ("evaluator_artifacts", "build_warnings", "llm_feedback"):
            if col in failed_df.columns:
                vals = failed_df[col].dropna().astype(str).tolist()
                if vals:
                    combined = "\n".join(vals)
                    parts.append(f"[{col}]\n{_truncate(combined, max_chars=600)}")

        if not parts:
            return []

        aggregated = "\n\n".join(parts)
        prompt = (
            "You are analyzing evaluator artifacts from failed iterations in an evolutionary "
            "code optimization experiment.\n\n"
            f"Aggregated failure artifacts:\n{aggregated}\n\n"
            "Identify recurring error/failure patterns and respond ONLY with valid JSON in this format:\n"
            '{\n'
            '  "patterns": [\n'
            '    {\n'
            '      "description": "<pattern description>",\n'
            '      "count": <int>,\n'
            '      "pct_of_failures": <float 0-100>,\n'
            '      "root_cause": "<root cause>",\n'
            '      "recommendation": "<recommendation>"\n'
            '    }\n'
            '  ]\n'
            '}'
        )

        try:
            raw = self._call_llm_cached(prompt, step_name="artifact_clusters")
        except Exception as exc:
            logger.warning("Step B LLM call failed: %s", exc)
            return []

        parsed = _extract_json(raw)
        if parsed and isinstance(parsed.get("patterns"), list):
            return parsed["patterns"]
        logger.warning("Step B: could not parse JSON response; returning empty.")
        return []

    # ── Step C: Per-mutation quality ──────────────────────────────────────────

    def _eval_mutation_quality(self, df: pd.DataFrame) -> List[dict]:
        # Only non-stagnation records
        if "streak_id" in df.columns:
            records = df[df["streak_id"].isna()]
        else:
            records = df

        if records.empty:
            return []

        # Sample if too large
        if len(records) > _MAX_SAMPLE_SIZE:
            records = records.sample(n=_MAX_SAMPLE_SIZE, random_state=42)

        def _evaluate_one(row_tuple) -> dict:
            idx, row = row_tuple
            iteration = int(row.get("iteration", idx))
            score_delta = float(row.get("score_delta", 0.0))
            mutation_type = str(row.get("mutation_type", "unknown"))

            # Get diff or code
            diff_text = ""
            for col in ("diff", "code_diff", "code"):
                val = row.get(col)
                if val and str(val).strip():
                    diff_text = _truncate(str(val), max_chars=500)
                    break

            prompt = (
                "Rate the quality of this code mutation attempt on a scale of 1 to 5, "
                "where 1 = very poor and 5 = excellent. Explain your rating concisely.\n\n"
                f"Score delta: {score_delta:.4f}\n"
                f"Mutation type: {mutation_type}\n"
                f"Diff (truncated to 500 chars): {diff_text}\n\n"
                "Respond with your numeric rating (1-5) on the first line, then your explanation."
            )

            try:
                raw = self._call_llm_cached(prompt, step_name="mutation_quality")
            except Exception as exc:
                logger.warning("Step C LLM call failed for iter %d: %s", iteration, exc)
                return {"iteration": iteration, "quality_rating": 0, "explanation": ""}

            # Parse rating: find first digit 1-5
            rating = 0
            match = re.search(r"\b([1-5])\b", raw)
            if match:
                rating = int(match.group(1))

            explanation_lines = raw.splitlines()
            explanation = " ".join(
                line.strip() for line in explanation_lines[1:] if line.strip()
            )

            return {"iteration": iteration, "quality_rating": rating, "explanation": explanation}

        row_inputs = [(idx, row) for idx, row in records.iterrows()]
        parallel_inputs = [(item,) for item in row_inputs]

        results: List[ParallelResult] = run_parallel(
            _evaluate_one,
            parallel_inputs,
            use_async=False,
            progress_desc="Step C: mutation quality",
        )

        output: List[dict] = []
        for res in results:
            if res is not None and res.is_success and res.result is not None:
                output.append(res.result)
        return output

    # ── Step D: Semantic compliance ───────────────────────────────────────────

    def _eval_semantic_compliance(self, df: pd.DataFrame) -> List[dict]:
        # Only run if system_message field is present
        sys_col = None
        for col in ("system_message", "system_prompt"):
            if col in df.columns and df[col].notna().any():
                sys_col = col
                break

        if sys_col is None:
            return []

        records = df[df[sys_col].notna()]
        if records.empty:
            return []

        if len(records) > _MAX_SAMPLE_SIZE:
            records = records.sample(n=_MAX_SAMPLE_SIZE, random_state=42)

        def _check_one(row_tuple) -> dict:
            idx, row = row_tuple
            iteration = int(row.get("iteration", idx))
            system_message = _truncate(str(row.get(sys_col, "")), max_chars=500)

            diff_or_code = ""
            for col in ("diff", "code_diff", "code"):
                val = row.get(col)
                if val and str(val).strip():
                    diff_or_code = _truncate(str(val), max_chars=1000)
                    break

            prompt = (
                f"System instructions: {system_message}\n\n"
                f"Code change: {diff_or_code}\n\n"
                "Does this change comply with the instructions?\n"
                'Answer with JSON only: {"compliance_level": "fully_compliant" | '
                '"partially_compliant" | "non_compliant", "violations": [...]}'
            )

            try:
                raw = self._call_llm_cached(prompt, step_name="semantic_compliance")
            except Exception as exc:
                logger.warning("Step D LLM call failed for iter %d: %s", iteration, exc)
                return {
                    "iteration": iteration,
                    "compliance_level": "unknown",
                    "violations": [],
                    "raw_response": "",
                }

            parsed = _extract_json(raw)
            if parsed:
                return {
                    "iteration": iteration,
                    "compliance_level": parsed.get("compliance_level", "unknown"),
                    "violations": parsed.get("violations", []),
                    "raw_response": raw,
                }
            return {
                "iteration": iteration,
                "compliance_level": "unknown",
                "violations": [],
                "raw_response": raw,
            }

        row_inputs = [(idx, row) for idx, row in records.iterrows()]
        parallel_inputs = [(item,) for item in row_inputs]

        results: List[ParallelResult] = run_parallel(
            _check_one,
            parallel_inputs,
            use_async=False,
            progress_desc="Step D: semantic compliance",
        )

        output: List[dict] = []
        for res in results:
            if res is not None and res.is_success and res.result is not None:
                output.append(res.result)
        return output

    # ── Step E: Exploration structure ─────────────────────────────────────────

    def _eval_exploration_structure(self, df: pd.DataFrame) -> dict:
        import difflib

        diff_col = None
        for col in ("diff", "code_diff"):
            if col in df.columns and df[col].notna().any():
                diff_col = col
                break

        # If no precomputed diff column, compute unified diffs from parent/child code.
        if diff_col is None and "parent_code" in df.columns and "child_code" in df.columns:
            mask = df["parent_code"].notna() & df["child_code"].notna() & (df["parent_code"] != df["child_code"])
            changed_df: pd.DataFrame = df.loc[mask].copy()
            if len(changed_df) == 0:
                return {"skipped_reason": SKIP_REASON_FULL_REWRITE}
            computed_diffs = []
            for _, row in changed_df.iterrows():
                computed_diffs.append(
                    "\n".join(
                        difflib.unified_diff(
                            str(row["parent_code"]).splitlines(),
                            str(row["child_code"]).splitlines(),
                            lineterm="",
                        )
                    )
                )
            changed_df = changed_df.assign(_computed_diff=computed_diffs)
            df = changed_df
            diff_col = "_computed_diff"

        if diff_col is None:
            # Fall back: use full code column if present (last resort)
            if "code" in df.columns and df["code"].notna().any():
                diff_col = "code"
            elif "mutation_type" in df.columns:
                types = df["mutation_type"].dropna().str.lower()
                if (types.str.contains("rewrite") | types.str.contains("full")).any():
                    return {"skipped_reason": SKIP_REASON_FULL_REWRITE}
            else:
                return {}

        sample_df: pd.DataFrame = df.loc[df[diff_col].notna()]
        if len(sample_df) > _MAX_SAMPLE_SIZE:
            sample_df = sample_df.sample(n=_MAX_SAMPLE_SIZE, random_state=42)

        _MAX_DIFFS_CHARS = 12_000
        per_diff_budget = max(500, _MAX_DIFFS_CHARS // max(len(sample_df), 1))
        diffs_text = "\n---\n".join(
            _truncate(str(v), max_chars=per_diff_budget) for v in sample_df[diff_col].tolist()
        )
        diffs_text = _truncate(diffs_text, max_chars=_MAX_DIFFS_CHARS)

        prompt = (
            "Analyze the exploration patterns in these code mutations from an evolutionary "
            "optimization experiment.\n\n"
            "Are they structurally diverse or repetitive? "
            "What mutation strategies are being used?\n\n"
            f"Sample diffs:\n{diffs_text}\n\n"
            "Respond with JSON only:\n"
            '{"diversity_assessment": "<str>", "dominant_strategy": "<str>", "recommendation": "<str>"}'
        )

        logger.info("Step E: exploration structure (1 call)")
        try:
            raw = self._call_llm_cached(prompt, step_name="exploration_structure")
        except Exception as exc:
            logger.warning("Step E LLM call failed: %s", exc)
            return {}

        parsed = _extract_json(raw)
        if parsed:
            result = {
                "diversity_assessment": parsed.get("diversity_assessment", ""),
                "dominant_strategy": parsed.get("dominant_strategy", ""),
                "recommendation": parsed.get("recommendation", ""),
            }
        else:
            result = {"diversity_assessment": raw, "dominant_strategy": "", "recommendation": ""}
        logger.info("Step E: exploration structure — done")
        return result

    # ── Step F: Meta quality (reasoning trace coherence) ─────────────────────

    def _eval_meta_quality(self, df: pd.DataFrame) -> dict:
        trace_col = None
        for col in ("reasoning_trace", "reasoning", "scratchpad"):
            if col in df.columns and df[col].notna().any():
                trace_col = col
                break

        if trace_col is None:
            return {"skipped_reason": SKIP_REASON_NO_TRACE}

        sample_df = df[df[trace_col].notna()]
        if len(sample_df) > _MAX_SAMPLE_SIZE:
            sample_df = sample_df.sample(n=_MAX_SAMPLE_SIZE, random_state=42)

        traces_text = "\n---\n".join(
            _truncate(str(v), max_chars=300) for v in sample_df[trace_col].tolist()
        )
        traces_text = _truncate(traces_text, max_chars=_MAX_PROMPT_CHARS)

        prompt = (
            "Analyze the quality and coherence of these LLM reasoning traces from an "
            "evolutionary optimization experiment.\n\n"
            "Are the traces coherent? Do they show self-contradiction? "
            "Are they logically sound?\n\n"
            f"Sample traces:\n{traces_text}\n\n"
            "Respond with JSON only:\n"
            '{"coherence_rating": <int 1-5>, "issues": ["<issue1>", ...], "recommendation": "<str>"}'
        )

        try:
            raw = self._call_llm_cached(prompt, step_name="meta_quality")
        except Exception as exc:
            logger.warning("Step F LLM call failed: %s", exc)
            return {}

        parsed = _extract_json(raw)
        if parsed:
            # Coerce coherence_rating to int
            rating = parsed.get("coherence_rating", 0)
            try:
                rating = int(rating)
            except (ValueError, TypeError):
                rating = 0
            return {
                "coherence_rating": rating,
                "issues": parsed.get("issues", []),
                "recommendation": parsed.get("recommendation", ""),
            }
        # Fallback
        match = re.search(r"\b([1-5])\b", raw)
        rating = int(match.group(1)) if match else 0
        return {"coherence_rating": rating, "issues": [], "recommendation": raw}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _call_llm_cached(self, prompt: str, step_name: str) -> str:
        """Call LLM with caching. Uses the per-step model override if configured."""
        client = self._get_step_client(step_name)
        return cache_call(client.invoke, self.cache_dir, prompt, model=getattr(client, "model", ""))

    def _get_step_client(self, step_name: str) -> LLMClient:
        """Return LLM client for the given step, using model overrides from config."""
        if step_name in self._step_clients:
            return self._step_clients[step_name]

        overrides: dict = self.config.get("llm", {}).get("overrides", {})
        step_override = overrides.get(step_name, {})
        override_model = step_override.get("model")

        if not override_model:
            return self.llm_client

        # Check if it's a different model from the main client
        main_model = getattr(self.llm_client, "model", None)
        if override_model == main_model:
            return self.llm_client

        # Build a new client using the same provider/base_url as the main client
        provider = getattr(self.llm_client, "provider", "openai")
        base_url = getattr(
            getattr(self.llm_client, "_client", None), "base_url", None
        )
        api_key_env: str = self.config.get("llm", {}).get(
            "api_key_env", "EVOLVE_ANALYZER_API_KEY"
        )
        parameters: dict = dict(getattr(self.llm_client, "params", {}))
        # Remove temperature from parameters — get_llm_client handles eval_mode
        parameters.pop("temperature", None)

        try:
            client = get_llm_client(
                provider=provider,
                model=override_model,
                base_url=str(base_url) if base_url else None,
                api_key_env=api_key_env,
                eval_mode=True,
                parameters=parameters,
            )
        except Exception as exc:
            logger.warning(
                "Could not create override client for step %s (model=%s): %s. "
                "Falling back to main client.",
                step_name,
                override_model,
                exc,
            )
            return self.llm_client

        self._step_clients[step_name] = client
        return client
