"""LLM-as-a-judge: scores programs via LLM feedback.

Override _parse_response() to support formats other than JSON.
"""

import json
import logging
import re
from typing import Any, Dict, Optional

from skydiscover.optimize.context_builder.base import ContextBuilder
from skydiscover.optimize.evaluation.evaluation_result import EvaluationResult
from skydiscover.optimize.llm.llm_pool import LLMPool
from skydiscover.optimize.search.base_database import ProgramDatabase

logger = logging.getLogger(__name__)


class LLMJudge:
    """
    Scores programs via LLM feedback.

    Override _parse_response() to change how LLM output is interpreted.
    """

    def __init__(
        self,
        llm_pool: LLMPool,
        context_builder: ContextBuilder,
        database: Optional[ProgramDatabase] = None,
    ):
        self.llm_pool = llm_pool
        self.context_builder = context_builder
        self.database = database

    async def evaluate(
        self, program_solution: str, program_id: str = ""
    ) -> Optional[EvaluationResult]:
        """Score a program via LLM. Returns None on failure."""
        try:
            tm = self.context_builder.template_manager
            eval_sys = self.context_builder.config.evaluator_system_message
            system_msg = tm.get_template(eval_sys) if eval_sys in tm.templates else eval_sys
            user_msg = tm.get_template("evaluator_user_message").format(
                current_program=program_solution
            )

            llm_responses = await self.llm_pool.generate_all(
                system_msg, [{"role": "user", "content": user_msg}]
            )
            response_texts = [r.text for r in llm_responses]

            if self.database and program_id:
                self.database.log_prompt(
                    program_id=program_id,
                    template_key="evaluator_user_message",
                    prompt={"system": system_msg, "user": user_msg},
                    responses=response_texts,
                )

            metrics: Dict[str, float] = {}
            artifacts: Dict[str, Any] = {}
            for i, response in enumerate(response_texts):
                parsed = self._parse_response(response)
                weight = self.llm_pool.weights[i] if self.llm_pool.weights else 1.0
                for key, value in parsed.items():
                    if isinstance(value, (int, float)):
                        metrics[key] = metrics.get(key, 0.0) + float(value) * weight
                    else:
                        artifacts[key] = value

            return EvaluationResult(metrics=metrics, artifacts=artifacts)
        except Exception as e:
            logger.warning(f"LLM judge failed: {e}")
            return None

    def _parse_response(self, response: str) -> dict:
        """
        Extract a JSON dict from an LLM response.

        Tries a fenced json block first, then the outermost { ... }.
        Numeric values become metrics; everything else becomes artifacts.
        Override for XML, YAML, or structured output formats.
        """
        match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
        if match:
            return json.loads(match.group(1))

        start, end = response.find("{"), response.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])

        return json.loads(response)
