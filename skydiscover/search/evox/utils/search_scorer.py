"""
Search algorithm scoring metrics for co-evolution.
"""

import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LogWindowScorer:
    """
    Log-weighted scorer for search algorithms.
    """

    def __init__(self, algorithm_id: Optional[str] = None):
        self.algorithm_id = algorithm_id or "unknown"
        self._start_score: Optional[float] = None
        self._start_iteration: Optional[int] = None
        self._best_scores: List[float] = []

    def reset_window(
        self,
        start_score: Optional[float],
        algorithm_id: Optional[str] = None,
        start_iteration: Optional[int] = None,
    ) -> None:
        self._start_score = float(start_score) if start_score is not None else 0.0
        self._start_iteration = start_iteration
        self._best_scores = []
        if algorithm_id:
            self.algorithm_id = algorithm_id

    def record_step(self, best_score: Optional[float]) -> None:
        if self._start_score is None:
            self.reset_window(best_score)
        if best_score is None:
            best_score = self._best_scores[-1] if self._best_scores else self._start_score
        self._best_scores.append(float(best_score))

    def get_window_size(self) -> int:
        return len(self._best_scores)

    def get_start_score(self) -> Optional[float]:
        return self._start_score

    def compute_metrics(
        self,
        start_score: Optional[float] = None,
        best_scores: Optional[List[float]] = None,
        horizon: Optional[int] = None,
        start_iteration: Optional[int] = None,
        total_iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        if start_iteration is None:
            start_iteration = self._start_iteration
        start = float(start_score if start_score is not None else (self._start_score or 0.0))
        scores_to_use = best_scores if best_scores is not None else self._best_scores
        T_obs = len(scores_to_use) if scores_to_use else 0
        horizon_int = int(horizon) if horizon else max(1, T_obs)

        running_best = start
        for s in scores_to_use:
            running_best = max(running_best, float(s))

        improvement = running_best - start
        log_weight = 1.0 + math.log(1.0 + max(0.0, start))
        combined_score = improvement * log_weight / math.sqrt(horizon_int)

        logger.debug(
            f"Search strategy score: combined={combined_score:.6f}, "
            f"improvement={improvement:.6f}, start={start:.6f}, "
            f"end={running_best:.6f}, horizon={horizon_int}"
        )

        return {
            "combined_score": combined_score,
            "window_start_iteration": start_iteration,
            "search_window_start_score": start,
            "search_window_end_score": running_best,
            "search_horizon": horizon_int,
        }
