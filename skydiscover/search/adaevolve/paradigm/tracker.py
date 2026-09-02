"""
ParadigmTracker - State management for paradigm breakthrough system.

Tracks improvement history and manages active paradigms. Pure state
container with simple methods - no LLM calls or I/O.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ParadigmTracker:
    """
    Tracks improvement history and manages paradigm state.

    Two separate stagnation concepts:
    1. Iteration-based stagnation: iterations_since_improvement > threshold
       -> Handled by existing multi-child generation (per-island)
    2. Improvement-rate stagnation: improvement_rate < threshold over window
       -> Triggers paradigm generation (global, this tracker)

    When both trigger, paradigm guidance is added to the prompt context
    while multi-child can still generate multiple children independently.
    """

    # Configuration (tunable hyperparameters)
    window_size: int = 30
    improvement_threshold: float = 0.05
    max_paradigm_uses: int = 5
    max_tried_paradigms: int = 10
    num_paradigms_to_generate: int = 3

    # Improvement tracking - bounded list of binary values
    improvement_history: List[float] = field(default_factory=list)

    # Active paradigms and usage tracking
    active_paradigms: List[Dict[str, Any]] = field(default_factory=list)
    paradigm_usage_counts: Dict[int, int] = field(default_factory=dict)
    current_paradigm_index: int = 0

    # Previously tried paradigms with outcomes - bounded list
    tried_paradigms: List[Dict[str, Any]] = field(default_factory=list)

    # Score tracking for outcome evaluation
    best_score_at_paradigm_gen: float = 0.0
    best_score_during_paradigm: float = 0.0

    # =========================================================================
    # Improvement Recording
    # =========================================================================

    def record_improvement(self, improved: bool, current_best_score: float = 0.0) -> None:
        """
        Record binary improvement (1.0 if global best changed, else 0.0).

        Called after each program is added to the database, after
        _update_best_program() determines if there was improvement.

        Args:
            improved: Whether the global best changed
            current_best_score: Current best score for outcome tracking
        """
        value = 1.0 if improved else 0.0
        self.improvement_history.append(value)

        # Keep bounded to window_size
        while len(self.improvement_history) > self.window_size:
            self.improvement_history.pop(0)

        # Track best score during paradigm usage for outcome evaluation
        if self.active_paradigms and current_best_score > self.best_score_during_paradigm:
            self.best_score_during_paradigm = current_best_score

    def get_improvement_rate(self) -> float:
        """
        Calculate improvement rate over the current window.

        Returns:
            Float in [0.0, 1.0] - fraction of recent iterations that improved.
        """
        if not self.improvement_history:
            return 0.0
        return sum(self.improvement_history) / len(self.improvement_history)

    # =========================================================================
    # Stagnation Detection
    # =========================================================================

    def is_paradigm_stagnating(self) -> bool:
        """
        Check if improvement rate is below threshold.

        Paradigm stagnation requires:
        1. Enough history (at least window_size iterations)
        2. Improvement rate below threshold
        3. No active paradigms currently available

        Returns:
            True if paradigm generation should be triggered.
        """
        # Need enough data to make a judgment
        if len(self.improvement_history) < self.window_size:
            return False

        # If we have active paradigms still available, use them first
        if self.has_active_paradigm():
            return False

        # Check improvement rate against threshold
        return self.get_improvement_rate() < self.improvement_threshold

    # =========================================================================
    # Paradigm Access
    # =========================================================================

    def has_active_paradigm(self) -> bool:
        """Check if there's an active paradigm available for use."""
        if not self.active_paradigms:
            return False

        # Check if current paradigm is exhausted
        current_uses = self.paradigm_usage_counts.get(self.current_paradigm_index, 0)
        if current_uses >= self.max_paradigm_uses:
            # Try to rotate to next available paradigm
            return self._try_rotate_paradigm()

        return True

    def get_current_paradigm(self) -> Optional[Dict[str, Any]]:
        """
        Get the current active paradigm if available.

        Returns:
            Paradigm dict with keys: idea, description, what_to_optimize,
            cautions, approach_type. Returns None if no active paradigm.
        """
        if not self.has_active_paradigm():
            return None

        return self.active_paradigms[self.current_paradigm_index]

    def use_paradigm(self) -> None:
        """
        Record one use of the current paradigm.

        Called when a child is generated using the paradigm guidance.
        Increments usage counter for round-robin tracking.
        """
        if not self.active_paradigms:
            return

        current_uses = self.paradigm_usage_counts.get(self.current_paradigm_index, 0)
        self.paradigm_usage_counts[self.current_paradigm_index] = current_uses + 1

        # Log paradigm usage with idea
        paradigm = self.active_paradigms[self.current_paradigm_index]
        logger.debug(
            f"Using paradigm {self.current_paradigm_index + 1}/{len(self.active_paradigms)} "
            f"({current_uses + 1}/{self.max_paradigm_uses}): {paradigm.get('idea', 'N/A')}"
        )

        # Rotate for next use
        self._try_rotate_paradigm()

    # =========================================================================
    # Paradigm Management
    # =========================================================================

    def set_paradigms(self, paradigms: List[Dict[str, Any]], current_best_score: float) -> None:
        """
        Set new paradigms from generator.

        Archives current paradigms with outcome data, then sets new batch.

        Args:
            paradigms: List of paradigm dicts from generator
            current_best_score: Best score at time of generation
        """
        # Archive current paradigms with their outcomes
        self._archive_current_paradigms()

        # Set new paradigms
        self.active_paradigms = paradigms
        self.paradigm_usage_counts = {}
        self.current_paradigm_index = 0
        self.best_score_at_paradigm_gen = current_best_score
        self.best_score_during_paradigm = current_best_score

        logger.debug(f"Set {len(paradigms)} new paradigms (best score: {current_best_score:.6f})")

    def clear_paradigms(self) -> None:
        """
        Clear all active paradigms.

        Called when paradigms are exhausted or if manual reset is needed.
        Archives current paradigms before clearing.
        """
        self._archive_current_paradigms()
        self.active_paradigms = []
        self.paradigm_usage_counts = {}
        self.current_paradigm_index = 0
        logger.debug("Cleared active paradigms")

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _try_rotate_paradigm(self) -> bool:
        """
        Try to rotate to the next available paradigm.

        Returns:
            True if rotation successful, False if all paradigms exhausted.
        """
        if not self.active_paradigms:
            return False

        # Look for a paradigm that isn't exhausted
        for i in range(len(self.active_paradigms)):
            next_idx = (self.current_paradigm_index + 1 + i) % len(self.active_paradigms)
            if self.paradigm_usage_counts.get(next_idx, 0) < self.max_paradigm_uses:
                self.current_paradigm_index = next_idx
                logger.debug(f"Rotated to paradigm {next_idx}")
                return True

        # All paradigms exhausted
        logger.debug("All paradigms exhausted, will archive on next check")
        return False

    def _archive_current_paradigms(self) -> None:
        """
        Archive current paradigms to tried list with outcome info.

        Stores each paradigm with its usage count and score improvement
        for potential feedback to the generator.
        """
        if not self.active_paradigms:
            return

        # Calculate improvement achieved during this paradigm batch
        score_improvement = self.best_score_during_paradigm - self.best_score_at_paradigm_gen

        for idx, paradigm in enumerate(self.active_paradigms):
            uses = self.paradigm_usage_counts.get(idx, 0)
            if uses == 0:
                continue  # Don't archive unused paradigms

            archived = {
                **paradigm,
                "uses": uses,
                "starting_score": self.best_score_at_paradigm_gen,
                "ending_score": self.best_score_during_paradigm,
                "score_improvement": score_improvement,
                "outcome": "SUCCESS" if score_improvement > 0.001 else "FAILED",
            }
            self.tried_paradigms.append(archived)

        # Keep tried list bounded
        while len(self.tried_paradigms) > self.max_tried_paradigms:
            self.tried_paradigms.pop(0)

        # Log archived paradigms with outcomes
        if self.active_paradigms:
            logger.debug(
                f"Archived {len(self.active_paradigms)} paradigms (improvement: {score_improvement:+.6f}):"
            )
            for idx, paradigm in enumerate(self.active_paradigms):
                uses = self.paradigm_usage_counts.get(idx, 0)
                if uses > 0:
                    outcome = "SUCCESS" if score_improvement > 0.001 else "FAILED"
                    logger.debug(f"  [{outcome}] {paradigm.get('idea', 'N/A')} (uses: {uses})")

    # =========================================================================
    # Feedback for Generator
    # =========================================================================

    def get_previously_tried_ideas(self) -> List[str]:
        """
        Get formatted list of previously tried ideas for paradigm generator.

        Provides feedback about what worked and what didn't to help
        the generator avoid repeating failed approaches.

        Returns:
            List of formatted strings describing previous paradigms.
        """
        if not self.tried_paradigms:
            return []

        result = []
        for p in self.tried_paradigms:
            outcome = p.get("outcome", "UNCLEAR")
            approach = p.get("approach_type", "unknown")
            idea = p.get("idea", "unknown")
            improvement = p.get("score_improvement", 0.0)

            # Format: "OUTCOME: approach_type - idea (improvement: +/-X.XXXX)"
            result.append(f"{outcome}: {approach} - {idea} (improvement: {improvement:+.4f})")

        return result

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for checkpointing."""
        return {
            "window_size": self.window_size,
            "improvement_threshold": self.improvement_threshold,
            "max_paradigm_uses": self.max_paradigm_uses,
            "max_tried_paradigms": self.max_tried_paradigms,
            "num_paradigms_to_generate": self.num_paradigms_to_generate,
            "improvement_history": list(self.improvement_history),
            "active_paradigms": list(self.active_paradigms),
            "paradigm_usage_counts": dict(self.paradigm_usage_counts),
            "current_paradigm_index": self.current_paradigm_index,
            "tried_paradigms": list(self.tried_paradigms),
            "best_score_at_paradigm_gen": self.best_score_at_paradigm_gen,
            "best_score_during_paradigm": self.best_score_during_paradigm,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParadigmTracker":
        """Deserialize state from checkpoint."""
        tracker = cls(
            window_size=data.get("window_size", 30),
            improvement_threshold=data.get("improvement_threshold", 0.05),
            max_paradigm_uses=data.get("max_paradigm_uses", 5),
            max_tried_paradigms=data.get("max_tried_paradigms", 10),
            num_paradigms_to_generate=data.get("num_paradigms_to_generate", 3),
        )
        tracker.improvement_history = list(data.get("improvement_history", []))
        tracker.active_paradigms = list(data.get("active_paradigms", []))
        tracker.paradigm_usage_counts = {
            int(k): v for k, v in data.get("paradigm_usage_counts", {}).items()
        }
        tracker.current_paradigm_index = data.get("current_paradigm_index", 0)
        tracker.tried_paradigms = list(data.get("tried_paradigms", []))
        tracker.best_score_at_paradigm_gen = data.get("best_score_at_paradigm_gen", 0.0)
        tracker.best_score_during_paradigm = data.get("best_score_during_paradigm", 0.0)
        return tracker
