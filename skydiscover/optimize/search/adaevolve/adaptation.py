"""
AdaEvolve Adaptation Engine

The core adaptive mechanisms for zeroth-order optimization.
Adapts search intensity per island based on accumulated improvement history.

Key Concepts:
- Improvement Signal (δ): Normalized magnitude of fitness improvement
- Accumulated Signal (G): Decayed sum of squared normalized improvements
- Search Intensity: How aggressively to explore vs exploit, adapted per-island

Formula:
    search_intensity = I_min + (I_max - I_min) / (1 + √(G + ε))

    - Low G → high search intensity (explore stagnating areas)
    - High G → low search intensity (exploit productive areas)
"""

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AdaptiveState:
    """
    Adaptive state for a single search dimension (e.g., an island).

    Tracks accumulated improvement signal (G) and computes search intensity
    based on historical productivity. Uses normalized delta to be scale-invariant.

    Attributes:
        accumulated_signal: G_t - decayed sum of squared normalized improvements
        best_score: Best fitness seen on this dimension
        improvement_count: Number of improvements found
        total_evaluations: Total programs evaluated
        decay: ρ - recency weight for exponential moving average
        epsilon: Numerical stability constant
        intensity_min: Minimum search intensity (more exploitation)
        intensity_max: Maximum search intensity (more exploration)
    """

    accumulated_signal: float = 0.0
    best_score: float = float("-inf")
    improvement_count: int = 0
    total_evaluations: int = 0

    # Hyperparameters
    decay: float = 0.9
    epsilon: float = 1e-8
    intensity_min: float = 0.1
    intensity_max: float = 0.7

    def _normalize_delta(self, raw_delta: float) -> float:
        """
        Normalize improvement delta to be scale-invariant.

        Uses abs(best_score) + epsilon to handle:
        - Infinite best_score (first evaluation, best_score = -inf)
        - Zero best_score (start of run)
        - Negative best_score (error minimization tasks)
        - Small positive best_score (prevents explosion)

        Args:
            raw_delta: The raw improvement (fitness - best_score)

        Returns:
            Normalized delta, capped at 1.0 to prevent extreme values
        """
        # Handle first evaluation where best_score is -inf
        # In this case, any finite improvement is significant but we cap it
        if math.isinf(self.best_score):
            return 1.0  # First improvement is always "significant"

        # Safe normalization: handles zero, negative, and small positive values
        # abs() handles negative fitness scales (e.g., error minimization)
        # epsilon prevents division by zero
        denominator = abs(self.best_score) + self.epsilon
        normalized = raw_delta / denominator

        # Cap at 1.0 to prevent extreme values from dominating G
        return min(normalized, 1.0)

    def record_evaluation(self, fitness: float) -> float:
        """
        Record a program evaluation and return normalized improvement delta.

        Normalizes delta by current best_score to make the algorithm
        scale-invariant. This prevents G from exploding with large fitness values.

        Args:
            fitness: The fitness of the evaluated program

        Returns:
            normalized_delta: Normalized improvement (0 if no improvement)
        """
        self.total_evaluations += 1

        if fitness > self.best_score:
            raw_delta = fitness - self.best_score
            normalized_delta = self._normalize_delta(raw_delta)

            self.best_score = fitness
            self.improvement_count += 1

            # Update accumulated signal with normalized squared delta
            # G_t = ρ * G_{t-1} + (1 - ρ) * δ²
            self.accumulated_signal = self.decay * self.accumulated_signal + (1 - self.decay) * (
                normalized_delta**2
            )

            return normalized_delta

        return 0.0

    def receive_external_improvement(self, fitness: float) -> float:
        """
        Handle an externally-received improvement (e.g., migration).

        Updates best_score and accumulated_signal WITHOUT updating counts.
        This ensures:
        1. Future delta calculations use correct baseline
        2. Search intensity drops to exploitation mode for the new solution
        3. UCB stats remain unaffected (island didn't earn the improvement)

        Args:
            fitness: The fitness of the received program

        Returns:
            normalized_delta: The improvement delta (0 if no improvement)
        """
        if fitness <= self.best_score:
            return 0.0

        raw_delta = fitness - self.best_score
        normalized_delta = self._normalize_delta(raw_delta)

        # Update best_score (CRITICAL: fixes future delta calculations)
        self.best_score = fitness

        # Update accumulated_signal (triggers exploitation mode)
        # The island now has a good solution worth refining
        self.accumulated_signal = self.decay * self.accumulated_signal + (1 - self.decay) * (
            normalized_delta**2
        )

        # NOTE: We do NOT update improvement_count or total_evaluations
        # because the island didn't earn this improvement

        return normalized_delta

    def get_search_intensity(self) -> float:
        """
        Compute search intensity based on accumulated signal.

        Uses inverse relationship with sqrt of accumulated signal:
            intensity = I_min + (I_max - I_min) / (1 + √(G + ε))

        - High G → intensity approaches I_min (exploit productive island)
        - Low G → intensity approaches I_max (explore stagnating island)

        Returns:
            intensity: Float in [intensity_min, intensity_max]
        """
        G = self.accumulated_signal

        intensity = self.intensity_min + (self.intensity_max - self.intensity_min) / (
            1 + math.sqrt(G + self.epsilon)
        )

        return intensity

    def get_productivity(self) -> float:
        """
        Get productivity metric for this dimension.

        Returns:
            Float representing improvement rate (improvements / evaluations)
        """
        if self.total_evaluations == 0:
            return 0.0
        return self.improvement_count / self.total_evaluations

    def reset(self) -> None:
        """Reset adaptive state (e.g., when spawning a new island)."""
        self.accumulated_signal = 0.0
        self.best_score = float("-inf")
        self.improvement_count = 0
        self.total_evaluations = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for checkpointing."""
        return {
            "accumulated_signal": self.accumulated_signal,
            "best_score": self.best_score,
            "improvement_count": self.improvement_count,
            "total_evaluations": self.total_evaluations,
            "decay": self.decay,
            "epsilon": self.epsilon,
            "intensity_min": self.intensity_min,
            "intensity_max": self.intensity_max,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdaptiveState":
        """Deserialize state from checkpoint."""
        state = cls(
            decay=data.get("decay", 0.9),
            epsilon=data.get("epsilon", 1e-8),
            intensity_min=data.get("intensity_min", 0.1),
            intensity_max=data.get("intensity_max", 0.7),
        )
        state.accumulated_signal = data.get("accumulated_signal", 0.0)
        state.best_score = data.get("best_score", float("-inf"))
        state.improvement_count = data.get("improvement_count", 0)
        state.total_evaluations = data.get("total_evaluations", 0)
        return state


@dataclass
class MultiDimensionalAdapter:
    """
    Manages adaptive state across multiple search dimensions (islands).

    Provides UCB-style selection with DECAYED magnitude rewards.
    This fixes the "breakthrough memory" problem where old breakthroughs
    would dominate island selection forever.

    KEY DESIGN: Two different normalizations for two different purposes:
    1. Search Intensity (per-island): Uses LOCAL best for scale-invariant adaptation
    2. UCB Rewards (cross-island): Uses GLOBAL best for fair comparison

    This fixes the "Poor Island Bias" where trash islands with high percentage
    gains would dominate UCB over productive islands with globally valuable
    improvements.

    Attributes:
        states: List of AdaptiveState for each dimension
        dimension_visits: Raw visit count per dimension (for exploration bonus)
        dimension_rewards: Decayed cumulative rewards per dimension (GLOBAL normalized)
        decayed_visits: Decayed visit count per dimension (for reward average)
        global_best_score: Best fitness seen across ALL dimensions (for UCB normalization)
        ucb_exploration: Exploration constant for UCB (√2 is classic)
        min_visits: Minimum visits before UCB kicks in
        decay: Decay factor for rewards (same as AdaptiveState)
        epsilon: Numerical stability constant

    Note on decayed_visits:
        Both rewards and visits must decay at the same rate for reward_avg
        to remain meaningful. Without decayed visits:
            reward_avg = (decaying_sum) / (growing_count) → 0 as visits grow
        With decayed visits:
            reward_avg = (decayed_rewards) / (decayed_visits) = recent reward per recent visit
    """

    states: List[AdaptiveState] = field(default_factory=list)
    dimension_visits: List[int] = field(default_factory=list)  # Raw counts for exploration
    dimension_rewards: List[float] = field(
        default_factory=list
    )  # Decayed rewards (GLOBAL normalized)
    decayed_visits: List[float] = field(default_factory=list)  # Decayed visits for reward_avg

    # Global tracking for UCB normalization
    global_best_score: float = float("-inf")  # Best across ALL dimensions

    # UCB parameters
    ucb_exploration: float = 1.41  # √2
    min_visits: int = 3
    decay: float = 0.9
    epsilon: float = 1e-8  # Numerical stability

    # Selection RNG, replaced by the owning database with its own so one seed governs
    # the whole engine. Excluded from compare/repr; to_dict serialises explicitly, so
    # it never reaches a checkpoint.
    rng: random.Random = field(default_factory=random.Random, compare=False, repr=False)

    def add_dimension(self, state: AdaptiveState = None) -> int:
        """
        Add a new dimension (e.g., spawn a new island).

        Args:
            state: Optional pre-configured AdaptiveState

        Returns:
            Index of the new dimension
        """
        if state is None:
            state = AdaptiveState(decay=self.decay)
        self.states.append(state)
        self.dimension_visits.append(0)  # Raw count
        self.dimension_rewards.append(0.0)  # Decayed rewards
        self.decayed_visits.append(0.0)  # Decayed visits
        return len(self.states) - 1

    def _normalize_by_global(self, raw_delta: float) -> float:
        """
        Normalize improvement delta by GLOBAL best score for UCB rewards.

        This ensures fair comparison across islands - a 10-point improvement
        is valued the same whether it comes from a high-fitness or low-fitness
        island.

        Args:
            raw_delta: The raw improvement (fitness - old_best)

        Returns:
            Normalized delta using global best, capped at 1.0
        """
        if raw_delta <= 0:
            return 0.0

        # Handle first evaluation where global_best is -inf
        if math.isinf(self.global_best_score):
            return 1.0  # First improvement is always "significant"

        # Safe normalization using global best
        denominator = abs(self.global_best_score) + self.epsilon
        normalized = raw_delta / denominator

        # Cap at 1.0 to prevent extreme values
        return min(normalized, 1.0)

    def record_evaluation(self, dim_idx: int, fitness: float) -> float:
        """
        Record an evaluation for a dimension.

        Updates both the dimension's AdaptiveState and the UCB rewards.

        KEY: Two different normalizations:
        1. AdaptiveState uses LOCAL best → search intensity adaptation
        2. UCB rewards use GLOBAL best → fair cross-island comparison

        This fixes "Poor Island Bias" where trash islands with high local
        percentage gains would dominate UCB over globally productive islands.

        Args:
            dim_idx: Index of the dimension
            fitness: Fitness of the evaluated program

        Returns:
            local_normalized_delta: The locally-normalized improvement (for search intensity)
        """
        if dim_idx < 0 or dim_idx >= len(self.states):
            raise ValueError(f"Invalid dimension index: {dim_idx}")

        # Get local best BEFORE update (needed for global UCB reward calculation)
        local_best_before = self.states[dim_idx].best_score

        # Update adaptive state with LOCAL normalization (for search intensity)
        # This returns locally-normalized delta
        local_normalized_delta = self.states[dim_idx].record_evaluation(fitness)

        # Update raw visit count (for exploration bonus)
        self.dimension_visits[dim_idx] += 1

        # Update DECAYED visits: V_t = ρ * V_{t-1} + 1
        self.decayed_visits[dim_idx] = self.decay * self.decayed_visits[dim_idx] + 1.0

        # Calculate GLOBAL-normalized delta for UCB rewards
        # This ensures fair comparison: a 10-point improvement is valued
        # equally whether from a high-fitness or low-fitness island
        if fitness > local_best_before:
            raw_delta = fitness - local_best_before
            global_normalized_delta = self._normalize_by_global(raw_delta)

            # Update global best if this is a new global record
            if fitness > self.global_best_score:
                self.global_best_score = fitness
        else:
            global_normalized_delta = 0.0

        # Update UCB rewards with GLOBAL-normalized delta and DECAY
        self.dimension_rewards[dim_idx] = (
            self.decay * self.dimension_rewards[dim_idx] + global_normalized_delta
        )

        return local_normalized_delta

    def receive_external_improvement(self, dim_idx: int, fitness: float) -> float:
        """
        Handle an externally-received improvement (e.g., migration) for a dimension.

        Updates the dimension's best_score and accumulated_signal for correct
        search intensity adaptation, but does NOT update UCB rewards or visits
        since the island didn't earn this improvement.

        Also updates global_best_score if this migration brings a new global best.

        Args:
            dim_idx: Index of the dimension
            fitness: Fitness of the received program

        Returns:
            normalized_delta: The improvement delta (0 if no improvement)
        """
        if dim_idx < 0 or dim_idx >= len(self.states):
            raise ValueError(f"Invalid dimension index: {dim_idx}")

        # Update global best if migration brings new global record
        # (Must be tracked for correct UCB normalization)
        if fitness > self.global_best_score:
            self.global_best_score = fitness

        # Delegate to AdaptiveState - updates best_score and G only
        # UCB stats (visits, rewards) remain unchanged
        return self.states[dim_idx].receive_external_improvement(fitness)

    def select_dimension_ucb(self, total_iterations: int) -> int:
        """
        Select next dimension using UCB with decayed magnitude rewards.

        UCB formula: reward_avg + C * sqrt(ln(N) / visits)

        With decayed magnitude rewards:
        - Islands that find recent breakthroughs are prioritized
        - Old breakthroughs decay away, preventing stalling

        Args:
            total_iterations: Total iterations across all dimensions

        Returns:
            dim_idx: Index of selected dimension
        """
        n_dims = len(self.states)

        if n_dims == 0:
            raise ValueError("No dimensions available")

        # Ensure minimum visits for all dimensions
        # Randomize order to avoid always returning dimension 0 when multiple
        # dimensions are underexplored (fixes biased exploration issue)
        underexplored = [i for i in range(n_dims) if self.dimension_visits[i] < self.min_visits]
        if underexplored:
            import random

            return self.rng.choice(underexplored)

        # UCB selection
        best_dim = 0
        best_ucb = float("-inf")

        for i in range(n_dims):
            raw_visits = self.dimension_visits[i]
            dec_visits = self.decayed_visits[i]

            # Recent reward average using DECAYED visits
            # This prevents reward_avg → 0 as raw visits grow
            # reward_avg = decayed_rewards / decayed_visits = recent reward per recent visit
            reward_avg = self.dimension_rewards[i] / dec_visits if dec_visits > 0 else 0.0

            # Exploration bonus uses RAW visits
            # We still want classic UCB exploration: visit underexplored islands
            # GUARD: Prevent division by zero if raw_visits is somehow 0
            # (shouldn't happen after min_visits check, but defensive programming)
            if raw_visits <= 0:
                exploration_bonus = float("inf")  # Force exploration of unvisited dimension
            else:
                exploration_bonus = self.ucb_exploration * math.sqrt(
                    math.log(total_iterations + 1) / raw_visits
                )

            ucb_score = reward_avg + exploration_bonus

            if ucb_score > best_ucb:
                best_ucb = ucb_score
                best_dim = i

        return best_dim

    def get_search_intensity(self, dim_idx: int) -> float:
        """
        Get search intensity for a specific dimension.

        Args:
            dim_idx: Index of the dimension

        Returns:
            intensity: Float in [intensity_min, intensity_max]
        """
        if dim_idx < 0 or dim_idx >= len(self.states):
            raise ValueError(f"Invalid dimension index: {dim_idx}")
        return self.states[dim_idx].get_search_intensity()

    def get_global_productivity(self) -> float:
        """
        Get aggregate productivity across all dimensions.

        Returns:
            Float representing overall improvement rate
        """
        total_improvements = sum(s.improvement_count for s in self.states)
        total_evaluations = sum(s.total_evaluations for s in self.states)

        if total_evaluations == 0:
            return 1.0  # Assume productive if no data

        return total_improvements / total_evaluations

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics for logging/debugging.

        Returns:
            Dictionary with per-dimension and aggregate stats
        """
        dim_stats = []
        for i, state in enumerate(self.states):
            dec_visits = self.decayed_visits[i] if i < len(self.decayed_visits) else 0.0
            dim_stats.append(
                {
                    "index": i,
                    "accumulated_signal": state.accumulated_signal,
                    "best_score": state.best_score,
                    "search_intensity": state.get_search_intensity(),
                    "improvements": state.improvement_count,
                    "evaluations": state.total_evaluations,
                    "raw_visits": self.dimension_visits[i],
                    "decayed_visits": dec_visits,
                    "decayed_reward": self.dimension_rewards[i],
                    "reward_avg": self.dimension_rewards[i] / dec_visits if dec_visits > 0 else 0.0,
                }
            )

        return {
            "num_dimensions": len(self.states),
            "global_best_score": self.global_best_score,
            "global_productivity": self.get_global_productivity(),
            "dimensions": dim_stats,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for checkpointing."""
        return {
            "states": [s.to_dict() for s in self.states],
            "dimension_visits": list(self.dimension_visits),
            "dimension_rewards": list(self.dimension_rewards),
            "decayed_visits": list(self.decayed_visits),
            "global_best_score": self.global_best_score,
            "ucb_exploration": self.ucb_exploration,
            "min_visits": self.min_visits,
            "decay": self.decay,
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiDimensionalAdapter":
        """Deserialize state from checkpoint."""
        adapter = cls(
            ucb_exploration=data.get("ucb_exploration", 1.41),
            min_visits=data.get("min_visits", 3),
            decay=data.get("decay", 0.9),
            epsilon=data.get("epsilon", 1e-8),
        )
        adapter.states = [AdaptiveState.from_dict(s) for s in data.get("states", [])]
        adapter.dimension_visits = list(data.get("dimension_visits", []))
        adapter.dimension_rewards = list(data.get("dimension_rewards", []))
        adapter.decayed_visits = list(data.get("decayed_visits", []))
        adapter.global_best_score = data.get("global_best_score", float("-inf"))

        # Backward compatibility: if decayed_visits not in checkpoint,
        # initialize from raw visits (loses decay history but functional)
        if not adapter.decayed_visits and adapter.dimension_visits:
            adapter.decayed_visits = [float(v) for v in adapter.dimension_visits]

        # Backward compatibility: if global_best_score not in checkpoint,
        # compute from per-dimension best scores
        if adapter.global_best_score == float("-inf") and adapter.states:
            adapter.global_best_score = (
                max(s.best_score for s in adapter.states if not math.isinf(s.best_score))
                if any(not math.isinf(s.best_score) for s in adapter.states)
                else float("-inf")
            )

        return adapter
