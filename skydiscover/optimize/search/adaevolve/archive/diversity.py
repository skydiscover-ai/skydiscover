"""
Diversity Strategies for measuring program difference.

This module provides a pluggable abstraction for computing how different
two programs are. All archive operations (k-NN, novelty, other context selection)
use this single abstraction.

Strategies:
- CodeDiversity: Based on code structure (fast, no dependencies)
- MetricDiversity: Based on evaluator metrics (normalized Euclidean)
- HybridDiversity: Weighted combination of multiple strategies
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from skydiscover.optimize.search.base_database import Program


class DiversityStrategy(ABC):
    """
    Abstract base for measuring how different two programs are.

    This is the SINGLE source of truth for distance/diversity/similarity.
    All archive operations use this abstraction.
    """

    @abstractmethod
    def distance(self, a: Program, b: Program) -> float:
        """
        Compute distance between two programs.

        Args:
            a: First program
            b: Second program

        Returns:
            float >= 0, where 0 = identical, higher = more different
        """
        pass

    def update(self, programs: List[Program]) -> None:
        """
        Update internal state based on current archive.

        Called after archive changes. Override for strategies that need
        normalization bounds or other population-dependent state.

        Args:
            programs: All programs currently in the archive
        """
        pass


class CodeDiversity(DiversityStrategy):
    """
    Diversity based on code structure and content.

    Fast computation with no external dependencies. Uses multiple signals:
    - Token-based Jaccard distance (captures vocabulary differences)
    - Structural features (imports, functions, classes)
    - Normalized length difference

    Good for: General use, when code structure reflects behavior.
    """

    def __init__(
        self,
        token_weight: float = 0.5,
        structure_weight: float = 0.3,
        length_weight: float = 0.2,
    ):
        """
        Args:
            token_weight: Weight for token Jaccard distance
            structure_weight: Weight for structural feature difference
            length_weight: Weight for normalized length difference
        """
        self.token_weight = token_weight
        self.structure_weight = structure_weight
        self.length_weight = length_weight

    def distance(self, a: Program, b: Program) -> float:
        solution1, solution2 = a.solution, b.solution

        if solution1 == solution2:
            return 0.0

        # 1. Token-based Jaccard distance (0 to 1)
        tokens1 = self._tokenize(solution1)
        tokens2 = self._tokenize(solution2)
        token_dist = self._jaccard_distance(tokens1, tokens2)

        # 2. Structural feature distance (0 to 1)
        struct_dist = self._structural_distance(solution1, solution2)

        # 3. Normalized length distance (0 to 1)
        max_len = max(len(solution1), len(solution2), 1)
        len_dist = abs(len(solution1) - len(solution2)) / max_len

        return (
            token_dist * self.token_weight
            + struct_dist * self.structure_weight
            + len_dist * self.length_weight
        )

    def _tokenize(self, code: str) -> set:
        """
        Extract meaningful tokens from code.

        Splits on whitespace and punctuation, filters short tokens.
        Captures identifiers, keywords, and significant patterns.
        """
        import re

        # Split on whitespace and common delimiters, keep meaningful tokens
        tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+\.?[0-9]*", code)
        # Filter very short tokens (likely noise) but keep keywords
        return set(t for t in tokens if len(t) >= 2)

    def _jaccard_distance(self, set1: set, set2: set) -> float:
        """Compute Jaccard distance: 1 - |intersection| / |union|"""
        if not set1 and not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        if union == 0:
            return 0.0
        return 1.0 - (intersection / union)

    def _structural_distance(self, solution1: str, solution2: str) -> float:
        """
        Compare structural features of two code snippets.

        Looks at imports, function definitions, class definitions.
        """
        # Extract structural features
        features1 = self._extract_features(solution1)
        features2 = self._extract_features(solution2)

        # Compare feature sets using Jaccard
        return self._jaccard_distance(features1, features2)

    def _extract_features(self, solution: str) -> set:
        """Extract structural features from code."""
        import re

        features = set()

        # Import statements (what libraries are used)
        imports = re.findall(r"(?:from\s+(\S+)\s+)?import\s+(\S+)", solution)
        for from_mod, imp in imports:
            if from_mod:
                features.add(f"import:{from_mod}")
            features.add(f"import:{imp.split('.')[0]}")

        # Function definitions
        functions = re.findall(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", solution)
        for func in functions:
            features.add(f"func:{func}")

        # Class definitions
        classes = re.findall(r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[:\(]", solution)
        for cls in classes:
            features.add(f"class:{cls}")

        # Key function calls (common libraries)
        calls = re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+)\s*\(", solution)
        for call in calls:
            features.add(f"call:{call}")

        # Control flow patterns
        if "for " in solution:
            features.add("pattern:for_loop")
        if "while " in solution:
            features.add("pattern:while_loop")
        if "try:" in solution or "try :" in solution:
            features.add("pattern:try_except")
        if "with " in solution:
            features.add("pattern:context_manager")
        if "yield " in solution:
            features.add("pattern:generator")
        if "async " in solution or "await " in solution:
            features.add("pattern:async")
        if "lambda " in solution:
            features.add("pattern:lambda")

        return features


class MetricDiversity(DiversityStrategy):
    """
    Diversity based on evaluator metrics.

    Computes normalized Euclidean distance in metric space.
    Each metric is normalized to [0, 1] based on observed min/max.

    Good for: When evaluator returns multiple meaningful metrics.
    """

    def __init__(self, higher_is_better: Optional[Dict[str, bool]] = None):
        """
        Args:
            higher_is_better: Dict mapping metric name to direction.
                              If None, assumes higher is better for all.
        """
        self.higher_is_better = higher_is_better or {}
        self._bounds: Dict[str, Tuple[float, float]] = {}

    def update(self, programs: List[Program]) -> None:
        """Update metric bounds from current archive."""
        self._bounds.clear()

        for p in programs:
            for key, val in p.metrics.items():
                if not isinstance(val, (int, float)):
                    continue
                val = float(val)
                if key not in self._bounds:
                    self._bounds[key] = (val, val)
                else:
                    lo, hi = self._bounds[key]
                    self._bounds[key] = (min(lo, val), max(hi, val))

    def _safe_get_numeric(self, metrics: Dict, key: str, default: float) -> Optional[float]:
        """Safely get a numeric value from metrics, returning None if not numeric."""
        val = metrics.get(key)
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return float(val)
        # Try to convert string to float (might be a stringified number)
        if isinstance(val, str):
            try:
                return float(val)
            except (ValueError, TypeError):
                return None
        return None

    def distance(self, a: Program, b: Program) -> float:
        # If bounds are empty, compute distance directly from programs' metrics
        if not self._bounds:
            return self._compute_fallback_distance(a, b)

        dist_sq = 0.0
        count = 0

        for key, (lo, hi) in self._bounds.items():
            # Get values safely (skip if either is non-numeric)
            mid = (lo + hi) / 2
            val_a = self._safe_get_numeric(a.metrics, key, mid)
            val_b = self._safe_get_numeric(b.metrics, key, mid)

            # Skip this metric if either value is non-numeric
            if val_a is None or val_b is None:
                continue

            # Normalize to [0, 1]
            if hi > lo:
                norm_a = (val_a - lo) / (hi - lo)
                norm_b = (val_b - lo) / (hi - lo)
            else:
                norm_a = norm_b = 0.5

            dist_sq += (norm_a - norm_b) ** 2
            count += 1

        if count == 0:
            return self._compute_fallback_distance(a, b)

        # Return normalized Euclidean distance
        return (dist_sq / count) ** 0.5

    def _compute_fallback_distance(self, a: Program, b: Program) -> float:
        """
        Compute distance when bounds are unavailable.

        Falls back to unnormalized Euclidean distance on shared numeric metrics.
        """
        # Find shared numeric metrics
        shared_keys = set()
        for key, val in a.metrics.items():
            if isinstance(val, (int, float)) and key in b.metrics:
                if isinstance(b.metrics[key], (int, float)):
                    shared_keys.add(key)

        if not shared_keys:
            # No shared numeric metrics - use code length difference as proxy
            len_diff = abs(len(a.solution) - len(b.solution))
            return min(len_diff / 1000.0, 1.0)  # Normalize roughly

        # Compute unnormalized distance on shared metrics
        dist_sq = 0.0
        for key in shared_keys:
            val_a = float(a.metrics[key])
            val_b = float(b.metrics[key])
            # Use relative difference to handle different scales
            max_val = max(abs(val_a), abs(val_b), 1e-10)
            diff = (val_a - val_b) / max_val
            dist_sq += diff**2

        return (dist_sq / len(shared_keys)) ** 0.5


class HybridDiversity(DiversityStrategy):
    """
    Combines multiple diversity strategies with weights.

    Useful for balancing code-based and metric-based diversity.

    Example:
        hybrid = HybridDiversity([
            (CodeDiversity(), 0.5),
            (MetricDiversity(), 0.5),
        ])
    """

    def __init__(self, strategies: List[Tuple[DiversityStrategy, float]]):
        """
        Args:
            strategies: List of (strategy, weight) pairs.
                        Weights are normalized to sum to 1.
        """
        if not strategies:
            raise ValueError("At least one strategy required")

        self.strategies = [s for s, _ in strategies]

        # Normalize weights
        total_weight = sum(w for _, w in strategies)
        if total_weight <= 0:
            raise ValueError("Total weight must be positive")
        self.weights = [w / total_weight for _, w in strategies]

    def update(self, programs: List[Program]) -> None:
        """Update all sub-strategies."""
        for strategy in self.strategies:
            strategy.update(programs)

    def distance(self, a: Program, b: Program) -> float:
        """Weighted sum of sub-strategy distances."""
        total = 0.0
        for strategy, weight in zip(self.strategies, self.weights):
            total += strategy.distance(a, b) * weight
        return total


def create_diversity_strategy(strategy_type: str = "code", **kwargs) -> DiversityStrategy:
    """
    Factory function to create diversity strategies.

    Args:
        strategy_type: One of "code", "metric", "hybrid"
        **kwargs: Strategy-specific arguments

    Returns:
        Configured DiversityStrategy instance
    """
    if strategy_type == "code":
        return CodeDiversity(
            token_weight=kwargs.get("token_weight", 0.5),
            structure_weight=kwargs.get("structure_weight", 0.3),
            length_weight=kwargs.get("length_weight", 0.2),
        )

    elif strategy_type == "text":
        # For natural language (prompts): token Jaccard + length, no code structure
        return CodeDiversity(
            token_weight=kwargs.get("token_weight", 0.7),
            structure_weight=0.0,
            length_weight=kwargs.get("length_weight", 0.3),
        )

    elif strategy_type == "metric":
        return MetricDiversity(
            higher_is_better=kwargs.get("higher_is_better"),
        )

    elif strategy_type == "hybrid":
        # Default: 50% code, 50% metric
        code_weight = kwargs.get("code_weight", 0.5)
        metric_weight = kwargs.get("metric_weight", 0.5)
        return HybridDiversity(
            [
                (CodeDiversity(), code_weight),
                (MetricDiversity(kwargs.get("higher_is_better")), metric_weight),
            ]
        )

    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
