"""
Quality-Diversity Archive for evolutionary search.

Provides:
- UnifiedArchive: Flat-list archive with unified elite scoring
- DiversityStrategy: Pluggable abstraction for measuring program difference
- Diversity implementations: CodeDiversity, MetricDiversity, HybridDiversity

The archive balances quality and diversity through:
- Pareto optimality (non-dominated programs protected)
- Fitness ranking (top performers protected)
- Novelty via k-NN (diverse programs valued)
- Deterministic crowding (similar programs compete)
"""

from skydiscover.optimize.search.adaevolve.archive.diversity import (
    CodeDiversity,
    DiversityStrategy,
    HybridDiversity,
    MetricDiversity,
    create_diversity_strategy,
)
from skydiscover.optimize.search.adaevolve.archive.unified_archive import (
    ArchiveConfig,
    UnifiedArchive,
)

__all__ = [
    # Archive
    "UnifiedArchive",
    "ArchiveConfig",
    # Diversity strategies
    "DiversityStrategy",
    "CodeDiversity",
    "MetricDiversity",
    "HybridDiversity",
    "create_diversity_strategy",
]
