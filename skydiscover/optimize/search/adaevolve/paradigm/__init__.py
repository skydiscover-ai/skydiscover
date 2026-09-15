"""
Paradigm Breakthrough System for AdaEvolve.

This module provides breakthrough idea generation for escaping stagnation
in evolutionary search. It detects improvement-rate based stagnation
(separate from iteration-based multi-child stagnation) and generates
LLM-guided breakthrough paradigms.

Components:
- ParadigmTracker: State management for improvement history and paradigms
- ParadigmGenerator: LLM interaction for generating breakthrough ideas
"""

from skydiscover.optimize.search.adaevolve.paradigm.generator import ParadigmGenerator
from skydiscover.optimize.search.adaevolve.paradigm.tracker import ParadigmTracker

__all__ = ["ParadigmTracker", "ParadigmGenerator"]
