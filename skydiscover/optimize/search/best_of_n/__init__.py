"""
Best of N module for selecting programs from the solution database.

This module provides a "best of N" database that reuses the same parent
for N consecutive iterations before sampling a new parent.
"""

from skydiscover.optimize.search.best_of_n.database import BestOfNDatabase

__all__ = ["BestOfNDatabase"]
