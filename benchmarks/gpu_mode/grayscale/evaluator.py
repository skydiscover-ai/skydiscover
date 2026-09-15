"""Evaluator for Grayscale — delegates to shared evaluator."""

import os
import sys

_problem_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_problem_dir)

if _problem_dir not in sys.path:
    sys.path.insert(0, _problem_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from shared_eval import evaluate, evaluate_stage1, evaluate_stage2
