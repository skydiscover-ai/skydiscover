from dataclasses import dataclass, field
from typing import Any, Dict, Union


@dataclass
class EvaluationResult:
    """
    Result of program evaluation containing both metrics and optional artifacts
    """

    metrics: Dict[str, float]
    artifacts: Dict[str, Union[str, bytes]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, metrics: Dict[str, float]) -> "EvaluationResult":
        return cls(metrics=metrics)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = dict(self.metrics)
        if self.artifacts:
            result["artifacts"] = self.artifacts
        return result
