"""
Standard evaluation representation and mate handling.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union


@dataclass(frozen=True)
class EvaluationResult:
    """Standardized representation of a Stockfish evaluation output."""
    score_cp: Optional[float]
    is_mate: bool = False
    mate_in: Optional[int] = None
    depth: int = 18

    @property
    def is_valid_numeric(self) -> bool:
        """Returns True if score is a valid finite centipawn evaluation without forced mate."""
        return (not self.is_mate) and (self.score_cp is not None)


def normalize_evaluation(
    raw_eval: Union[Dict[str, Any], EvaluationResult],
    default_depth: int = 18
) -> EvaluationResult:
    """
    Converts raw dictionary from StockfishEngine into an immutable EvaluationResult.
    """
    if isinstance(raw_eval, EvaluationResult):
        return raw_eval

    score_cp = raw_eval.get("score_cp")
    if score_cp is not None:
        score_cp = float(score_cp)

    return EvaluationResult(
        score_cp=score_cp,
        is_mate=bool(raw_eval.get("is_mate", False)),
        mate_in=raw_eval.get("mate_in"),
        depth=int(raw_eval.get("depth", default_depth)),
    )
