import logging
from typing import Any, Dict, List, Optional, Tuple
from app.schemas.analysis import LiveScoringMetric

logger = logging.getLogger(__name__)


class ExerciseScoringEngine:
    """
    Weighted biomechanical scoring engine with dynamic weight normalization,
    confidence-aware smoothing, and reliability calculation.
    """

    def __init__(self, smoothing_alpha: float = 0.35, min_confidence_threshold: float = 0.25):
        self.smoothing_alpha = smoothing_alpha
        self.min_confidence_threshold = min_confidence_threshold
        self.prev_score: Optional[float] = None

    def reset(self):
        self.prev_score = None

    def evaluate_score(
        self,
        candidate_metrics: Optional[List[LiveScoringMetric]] = None,
        min_valid_weight_ratio: float = 0.40,
        metrics: Optional[List[LiveScoringMetric]] = None,
        **kwargs: Any
    ) -> Tuple[Optional[float], float, str, List[LiveScoringMetric]]:
        """
        Calculates normalized technique score and reliability from individual metrics.
        Returns:
            (overall_score: Optional[float], score_confidence: float, status: str, evaluated_metrics)
        """
        metric_list = candidate_metrics or metrics or []
        if not metric_list:
            return None, 0.0, "INSUFFICIENT_DATA", []

        valid_metrics: List[LiveScoringMetric] = []
        total_weight = sum(m.weight for m in metric_list)
        valid_weight = 0.0
        weighted_score_sum = 0.0
        weighted_conf_sum = 0.0

        for m in metric_list:
            # Check confidence validity
            if m.confidence < self.min_confidence_threshold:
                m.status = "LOW_CONFIDENCE"
                continue

            m.status = "VALID"
            valid_metrics.append(m)
            valid_weight += m.weight
            weighted_score_sum += m.score * m.weight
            weighted_conf_sum += m.confidence * m.weight

        # Check if enough valid weight remains
        weight_coverage = valid_weight / total_weight if total_weight > 0 else 0.0

        if weight_coverage < min_valid_weight_ratio or valid_weight <= 0:
            return None, round(weighted_conf_sum / max(1e-4, total_weight), 2), "INSUFFICIENT_DATA", candidate_metrics

        # Normalized weighted average
        raw_score = weighted_score_sum / valid_weight
        raw_score = max(0.0, min(100.0, raw_score))

        # Overall score reliability
        score_conf = weighted_conf_sum / valid_weight
        score_conf = max(0.0, min(1.0, score_conf))

        # Confidence-aware score smoothing
        if self.prev_score is not None:
            # Smooth more when confidence is higher to avoid erratic fluctuations
            alpha = self.smoothing_alpha * score_conf
            smooth_score = alpha * raw_score + (1.0 - alpha) * self.prev_score
        else:
            smooth_score = raw_score

        self.prev_score = smooth_score
        status = "VALID" if score_conf >= 0.50 else "LOW_CONFIDENCE"

        return round(smooth_score, 1), round(score_conf, 2), status, candidate_metrics
