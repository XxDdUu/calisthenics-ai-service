import math
from typing import Dict, Optional
from app.core.config import settings
from app.schemas.analysis import KeypointData


class TemporalKeypointSmoother:
    """
    Applies Exponential Moving Average (EMA) smoothing to pose keypoints
    over consecutive temporal video frames to suppress detector jitter.

    formula: smoothed = alpha * current + (1 - alpha) * previous
    """

    def __init__(self, alpha: Optional[float] = None, max_jump_distance_px: float = 120.0):
        self.alpha = alpha if alpha is not None else settings.SMOOTHING_ALPHA
        self.max_jump_distance_px = max_jump_distance_px
        self._prev_keypoints: Dict[str, KeypointData] = {}

    def smooth(self, keypoints: Dict[str, KeypointData]) -> Dict[str, KeypointData]:
        """
        Smooths incoming keypoints against previous smoothed frame.
        Returns a new dictionary with smoothed KeypointData objects.
        """
        if not keypoints:
            self._prev_keypoints.clear()
            return {}

        smoothed_result: Dict[str, KeypointData] = {}

        for name, curr in keypoints.items():
            prev = self._prev_keypoints.get(name)

            if prev is not None and prev.visible and curr.visible:
                # Check for large teleportation jumps (tracking loss / person switch)
                dist = math.hypot(curr.x - prev.x, curr.y - prev.y)
                if dist <= self.max_jump_distance_px:
                    smooth_x = self.alpha * curr.x + (1.0 - self.alpha) * prev.x
                    smooth_y = self.alpha * curr.y + (1.0 - self.alpha) * prev.y
                    smooth_conf = self.alpha * curr.confidence + (1.0 - self.alpha) * prev.confidence
                    smoothed_kp = KeypointData(
                        x=round(smooth_x, 2),
                        y=round(smooth_y, 2),
                        confidence=round(smooth_conf, 3),
                        name=name,
                        visible=True
                    )
                else:
                    # Reset point if jumped excessively
                    smoothed_kp = curr.model_copy()
            else:
                smoothed_kp = curr.model_copy()

            smoothed_result[name] = smoothed_kp

        self._prev_keypoints = smoothed_result
        return smoothed_result

    def reset(self):
        """Reset historical smoothing state."""
        self._prev_keypoints.clear()
