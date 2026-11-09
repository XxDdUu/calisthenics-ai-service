import math
from typing import Dict, Optional, Tuple
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


class LowPassFilter:
    """Standard exponential 1D low pass filter for 1€ Filter."""

    def __init__(self, alpha: float = 0.5):
        self.alpha: float = max(1e-4, min(1.0, alpha))
        self.s: Optional[float] = None

    def filter(self, value: float, alpha: Optional[float] = None) -> float:
        if alpha is not None:
            self.alpha = max(1e-4, min(1.0, alpha))
        if self.s is None:
            self.s = value
        else:
            self.s = self.alpha * value + (1.0 - self.alpha) * self.s
        return self.s

    def reset(self):
        self.s = None


class OneEuroFilter:
    """
    1€ Filter: Adaptive low-pass filter for noisy human pose keypoints.
    Minimizes jitter when stationary (low cutoff frequency) and eliminates lag
    during rapid movement (increasing cutoff frequency proportionally to velocity).
    Reference: Casiez, Roussel, Vogel (CHI 2012)
    """

    def __init__(
        self,
        freq: float = 30.0,
        mincutoff: float = 1.0,
        beta: float = 0.007,
        dcutoff: float = 1.0
    ):
        self.freq = max(1e-3, float(freq))
        self.mincutoff = max(1e-4, float(mincutoff))
        self.beta = max(0.0, float(beta))
        self.dcutoff = max(1e-4, float(dcutoff))
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_time: Optional[float] = None

    def _alpha(self, rate: float, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / rate
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float, timestamp: Optional[float] = None) -> float:
        if self.last_time is not None and timestamp is not None and timestamp > self.last_time:
            rate = 1.0 / (timestamp - self.last_time)
        else:
            rate = self.freq
        self.last_time = timestamp

        prev_x = self.x_filter.s
        dx = 0.0 if prev_x is None else (x - prev_x) * rate
        edx = self.dx_filter.filter(dx, self._alpha(rate, self.dcutoff))

        cutoff = self.mincutoff + self.beta * abs(edx)
        return self.x_filter.filter(x, self._alpha(rate, cutoff))

    def reset(self):
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_time = None


class OneEuroFilter2D:
    """2D One Euro Filter for (X, Y) pixel or normalized coordinates."""

    def __init__(
        self,
        freq: float = 30.0,
        mincutoff: float = 1.0,
        beta: float = 0.007,
        dcutoff: float = 1.0
    ):
        self.filter_x = OneEuroFilter(freq=freq, mincutoff=mincutoff, beta=beta, dcutoff=dcutoff)
        self.filter_y = OneEuroFilter(freq=freq, mincutoff=mincutoff, beta=beta, dcutoff=dcutoff)

    def filter(self, x: float, y: float, timestamp: Optional[float] = None) -> Tuple[float, float]:
        fx = self.filter_x.filter(x, timestamp)
        fy = self.filter_y.filter(y, timestamp)
        return fx, fy

    def reset(self):
        self.filter_x.reset()
        self.filter_y.reset()


class OneEuroKeypointSmoother:
    """
    Temporal pose keypoint smoother leveraging One Euro Filter.
    Applies adaptive jitter filtering per COCO keypoint while preserving fast athletic transitions.
    """

    def __init__(
        self,
        freq: float = 30.0,
        mincutoff: float = 1.0,
        beta: float = 0.007,
        dcutoff: float = 1.0,
        max_jump_distance_px: float = 120.0
    ):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.max_jump_distance_px = max_jump_distance_px
        self._filters: Dict[str, OneEuroFilter2D] = {}
        self._prev_keypoints: Dict[str, KeypointData] = {}

    def smooth(
        self,
        keypoints: Dict[str, KeypointData],
        timestamp_sec: Optional[float] = None
    ) -> Dict[str, KeypointData]:
        if not keypoints:
            self.reset()
            return {}

        smoothed_result: Dict[str, KeypointData] = {}

        for name, curr in keypoints.items():
            if name not in self._filters:
                self._filters[name] = OneEuroFilter2D(
                    freq=self.freq,
                    mincutoff=self.mincutoff,
                    beta=self.beta,
                    dcutoff=self.dcutoff
                )

            filter_2d = self._filters[name]
            prev = self._prev_keypoints.get(name)

            if prev is not None and prev.visible and curr.visible:
                dist = math.hypot(curr.x - prev.x, curr.y - prev.y)
                if dist > self.max_jump_distance_px:
                    # Large teleportation jump - reset filter for this landmark
                    filter_2d.reset()
                    sx, sy = filter_2d.filter(curr.x, curr.y, timestamp_sec)
                else:
                    sx, sy = filter_2d.filter(curr.x, curr.y, timestamp_sec)
                
                # Confidence is smoothed via weighted EMA
                smooth_conf = 0.7 * curr.confidence + 0.3 * prev.confidence
                smoothed_kp = KeypointData(
                    x=round(sx, 2),
                    y=round(sy, 2),
                    confidence=round(smooth_conf, 3),
                    name=name,
                    visible=True
                )
            else:
                filter_2d.reset()
                sx, sy = filter_2d.filter(curr.x, curr.y, timestamp_sec)
                smoothed_kp = KeypointData(
                    x=round(sx, 2),
                    y=round(sy, 2),
                    confidence=round(curr.confidence, 3),
                    name=name,
                    visible=curr.visible
                )

            smoothed_result[name] = smoothed_kp

        self._prev_keypoints = smoothed_result
        return smoothed_result

    def reset(self):
        for f in self._filters.values():
            f.reset()
        self._prev_keypoints.clear()
