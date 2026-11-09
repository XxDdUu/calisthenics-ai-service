import logging
import time
from typing import Dict, List, Optional
from app.schemas.analysis import FeedbackSeverity, LiveFeedbackItem

logger = logging.getLogger(__name__)

SEVERITY_ORDER = {
    FeedbackSeverity.ERROR: 0,
    FeedbackSeverity.WARNING: 1,
    FeedbackSeverity.INFO: 2
}


class FeedbackEngine:
    """
    Intelligent coaching feedback engine:
    - Prioritizes top issue by severity (CRITICAL/ERROR > WARNING > INFO)
    - Applies temporal cooldown / debouncing per feedback code
    - Prevents contradictory or overwhelming simultaneous alerts
    """

    def __init__(self, cooldown_sec: float = 2.0, max_items: int = 3):
        self.cooldown_sec = cooldown_sec
        self.max_items = max_items
        self._last_emitted_timestamps: Dict[str, float] = {}

    def reset(self):
        self._last_emitted_timestamps.clear()

    def process_feedback(
        self,
        raw_items: List[LiveFeedbackItem],
        timestamp_sec: Optional[float] = None
    ) -> List[LiveFeedbackItem]:
        """
        Filters and prioritizes feedback items for the current frame.
        """
        if not raw_items:
            return []

        now = timestamp_sec if timestamp_sec is not None and timestamp_sec > 0 else time.time()

        # Sort raw items by severity (most critical first)
        sorted_items = sorted(
            raw_items,
            key=lambda item: SEVERITY_ORDER.get(item.severity, 99)
        )

        filtered: List[LiveFeedbackItem] = []
        seen_codes = set()

        for item in sorted_items:
            if item.code in seen_codes:
                continue

            last_time = self._last_emitted_timestamps.get(item.code, 0.0)
            
            # If critical/error, always allow or shorter cooldown
            cooldown = self.cooldown_sec * 0.5 if item.severity == FeedbackSeverity.ERROR else self.cooldown_sec

            if now - last_time >= cooldown:
                filtered.append(item)
                seen_codes.add(item.code)
                self._last_emitted_timestamps[item.code] = now

            if len(filtered) >= self.max_items:
                break

        # If all candidates were suppressed by cooldown but we have a critical item, keep it
        if not filtered and sorted_items:
            top_item = sorted_items[0]
            if top_item.severity in (FeedbackSeverity.ERROR, FeedbackSeverity.WARNING):
                filtered.append(top_item)

        return filtered
