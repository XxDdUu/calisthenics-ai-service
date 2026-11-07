import logging
import math

import numpy as np

from app.core.logging_context import get_request_id

logger = logging.getLogger(__name__)


def refine_wrists(kp_array: np.ndarray, elbow_conf_min: float = 0.8, wrist_conf_max: float = 0.6) -> np.ndarray:
    """Repair foreshortened arm extremities without pulling support hands upward."""
    pairs = [(5, 7, 9, 11, "left"), (6, 8, 10, 12, "right")]
    for shoulder_idx, elbow_idx, wrist_idx, hip_idx, side in pairs:
        if kp_array.ndim != 2 or kp_array.shape[0] <= max(shoulder_idx, elbow_idx, wrist_idx):
            continue
        shoulder, elbow, wrist = kp_array[shoulder_idx], kp_array[elbow_idx], kp_array[wrist_idx]
        rid = get_request_id()
        elbow_before = (float(elbow[0]), float(elbow[1]))
        elbow_conf = float(elbow[2])
        wrist_conf = float(wrist[2])
        if elbow_conf < elbow_conf_min or wrist_conf >= wrist_conf_max:
            reason = "elbow_confidence_below_threshold" if elbow_conf < elbow_conf_min else "wrist_confidence_above_threshold"
            logger.info(
                "[POSE_REFINEMENT] request_id=%s joint=%s_elbow status=not_applied reason=%s "
                "raw=(%.2f,%.2f) final=(%.2f,%.2f) elbow_conf=%.4f wrist_conf=%.4f",
                rid, side, reason, *elbow_before, *elbow_before, elbow_conf, wrist_conf
            )
            continue
        sx, sy = map(float, shoulder[:2]); ex, ey = map(float, elbow[:2]); wx, wy = map(float, wrist[:2])
        upper = math.hypot(ex - sx, ey - sy)
        forearm = math.hypot(wx - ex, wy - ey)
        torso = None
        if hip_idx < kp_array.shape[0]:
            torso = math.hypot(float(kp_array[hip_idx, 0]) - sx, float(kp_array[hip_idx, 1]) - sy)

        # A support wrist must travel down from the elbow. Never project it up.
        if forearm > 1e-3 and wy >= ey:
            if forearm < 0.85 * upper:
                target = 0.92 * upper
                ux, uy = (wx - ex) / forearm, (wy - ey) / forearm
                new_x, new_y = ex + ux * target, ey + uy * target
                kp_array[wrist_idx, 0] = new_x
                kp_array[wrist_idx, 1] = max(wy, new_y)
                logger.info(
                    "[POSE_REFINEMENT] request_id=%s joint=%s_wrist status=applied reason=short_support_forearm "
                    "before=(%.2f,%.2f) after=(%.2f,%.2f)",
                    rid, side, wx, wy, float(kp_array[wrist_idx, 0]), float(kp_array[wrist_idx, 1])
                )

            # Correct an elbow collapsed close to the shoulder, while retaining
            # the observed shoulder-to-wrist direction.
            if torso and upper < 0.45 * torso:
                shoulder_to_wrist = math.hypot(wx - sx, wy - sy)
                if shoulder_to_wrist > 1e-3:
                    ux, uy = (wx - sx) / shoulder_to_wrist, (wy - sy) / shoulder_to_wrist
                    target_upper = min(0.55 * torso, 0.52 * shoulder_to_wrist)
                    kp_array[elbow_idx, 0] = sx + ux * target_upper
                    kp_array[elbow_idx, 1] = sy + uy * target_upper
                    logger.info(
                        "[POSE_REFINEMENT] request_id=%s joint=%s_elbow status=applied "
                        "reason=upper_arm_short_vs_torso raw=(%.2f,%.2f) final=(%.2f,%.2f) "
                        "upper_arm_px=%.2f torso_px=%.2f",
                        rid, side, *elbow_before,
                        float(kp_array[elbow_idx, 0]), float(kp_array[elbow_idx, 1]), upper, torso
                    )
                else:
                    logger.info(
                        "[POSE_REFINEMENT] request_id=%s joint=%s_elbow status=not_applied "
                        "reason=shoulder_wrist_vector_too_short raw=(%.2f,%.2f) final=(%.2f,%.2f)",
                        rid, side, *elbow_before, *elbow_before
                    )
            else:
                logger.info(
                    "[POSE_REFINEMENT] request_id=%s joint=%s_elbow status=not_applied "
                    "reason=upper_arm_not_short raw=(%.2f,%.2f) final=(%.2f,%.2f) "
                    "upper_arm_px=%.2f torso_px=%s",
                    rid, side, *elbow_before, *elbow_before, upper,
                    f"{torso:.2f}" if torso is not None else "unavailable"
                )
        else:
            # Explicitly preserve raw y; this replaces the erroneous max(wy, wx).
            kp_array[wrist_idx, 1] = wy
            logger.info(
                "[POSE_REFINEMENT] request_id=%s joint=%s_elbow status=not_applied "
                "reason=not_support_arm_geometry raw=(%.2f,%.2f) final=(%.2f,%.2f)",
                rid, side, *elbow_before, *elbow_before
            )
    return kp_array
