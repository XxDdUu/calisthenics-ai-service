import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


def refine_wrists(kp_array: np.ndarray, elbow_conf_min: float = 0.8, wrist_conf_max: float = 0.6) -> np.ndarray:
    """Repair foreshortened arm extremities without pulling support hands upward."""
    pairs = [(5, 7, 9, 11, "left"), (6, 8, 10, 12, "right")]
    for shoulder_idx, elbow_idx, wrist_idx, hip_idx, side in pairs:
        if kp_array.ndim != 2 or kp_array.shape[0] <= max(shoulder_idx, elbow_idx, wrist_idx):
            continue
        shoulder, elbow, wrist = kp_array[shoulder_idx], kp_array[elbow_idx], kp_array[wrist_idx]
        if float(elbow[2]) < elbow_conf_min or float(wrist[2]) >= wrist_conf_max:
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
                logger.debug("Extended %s wrist along support ray", side)

            # Correct an elbow collapsed close to the shoulder, while retaining
            # the observed shoulder-to-wrist direction.
            if torso and upper < 0.45 * torso:
                shoulder_to_wrist = math.hypot(wx - sx, wy - sy)
                if shoulder_to_wrist > 1e-3:
                    ux, uy = (wx - sx) / shoulder_to_wrist, (wy - sy) / shoulder_to_wrist
                    target_upper = min(0.55 * torso, 0.52 * shoulder_to_wrist)
                    kp_array[elbow_idx, 0] = sx + ux * target_upper
                    kp_array[elbow_idx, 1] = sy + uy * target_upper
        else:
            # Explicitly preserve raw y; this replaces the erroneous max(wy, wx).
            kp_array[wrist_idx, 1] = wy
    return kp_array
