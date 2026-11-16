import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from app.biomechanics.angles import is_valid_coord
from app.biomechanics.confidence import (
    BODY_LANDMARK_GROUPS,
    check_body_visibility,
    is_valid_keypoint,
)
from app.schemas.analysis import CameraQualityAssessment, KeypointData

CORE_BODY_LANDMARKS: Tuple[str, ...] = (
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)

SIDE_VIEW_RECOMMENDED_EXERCISES = (
    "PLANCHE",
    "FRONT_LEVER",
    "PUSH_UP",
    "DIPS",
    "PIKE_PUSH_UP",
    "L_SIT",
)


class CameraQualityEvaluator:
    """
    Evaluates camera framing, visibility, landmark confidence, and dominant camera angle
    across sampled video frames.
    """

    @classmethod
    def evaluate(
        cls,
        frame_samples: List[Dict[str, Any]],
        exercise_name: Optional[str] = None,
        conf_threshold: float = 0.25,
    ) -> CameraQualityAssessment:
        if not frame_samples:
            return CameraQualityAssessment(
                overallQuality="POOR",
                score=0.0,
                viewAngle="UNKNOWN",
                fullBodyVisibleRatio=0.0,
                averageConfidence=0.0,
                missingLandmarks=list(CORE_BODY_LANDMARKS),
                warnings=["No video frames were sampled for camera quality assessment."],
            )

        total_frames = len(frame_samples)
        detected_frames = 0
        full_body_frames = 0
        confidence_sum = 0.0
        view_votes: Counter[str] = Counter()

        landmark_valid_counts: Counter[str] = Counter()
        pair_valid_counts: Counter[str] = Counter()

        for sample in frame_samples:
            pose_detected = bool(sample.get("pose_detected", False))
            kps: Dict[str, KeypointData] = (
                sample.get("raw_keypoints")
                or sample.get("keypoints")
                or {}
            )

            if not pose_detected or not kps:
                continue

            detected_frames += 1

            vis_info = check_body_visibility(kps, threshold=conf_threshold)
            if vis_info["full_body_visible"]:
                full_body_frames += 1

            frame_conf = cls._compute_frame_confidence(kps)
            confidence_sum += frame_conf

            frame_view = cls._classify_frame_view(kps, conf_threshold=conf_threshold)
            if frame_view != "UNKNOWN":
                view_votes[frame_view] += 1

            for lm_name in CORE_BODY_LANDMARKS:
                if is_valid_keypoint(kps.get(lm_name), threshold=conf_threshold):
                    landmark_valid_counts[lm_name] += 1

            for group_name, l_name, r_name in BODY_LANDMARK_GROUPS:
                if is_valid_keypoint(kps.get(l_name), conf_threshold) or is_valid_keypoint(
                    kps.get(r_name), conf_threshold
                ):
                    pair_valid_counts[group_name] += 1

        if detected_frames == 0:
            return CameraQualityAssessment(
                overallQuality="POOR",
                score=0.0,
                viewAngle="UNKNOWN",
                fullBodyVisibleRatio=0.0,
                averageConfidence=0.0,
                missingLandmarks=list(CORE_BODY_LANDMARKS),
                warnings=["No person pose was detected in the sampled frames."],
            )

        detection_ratio = detected_frames / float(total_frames)
        full_body_ratio = round(full_body_frames / float(total_frames), 3)
        avg_confidence = round(confidence_sum / float(total_frames), 3)

        view_angle = view_votes.most_common(1)[0][0] if view_votes else "UNKNOWN"

        missing_landmarks: List[str] = []
        if view_angle == "SIDE":
            for group_name, l_name, r_name in BODY_LANDMARK_GROUPS:
                if (pair_valid_counts[group_name] / float(total_frames)) < 0.50:
                    missing_landmarks.extend([l_name, r_name])
        else:
            for lm_name in CORE_BODY_LANDMARKS:
                if (landmark_valid_counts[lm_name] / float(total_frames)) < 0.50:
                    missing_landmarks.append(lm_name)

        score = round(
            max(
                0.0,
                min(
                    100.0,
                    100.0 * (0.30 * detection_ratio + 0.40 * full_body_ratio + 0.30 * avg_confidence),
                ),
            ),
            1,
        )

        if full_body_ratio < 0.35 or avg_confidence < 0.30 or score < 50.0:
            overall_quality = "POOR"
        elif score >= 75.0 and full_body_ratio >= 0.70 and avg_confidence >= 0.55:
            overall_quality = "GOOD"
        else:
            overall_quality = "FAIR"

        warnings: List[str] = []
        if detection_ratio < 0.70:
            warnings.append("Person was not consistently detected across the video.")
        if full_body_ratio < 0.70:
            warnings.append(
                "Full body is not consistently visible; keep shoulders through ankles inside the frame."
            )
        if avg_confidence < 0.50:
            warnings.append(
                "Low landmark detection confidence; improve lighting, contrast, or camera focus."
            )
        if missing_landmarks:
            warnings.append(
                f"Frequently missing or occluded landmarks: {', '.join(missing_landmarks[:6])}."
            )

        ex_upper = (exercise_name or "").upper()
        if view_angle == "FRONT" and any(tag in ex_upper for tag in SIDE_VIEW_RECOMMENDED_EXERCISES):
            warnings.append(
                "Side-on or 45-degree camera angle is recommended for accurate depth and alignment tracking."
            )
        elif view_angle == "UNKNOWN":
            warnings.append(
                "Camera view angle could not be reliably determined; ensure torso and hips are clearly visible."
            )

        return CameraQualityAssessment(
            overallQuality=overall_quality,
            score=score,
            viewAngle=view_angle,
            fullBodyVisibleRatio=full_body_ratio,
            averageConfidence=avg_confidence,
            missingLandmarks=missing_landmarks,
            warnings=warnings,
        )

    @classmethod
    def _compute_frame_confidence(cls, keypoints: Dict[str, KeypointData]) -> float:
        """
        Compute representative body-landmark confidence for one frame using
        max(left, right) per bilateral joint group so clean side-view frames are
        not penalized for far-side self-occlusion.
        """
        group_confs: List[float] = []
        for _, l_name, r_name in BODY_LANDMARK_GROUPS:
            l_kp = keypoints.get(l_name)
            r_kp = keypoints.get(r_name)
            l_c = float(l_kp.confidence) if (l_kp is not None and is_valid_coord(l_kp.confidence)) else 0.0
            r_c = float(r_kp.confidence) if (r_kp is not None and is_valid_coord(r_kp.confidence)) else 0.0
            group_confs.append(max(0.0, min(1.0, max(l_c, r_c))))

        if not group_confs:
            return 0.0
        return sum(group_confs) / len(group_confs)

    @classmethod
    def _classify_frame_view(
        cls,
        keypoints: Dict[str, KeypointData],
        conf_threshold: float = 0.25,
    ) -> str:
        """
        Classify camera perspective as SIDE, FRONT, OBLIQUE, or UNKNOWN using
        orientation-invariant shoulder/hip width to torso-length ratios and
        bilateral visibility symmetry.
        """
        l_sh = keypoints.get("left_shoulder")
        r_sh = keypoints.get("right_shoulder")
        l_hip = keypoints.get("left_hip")
        r_hip = keypoints.get("right_hip")

        l_sh_ok = is_valid_keypoint(l_sh, conf_threshold)
        r_sh_ok = is_valid_keypoint(r_sh, conf_threshold)
        l_hip_ok = is_valid_keypoint(l_hip, conf_threshold)
        r_hip_ok = is_valid_keypoint(r_hip, conf_threshold)

        if not (l_sh_ok or r_sh_ok) or not (l_hip_ok or r_hip_ok):
            return "UNKNOWN"

        # Pure unilateral shoulder or hip visibility indicates profile/side orientation
        if (l_sh_ok != r_sh_ok) and (l_hip_ok != r_hip_ok):
            return "SIDE"

        # Compute shoulder and hip midpoints to obtain torso scale
        if l_sh_ok and r_sh_ok:
            assert l_sh is not None and r_sh is not None
            sh_mid = ((l_sh.x + r_sh.x) / 2.0, (l_sh.y + r_sh.y) / 2.0)
            sh_sep = math.hypot(l_sh.x - r_sh.x, l_sh.y - r_sh.y)
        else:
            single_sh = l_sh if l_sh_ok else r_sh
            assert single_sh is not None
            sh_mid = (single_sh.x, single_sh.y)
            sh_sep = None

        if l_hip_ok and r_hip_ok:
            assert l_hip is not None and r_hip is not None
            hip_mid = ((l_hip.x + r_hip.x) / 2.0, (l_hip.y + r_hip.y) / 2.0)
            hip_sep = math.hypot(l_hip.x - r_hip.x, l_hip.y - r_hip.y)
        else:
            single_hip = l_hip if l_hip_ok else r_hip
            assert single_hip is not None
            hip_mid = (single_hip.x, single_hip.y)
            hip_sep = None

        torso_len = math.hypot(sh_mid[0] - hip_mid[0], sh_mid[1] - hip_mid[1])
        if torso_len < 10.0:
            # Extremely foreshortened torso (e.g. horizontal body facing straight at camera)
            if sh_sep is not None and sh_sep >= 20.0:
                return "FRONT"
            return "UNKNOWN"

        if sh_sep is None:
            return "SIDE"

        sh_ratio = sh_sep / torso_len
        hip_ratio = (hip_sep / torso_len) if hip_sep is not None else sh_ratio * 0.7

        # Check strong bilateral confidence imbalance typical of side-profile occlusion
        l_limb_conf = sum(
            keypoints[k].confidence
            for k in ("left_shoulder", "left_elbow", "left_wrist", "left_hip", "left_knee", "left_ankle")
            if k in keypoints and is_valid_keypoint(keypoints[k], conf_threshold)
        )
        r_limb_conf = sum(
            keypoints[k].confidence
            for k in ("right_shoulder", "right_elbow", "right_wrist", "right_hip", "right_knee", "right_ankle")
            if k in keypoints and is_valid_keypoint(keypoints[k], conf_threshold)
        )
        conf_asymmetry = abs(l_limb_conf - r_limb_conf)

        if sh_ratio < 0.25 and hip_ratio < 0.22:
            return "SIDE"
        if sh_ratio < 0.30 and conf_asymmetry > 1.2:
            return "SIDE"
        if sh_ratio >= 0.48 and hip_ratio >= 0.22:
            return "FRONT"
        if sh_ratio >= 0.48 and l_hip_ok and r_hip_ok:
            return "FRONT"
        return "OBLIQUE"
