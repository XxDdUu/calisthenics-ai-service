import math
from typing import Any, Dict, List, Optional
import pytest

from app.biomechanics.pull_up_counter import (
    PullUpCounterConfig,
    PullUpRepCounter,
    PullUpState,
    extract_pull_up_geometry,
)
from app.biomechanics.push_up_counter import extract_push_up_geometry
from app.biomechanics.squat_counter import extract_squat_geometry
from app.exercises.planche import PlancheAnalyzer
from app.exercises.pull_up import PullUpAnalyzer
from app.exercises.push_up import PushUpAnalyzer
from app.exercises.registry import exercise_registry
from app.exercises.squat import SquatAnalyzer
from app.schemas.analysis import KeypointData
from app.services.live_session import LiveSession


def _kp(
    name: str,
    x: float,
    y: float,
    conf: float = 0.95,
    visible: bool = True,
) -> KeypointData:
    return KeypointData(name=name, x=float(x), y=float(y), confidence=float(conf), visible=visible)


def _sample(
    frame_idx: int,
    fps: float,
    kps: Dict[str, KeypointData],
    pose_detected: bool = True,
    det_conf: float = 0.95,
) -> Dict[str, Any]:
    ts_sec = frame_idx / fps
    return {
        "frame_index": frame_idx,
        "timestamp_ms": int(round(ts_sec * 1000.0)),
        "timestamp_sec": ts_sec,
        "keypoints": kps,
        "raw_keypoints": kps,
        "pose_detected": pose_detected,
        "detection_confidence": det_conf,
    }


def _make_squat_kps(
    knee_angle_deg: float,
    torso_lean_deg: float = 18.0,
    valgus_ratio: float = 1.0,
    right_knee_angle_deg: Optional[float] = None,
    include_shoulders: bool = True,
    shoulder_conf: float = 0.95,
    occlude_right_leg: bool = False,
) -> Dict[str, KeypointData]:
    """Generate geometrically exact COCO keypoints for a Squat frame."""
    l_thigh = 100.0
    l_shank = 100.0
    l_torso = 120.0

    def _leg_and_sh(side_x: float, knee_x_offset: float, k_deg: float) -> Dict[str, KeypointData]:
        rad_k = math.radians(k_deg)
        ak_x, ak_y = side_x, 400.0
        kn_x, kn_y = side_x + knee_x_offset, 400.0 - l_shank
        # Vector K->H at interior angle k_deg from vertical downward K->A (0, +1)
        hip_x = kn_x - l_thigh * math.sin(rad_k)
        hip_y = kn_y + l_thigh * math.cos(rad_k)

        rad_t = math.radians(torso_lean_deg)
        sh_x = hip_x + l_torso * math.sin(rad_t)
        sh_y = hip_y - l_torso * math.cos(rad_t)
        return {
            "hip": _kp("hip", hip_x, hip_y),
            "knee": _kp("knee", kn_x, kn_y),
            "ankle": _kp("ankle", ak_x, ak_y),
            "shoulder": _kp("shoulder", sh_x, sh_y, conf=shoulder_conf, visible=shoulder_conf >= 0.25),
        }

    # Ankle separation = 80px (left=160, right=240)
    ankle_half = 40.0
    knee_inward = ankle_half * (1.0 - valgus_ratio)
    r_deg = right_knee_angle_deg if right_knee_angle_deg is not None else knee_angle_deg

    left_pts = _leg_and_sh(160.0, +knee_inward, knee_angle_deg)
    right_pts = _leg_and_sh(240.0, -knee_inward, r_deg)

    kps: Dict[str, KeypointData] = {
        "left_hip": _kp("left_hip", left_pts["hip"].x, left_pts["hip"].y),
        "left_knee": _kp("left_knee", left_pts["knee"].x, left_pts["knee"].y),
        "left_ankle": _kp("left_ankle", left_pts["ankle"].x, left_pts["ankle"].y),
    }
    if not occlude_right_leg:
        kps["right_hip"] = _kp("right_hip", right_pts["hip"].x, right_pts["hip"].y)
        kps["right_knee"] = _kp("right_knee", right_pts["knee"].x, right_pts["knee"].y)
        kps["right_ankle"] = _kp("right_ankle", right_pts["ankle"].x, right_pts["ankle"].y)

    if include_shoulders:
        kps["left_shoulder"] = _kp(
            "left_shoulder", left_pts["shoulder"].x, left_pts["shoulder"].y,
            conf=shoulder_conf, visible=shoulder_conf >= 0.25
        )
        if not occlude_right_leg:
            kps["right_shoulder"] = _kp(
                "right_shoulder", right_pts["shoulder"].x, right_pts["shoulder"].y,
                conf=shoulder_conf, visible=shoulder_conf >= 0.25
            )
    return kps


def _make_push_up_kps(
    elbow_angle_deg: float,
    hip_y_offset: float = 0.0,
    right_elbow_angle_deg: Optional[float] = None,
    shoulder_shift_x: float = 0.0,
    include_lower_body: bool = True,
    occlude_right_arm: bool = False,
) -> Dict[str, KeypointData]:
    """Generate geometrically exact COCO keypoints for a Push-Up frame."""
    seg = 60.0

    def _arm_pts(wx: float, e_deg: float) -> Dict[str, KeypointData]:
        beta = math.radians((180.0 - e_deg) / 2.0)
        el_x = wx + seg * math.sin(beta)
        el_y = 300.0 - seg * math.cos(beta)
        sh_x = wx + shoulder_shift_x
        sh_y = el_y - seg * math.cos(beta)
        return {
            "wrist": _kp("wrist", wx, 300.0),
            "elbow": _kp("elbow", el_x, el_y),
            "shoulder": _kp("shoulder", sh_x, sh_y),
        }

    r_deg = right_elbow_angle_deg if right_elbow_angle_deg is not None else elbow_angle_deg
    l_arm = _arm_pts(100.0, elbow_angle_deg)
    r_arm = _arm_pts(120.0, r_deg)

    kps: Dict[str, KeypointData] = {
        "left_wrist": _kp("left_wrist", l_arm["wrist"].x, l_arm["wrist"].y),
        "left_elbow": _kp("left_elbow", l_arm["elbow"].x, l_arm["elbow"].y),
        "left_shoulder": _kp("left_shoulder", l_arm["shoulder"].x, l_arm["shoulder"].y),
    }
    if not occlude_right_arm:
        kps["right_wrist"] = _kp("right_wrist", r_arm["wrist"].x, r_arm["wrist"].y)
        kps["right_elbow"] = _kp("right_elbow", r_arm["elbow"].x, r_arm["elbow"].y)
        kps["right_shoulder"] = _kp("right_shoulder", r_arm["shoulder"].x, r_arm["shoulder"].y)

    if include_lower_body:
        sh_x, sh_y = l_arm["shoulder"].x, l_arm["shoulder"].y
        kps["left_hip"] = _kp("left_hip", sh_x + 100.0, sh_y + hip_y_offset)
        kps["left_ankle"] = _kp("left_ankle", sh_x + 200.0, sh_y)
        if not occlude_right_arm:
            kps["right_hip"] = _kp("right_hip", sh_x + 100.0, sh_y + hip_y_offset)
            kps["right_ankle"] = _kp("right_ankle", sh_x + 200.0, sh_y)

    return kps


def _make_pull_up_kps(
    elbow_angle_deg: float,
    chin_over_bar: bool = True,
    swing_x_offset: float = 0.0,
    right_elbow_angle_deg: Optional[float] = None,
    include_head: bool = True,
    include_lower_body: bool = True,
    occlude_right_arm: bool = False,
) -> Dict[str, KeypointData]:
    """Generate geometrically exact COCO keypoints for a Pull-Up frame."""
    seg = 55.0
    torso_len = 100.0
    leg_len = 90.0

    def _arm_pts(wx: float, dir_sign: float, e_deg: float) -> Dict[str, KeypointData]:
        beta = math.radians((180.0 - e_deg) / 2.0)
        el_x = wx + dir_sign * seg * math.sin(beta)
        el_y = 100.0 + seg * math.cos(beta)
        sh_x = wx
        sh_y = el_y + seg * math.cos(beta)
        return {
            "wrist": _kp("wrist", wx, 100.0),
            "elbow": _kp("elbow", el_x, el_y),
            "shoulder": _kp("shoulder", sh_x, sh_y),
        }

    r_deg = right_elbow_angle_deg if right_elbow_angle_deg is not None else elbow_angle_deg
    l_arm = _arm_pts(180.0, -1.0, elbow_angle_deg)
    r_arm = _arm_pts(240.0, +1.0, r_deg)

    kps: Dict[str, KeypointData] = {
        "left_wrist": _kp("left_wrist", l_arm["wrist"].x, l_arm["wrist"].y),
        "left_elbow": _kp("left_elbow", l_arm["elbow"].x, l_arm["elbow"].y),
        "left_shoulder": _kp("left_shoulder", l_arm["shoulder"].x, l_arm["shoulder"].y),
    }
    if not occlude_right_arm:
        kps["right_wrist"] = _kp("right_wrist", r_arm["wrist"].x, r_arm["wrist"].y)
        kps["right_elbow"] = _kp("right_elbow", r_arm["elbow"].x, r_arm["elbow"].y)
        kps["right_shoulder"] = _kp("right_shoulder", r_arm["shoulder"].x, r_arm["shoulder"].y)

    mid_sh_y = l_arm["shoulder"].y
    if include_head:
        if elbow_angle_deg <= 95.0:
            nose_y = 85.0 if chin_over_bar else 120.0
        else:
            nose_y = mid_sh_y - 25.0
        kps["nose"] = _kp("nose", 210.0, nose_y)

    if include_lower_body:
        kps["left_hip"] = _kp("left_hip", 190.0 + swing_x_offset, mid_sh_y + torso_len)
        kps["left_ankle"] = _kp("left_ankle", 190.0 + swing_x_offset * 1.4, mid_sh_y + torso_len + leg_len)
        if not occlude_right_arm:
            kps["right_hip"] = _kp("right_hip", 230.0 + swing_x_offset, mid_sh_y + torso_len)
            kps["right_ankle"] = _kp("right_ankle", 230.0 + swing_x_offset * 1.4, mid_sh_y + torso_len + leg_len)

    return kps


def _interpolate_cycle(angles: List[float]) -> List[float]:
    return list(angles)


# ==============================================================================
# 1. BASE CONTRACT & REGISTRY TESTS
# ==============================================================================

def test_base_and_registry_v1():
    assert PlancheAnalyzer().analyze_video_sequence([], fps=10.0) is None
    assert isinstance(exercise_registry.get_analyzer("BODYWEIGHT_SQUAT"), SquatAnalyzer)
    assert isinstance(exercise_registry.get_analyzer("PUSHUPS"), PushUpAnalyzer)
    assert isinstance(exercise_registry.get_analyzer("CHIN_UPS"), PullUpAnalyzer)


# ==============================================================================
# 2. SQUAT BIOMECHANICS v1 TESTS
# ==============================================================================

def test_squat_3_clean_valid_reps():
    analyzer = SquatAnalyzer()
    fps = 10.0
    rep = [172.0, 170.0, 145.0, 120.0, 96.0, 88.0, 90.0, 115.0, 145.0, 170.0, 172.0, 172.0]
    seq = rep * 3
    samples = [_sample(i, fps, _make_squat_kps(a, torso_lean_deg=18.0, valgus_ratio=1.0)) for i, a in enumerate(seq)]

    res = analyzer.analyze_video_sequence(samples, fps=fps, side_view=False)
    assert res["status"] == "COMPLETED"
    assert res["repCount"] == 3
    assert res["validReps"] == 3
    assert res["incompleteReps"] == 0
    assert res["totalAttemptedReps"] == 3
    assert res["score"] is not None and res["score"] >= 85.0
    assert res["formAccuracy"] is not None and res["formAccuracy"] >= 85.0
    assert res["stability"] is not None and res["stability"] >= 75.0
    assert res["alignment"] is not None and res["alignment"] >= 85.0

    assert len(res["reps"]) == 3
    for idx, r in enumerate(res["reps"], start=1):
        assert r["repNumber"] == idx
        assert r["isValid"] is True
        assert r["status"] == "VALID"
        assert r["depthClassification"] == "GOOD"
        assert r["minAngleDeg"] <= 100.0
        assert r["durationMs"] > 0

    mm = res["movementMetrics"]
    for key in ("squat_depth", "hip_angle", "torso_inclination", "knee_alignment", "movement_consistency"):
        assert mm[key]["status"] == "VALID"
        assert mm[key]["value"] is not None
        assert mm[key]["score"] is not None
        assert mm[key]["confidence"] > 0.5


def test_squat_mixed_valid_and_incomplete_reps():
    analyzer = SquatAnalyzer()
    fps = 10.0
    valid_rep = [172.0, 145.0, 115.0, 92.0, 86.0, 94.0, 125.0, 155.0, 172.0, 172.0, 172.0]
    shallow_rep = [172.0, 148.0, 130.0, 118.0, 126.0, 145.0, 165.0, 172.0, 172.0, 172.0]
    seq = valid_rep + shallow_rep + valid_rep
    samples = [_sample(i, fps, _make_squat_kps(a)) for i, a in enumerate(seq)]

    res = analyzer.analyze_video_sequence(samples, fps=fps, side_view=False)
    assert res["status"] == "COMPLETED"
    assert res["repCount"] == 3
    assert res["validReps"] == 2
    assert res["incompleteReps"] == 1
    assert res["totalAttemptedReps"] == 3
    assert [r["status"] for r in res["reps"]] == ["VALID", "INCOMPLETE", "VALID"]
    assert res["reps"][1]["isValid"] is False
    assert "SHALLOW_DEPTH" in res["reps"][1]["issues"]
    assert any(iss["code"] == "SHALLOW_DEPTH" and iss["affectedReps"] == [2] for iss in res["issues"])


def test_squat_boundary_jitter_no_double_counting():
    analyzer = SquatAnalyzer()
    fps = 10.0
    # Jitter around 150 threshold (no attempt), then 1 rep with bottom noise near 100
    seq = [
        172.0, 152.0, 148.0, 151.0, 149.0, 153.0, 170.0, 170.0, 170.0,
        142.0, 118.0, 102.0, 96.0, 103.0, 97.0, 105.0, 118.0, 142.0, 168.0, 170.0
    ]
    samples = [_sample(i, fps, _make_squat_kps(a)) for i, a in enumerate(seq)]
    res = analyzer.analyze_video_sequence(samples, fps=fps)

    assert res["repCount"] == 1
    assert res["validReps"] == 1
    assert res["incompleteReps"] == 0


def test_squat_valgus_and_forward_lean_and_side_view_unavailable():
    analyzer = SquatAnalyzer()
    fps = 10.0
    rep = [172.0, 145.0, 115.0, 92.0, 88.0, 95.0, 125.0, 165.0, 172.0, 172.0]
    samples_bad = [
        _sample(i, fps, _make_squat_kps(a, torso_lean_deg=54.0 if a < 140.0 else 15.0, valgus_ratio=0.65 if a < 140.0 else 1.0))
        for i, a in enumerate(rep * 2)
    ]
    res_bad = analyzer.analyze_video_sequence(samples_bad, fps=fps, side_view=False)
    codes = {iss["code"] for iss in res_bad["issues"]}
    assert "EXCESSIVE_FORWARD_LEAN" in codes
    assert "KNEE_VALGUS" in codes

    # In strict side_view=True, knee_alignment must be UNAVAILABLE
    res_side = analyzer.analyze_video_sequence(samples_bad, fps=fps, side_view=True)
    ka = res_side["movementMetrics"]["knee_alignment"]
    assert ka["status"] == "UNAVAILABLE"
    assert ka["value"] is None
    assert ka["score"] is None
    assert ka["confidence"] == 0.0


def test_squat_missing_shoulders_no_fabricated_metrics():
    analyzer = SquatAnalyzer()
    fps = 10.0
    rep = [172.0, 145.0, 115.0, 90.0, 88.0, 95.0, 125.0, 165.0, 172.0]
    samples = [
        _sample(i, fps, _make_squat_kps(a, include_shoulders=False))
        for i, a in enumerate(rep)
    ]
    # Direct extract_measurements must NOT fabricate 15.0 or 180.0 for missing shoulders
    m = analyzer.extract_measurements(samples[3]["keypoints"])
    assert "knee_angle" in m
    assert "torso_angle" not in m
    assert "hip_angle" not in m

    res = analyzer.analyze_video_sequence(samples, fps=fps)
    assert res["status"] == "COMPLETED"
    assert res["repCount"] == 1
    assert res["movementMetrics"]["squat_depth"]["status"] == "VALID"
    assert res["movementMetrics"]["torso_inclination"]["status"] == "UNAVAILABLE"
    assert res["movementMetrics"]["torso_inclination"]["value"] is None
    assert res["movementMetrics"]["hip_angle"]["status"] == "UNAVAILABLE"
    assert res["movementMetrics"]["hip_angle"]["score"] is None
    # Single rep -> movement_consistency UNAVAILABLE
    assert res["movementMetrics"]["movement_consistency"]["status"] == "UNAVAILABLE"


# ==============================================================================
# 3. PUSH-UP BIOMECHANICS v1 TESTS
# ==============================================================================

def test_push_up_3_clean_valid_reps():
    analyzer = PushUpAnalyzer()
    fps = 10.0
    rep = [170.0, 168.0, 140.0, 115.0, 88.0, 84.0, 90.0, 118.0, 145.0, 168.0, 170.0, 170.0]
    samples = [_sample(i, fps, _make_push_up_kps(a, hip_y_offset=0.0)) for i, a in enumerate(rep * 3)]

    res = analyzer.analyze_video_sequence(samples, fps=fps, side_view=False)
    assert res["status"] == "COMPLETED"
    assert res["repCount"] == 3
    assert res["validReps"] == 3
    assert res["incompleteReps"] == 0
    assert res["score"] is not None and res["score"] >= 85.0

    mm = res["movementMetrics"]
    assert mm["range_of_motion"]["status"] == "VALID"
    assert mm["body_alignment"]["status"] == "VALID"
    assert mm["hip_sag"]["value"] == 0.0
    assert mm["hip_piking"]["value"] == 0.0
    assert mm["elbow_symmetry"]["status"] == "VALID"


def test_push_up_mixed_valid_and_incomplete_reps():
    analyzer = PushUpAnalyzer()
    fps = 10.0
    valid_rep = [170.0, 140.0, 110.0, 85.0, 84.0, 115.0, 145.0, 168.0, 170.0, 170.0]
    partial_rep = [170.0, 140.0, 122.0, 112.0, 125.0, 148.0, 168.0, 170.0, 170.0]
    seq = valid_rep + partial_rep + valid_rep
    samples = [_sample(i, fps, _make_push_up_kps(a)) for i, a in enumerate(seq)]

    res = analyzer.analyze_video_sequence(samples, fps=fps)
    assert res["repCount"] == 3
    assert res["validReps"] == 2
    assert res["incompleteReps"] == 1
    assert [r["status"] for r in res["reps"]] == ["VALID", "INCOMPLETE", "VALID"]
    assert "INSUFFICIENT_DEPTH" in res["reps"][1]["issues"]


def test_push_up_hip_sag_and_hip_piking():
    analyzer = PushUpAnalyzer()
    fps = 10.0
    rep = [170.0, 140.0, 110.0, 86.0, 84.0, 115.0, 145.0, 168.0, 170.0, 170.0]

    # Hip sag: +45px downward
    sag_samples = [_sample(i, fps, _make_push_up_kps(a, hip_y_offset=45.0)) for i, a in enumerate(rep * 2)]
    res_sag = analyzer.analyze_video_sequence(sag_samples, fps=fps)
    assert any(iss["code"] == "HIP_SAG" for iss in res_sag["issues"])
    assert res_sag["movementMetrics"]["hip_sag"]["value"] > 50.0
    assert res_sag["movementMetrics"]["hip_piking"]["value"] == 0.0

    # Hip pike: -45px upward
    pike_samples = [_sample(i, fps, _make_push_up_kps(a, hip_y_offset=-45.0)) for i, a in enumerate(rep * 2)]
    res_pike = analyzer.analyze_video_sequence(pike_samples, fps=fps)
    assert any(iss["code"] == "HIP_PIKE" for iss in res_pike["issues"])
    assert res_pike["movementMetrics"]["hip_piking"]["value"] > 50.0
    assert res_pike["movementMetrics"]["hip_sag"]["value"] == 0.0


def test_push_up_occluded_lower_body_and_side_view_unavailable():
    analyzer = PushUpAnalyzer()
    fps = 10.0
    rep = [170.0, 140.0, 110.0, 86.0, 84.0, 115.0, 145.0, 168.0, 170.0]
    samples = [
        _sample(i, fps, _make_push_up_kps(a, include_lower_body=False, occlude_right_arm=True))
        for i, a in enumerate(rep)
    ]

    geom = extract_push_up_geometry(samples[3]["keypoints"], side_view=True)
    assert geom["is_pose_valid"] is True
    assert geom["body_alignment"] is None

    res = analyzer.analyze_video_sequence(samples, fps=fps, side_view=True)
    assert res["status"] == "COMPLETED"
    assert res["repCount"] == 1
    assert res["movementMetrics"]["range_of_motion"]["status"] == "VALID"
    for unavail_key in ("body_alignment", "hip_sag", "hip_piking", "elbow_symmetry"):
        assert res["movementMetrics"][unavail_key]["status"] == "UNAVAILABLE"
        assert res["movementMetrics"][unavail_key]["value"] is None
        assert res["movementMetrics"][unavail_key]["score"] is None
        assert res["movementMetrics"][unavail_key]["confidence"] == 0.0


# ==============================================================================
# 4. PULL-UP COUNTER & BIOMECHANICS v1 TESTS
# ==============================================================================

def test_pull_up_counter_state_machine_and_incomplete_cases():
    # 1. Valid rep + partial pull (< 130 without reaching top)
    counter = PullUpRepCounter()
    valid = [170.0, 145.0, 118.0, 80.0, 76.0, 90.0, 120.0, 152.0, 170.0, 170.0, 170.0]
    partial = [170.0, 138.0, 120.0, 112.0, 125.0, 150.0, 170.0]
    for idx, a in enumerate(valid + partial):
        counter.process_angle(a, timestamp_sec=idx * 0.1, frame_index=idx, timestamp_ms=idx * 100)
    records = counter.finalize()
    assert counter.rep_count == 1
    assert counter.incomplete_rep_count == 1
    assert len(records) == 2
    assert records[0]["isValid"] is True
    assert records[1]["isValid"] is False
    assert "INCOMPLETE_PULL" in records[1]["issues"]

    # 2. Incomplete bottom dead-hang extension (reversing at 134° < 145°)
    c2 = PullUpRepCounter()
    no_ext_seq = [
        170.0, 135.0, 100.0, 78.0, 76.0, 105.0, 134.0,
        112.0, 80.0, 76.0, 105.0, 135.0, 165.0
    ]
    for idx, a in enumerate(no_ext_seq):
        c2.process_angle(a, timestamp_sec=idx * 0.1, frame_index=idx, timestamp_ms=idx * 100)
    rec2 = c2.finalize()
    assert c2.rep_count == 1
    assert c2.incomplete_rep_count == 1
    assert "INCOMPLETE_EXTENSION" in rec2[0]["issues"]
    assert "INCOMPLETE_EXTENSION" in rec2[1]["issues"]

    # 3. Tracking loss safety
    c3 = PullUpRepCounter()
    c3.process_angle(170.0, timestamp_sec=0.0)
    c3.process_angle(125.0, timestamp_sec=0.1)
    assert c3.state == PullUpState.PULLING
    c3.process_angle(None, timestamp_sec=0.2)
    assert c3.is_tracking is False
    c3.process_angle(None, timestamp_sec=3.0)
    assert c3.state == PullUpState.DEAD_HANG
    assert c3.rep_count == 0


def test_pull_up_session_isolation_and_live_metrics_not_hardcoded():
    analyzer = PullUpAnalyzer()
    s1 = LiveSession(session_id="pu_user_1", exercise="PULL_UP")
    s2 = LiveSession(session_id="pu_user_2", exercise="PULL_UP")

    for i, a in enumerate([170.0, 120.0, 70.0, 70.0, 115.0, 165.0]):
        analyzer.update_dynamic_state(s1, True, {"elbow_angle": a}, i * 0.2)
    analyzer.update_dynamic_state(s2, True, {"elbow_angle": 170.0}, 0.0)

    assert s1.rep_count == 1
    assert s2.rep_count == 0

    # With missing keypoints / hips, evaluate_scoring_metrics must NOT return hardcoded 88.0 / 85.0
    metrics = analyzer.evaluate_scoring_metrics({"elbow_angle": 80.0}, keypoints=None, side_view=True)
    by_name = {m.name: m for m in metrics}
    assert by_name["body_control"].status == "INSUFFICIENT_DATA"
    assert by_name["body_control"].score == 0.0
    assert by_name["shoulder_positioning"].status == "INSUFFICIENT_DATA"
    assert by_name["shoulder_positioning"].score == 0.0
    assert by_name["movement_symmetry"].status == "INSUFFICIENT_DATA"


def test_pull_up_3_clean_reps_and_chin_over_bar_vs_below():
    analyzer = PullUpAnalyzer()
    fps = 10.0
    rep = [170.0, 168.0, 138.0, 108.0, 78.0, 72.0, 78.0, 108.0, 140.0, 168.0, 170.0, 170.0]

    # Clean chin-over-bar reps
    good_samples = [_sample(i, fps, _make_pull_up_kps(a, chin_over_bar=True)) for i, a in enumerate(rep * 3)]
    res_good = analyzer.analyze_video_sequence(good_samples, fps=fps)
    assert res_good["status"] == "COMPLETED"
    assert res_good["repCount"] == 3
    assert res_good["validReps"] == 3
    assert res_good["incompleteReps"] == 0
    assert res_good["movementMetrics"]["chin_to_bar"]["status"] == "VALID"
    assert res_good["movementMetrics"]["chin_to_bar"]["value"] > 0.0
    assert not any(iss["code"] == "INCOMPLETE_ROM_TOP" for iss in res_good["issues"])

    # Live check distinguishes CHIN_OVER_BAR vs INCOMPLETE_ROM_TOP
    _, fb_over, _ = analyzer.evaluate_live_form({"elbow_angle": 75.0, "chin_to_bar_clearance": 0.12})
    assert any(f.code == "CHIN_OVER_BAR" for f in fb_over)
    _, fb_below, _ = analyzer.evaluate_live_form({"elbow_angle": 75.0, "chin_to_bar_clearance": -0.15})
    assert any(f.code == "INCOMPLETE_ROM_TOP" for f in fb_below)

    # Sequence where chin stays below wrist/bar line
    low_chin_samples = [_sample(i, fps, _make_pull_up_kps(a, chin_over_bar=False)) for i, a in enumerate(rep * 2)]
    res_low = analyzer.analyze_video_sequence(low_chin_samples, fps=fps)
    assert res_low["movementMetrics"]["chin_to_bar"]["value"] < 0.0
    assert any(iss["code"] == "INCOMPLETE_ROM_TOP" for iss in res_low["issues"])


def test_pull_up_excessive_kipping_swing_and_occlusions():
    analyzer = PullUpAnalyzer()
    fps = 10.0
    rep = [170.0, 138.0, 108.0, 78.0, 74.0, 108.0, 140.0, 168.0, 170.0, 170.0]
    swings = [0.0, -45.0, 55.0, -35.0, 45.0, -40.0, 35.0, -20.0, 0.0, 0.0]

    kip_samples = [
        _sample(i, fps, _make_pull_up_kps(a, chin_over_bar=True, swing_x_offset=s_off))
        for i, (a, s_off) in enumerate(zip(rep * 2, swings * 2))
    ]
    res_kip = analyzer.analyze_video_sequence(kip_samples, fps=fps)
    assert any(iss["code"] == "EXCESSIVE_SWING" for iss in res_kip["issues"])
    assert res_kip["movementMetrics"]["body_swing"]["value"] > 0.30
    assert res_kip["stability"] is not None and res_kip["stability"] < 75.0

    # Occluded head & lower body -> chin_to_bar, body_swing, shoulder_angle all UNAVAILABLE
    occ_samples = [
        _sample(
            i, fps,
            _make_pull_up_kps(a, include_head=False, include_lower_body=False, occlude_right_arm=True),
        )
        for i, a in enumerate(rep)
    ]
    geo = extract_pull_up_geometry(occ_samples[3]["keypoints"], side_view=True)
    assert geo["is_pose_valid"] is True
    assert geo["chin_to_bar_clearance"] is None
    assert geo["shoulder_angle"] is None

    res_occ = analyzer.analyze_video_sequence(occ_samples, fps=fps, side_view=True)
    assert res_occ["status"] == "COMPLETED"
    assert res_occ["repCount"] == 1
    for unavail_key in ("chin_to_bar", "body_swing", "shoulder_angle", "elbow_symmetry"):
        assert res_occ["movementMetrics"][unavail_key]["status"] == "UNAVAILABLE"
        assert res_occ["movementMetrics"][unavail_key]["value"] is None
        assert res_occ["movementMetrics"][unavail_key]["score"] is None
        assert res_occ["movementMetrics"][unavail_key]["confidence"] == 0.0


# ==============================================================================
# 5. INSUFFICIENT POSE QUALITY & NO PERSON DETECTED TESTS
# ==============================================================================

@pytest.mark.parametrize("analyzer", [SquatAnalyzer(), PushUpAnalyzer(), PullUpAnalyzer()])
def test_no_person_and_insufficient_pose_quality_never_fabricate(analyzer):
    # 1. Empty / no person detected
    res_empty = analyzer.analyze_video_sequence([], fps=10.0)
    assert res_empty["status"] == "NO_PERSON_DETECTED"
    assert res_empty["score"] is None
    assert res_empty["repCount"] == 0
    for m in res_empty["movementMetrics"].values():
        assert m["status"] == "UNAVAILABLE"
        assert m["value"] is None
        assert m["score"] is None

    # 2. All frames have pose_detected=False
    no_pose_samples = [_sample(i, 10.0, {}, pose_detected=False, det_conf=0.0) for i in range(10)]
    res_none = analyzer.analyze_video_sequence(no_pose_samples, fps=10.0)
    assert res_none["status"] == "NO_PERSON_DETECTED"
    assert res_none["score"] is None

    # 3. Low-confidence / < 20% valid frames (only 2 valid frames out of 15)
    low_conf_kp = {
        "left_shoulder": _kp("left_shoulder", 100, 100, conf=0.10, visible=True),
        "left_elbow": _kp("left_elbow", 100, 150, conf=0.05, visible=False),
        "left_wrist": _kp("left_wrist", 100, 200, conf=0.10, visible=True),
        "left_hip": _kp("left_hip", 100, 220, conf=0.10, visible=True),
        "left_knee": _kp("left_knee", 100, 300, conf=0.08, visible=True),
        "left_ankle": _kp("left_ankle", 100, 380, conf=0.05, visible=False),
    }
    valid_kps = (
        _make_squat_kps(170.0) if isinstance(analyzer, SquatAnalyzer)
        else _make_push_up_kps(170.0) if isinstance(analyzer, PushUpAnalyzer)
        else _make_pull_up_kps(170.0)
    )
    mixed_samples = [
        _sample(i, 10.0, valid_kps if i < 2 else low_conf_kp, pose_detected=True)
        for i in range(15)
    ]
    res_low = analyzer.analyze_video_sequence(mixed_samples, fps=10.0)
    assert res_low["status"] == "INSUFFICIENT_POSE_QUALITY"
    assert res_low["score"] is None
    assert res_low["formAccuracy"] is None
    assert res_low["repCount"] == 0
    for m in res_low["movementMetrics"].values():
        assert m["status"] == "UNAVAILABLE"
        assert m["value"] is None
        assert m["score"] is None
        assert m["confidence"] == 0.0


def test_squat_incomplete_extension_and_normalized_coords():
    analyzer = SquatAnalyzer()
    fps = 10.0
    # Reaches bottom 88, ascends only to 142 (failed lockout), reverses to second rep, finishes cleanly
    seq = [
        172.0, 145.0, 115.0, 88.0, 86.0, 110.0, 142.0,
        118.0, 88.0, 86.0, 115.0, 145.0, 170.0, 172.0
    ]
    samples = [_sample(i, fps, _make_squat_kps(a)) for i, a in enumerate(seq)]
    res = analyzer.analyze_video_sequence(samples, fps=fps)
    assert res["validReps"] == 1
    assert res["incompleteReps"] == 1
    assert "INCOMPLETE_EXTENSION" in res["reps"][0]["issues"]

    # Scale-invariance check with normalized [0..1] coordinates for knee valgus
    kps_px = _make_squat_kps(90.0, valgus_ratio=0.70)
    kps_norm = {
        name: _kp(name, kp.x / 640.0, kp.y / 480.0, conf=kp.confidence, visible=kp.visible)
        for name, kp in kps_px.items()
    }
    geo_norm = extract_squat_geometry(kps_norm, side_view=False)
    assert geo_norm["is_pose_valid"] is True
    assert geo_norm["knee_ankle_ratio"] is not None
    assert geo_norm["knee_ankle_ratio"] < 0.80


def test_push_up_boundary_jitter_and_incomplete_lockout():
    analyzer = PushUpAnalyzer()
    fps = 10.0
    # Jitter around 145 deg, then 1 clean rep, then 1 rep that stalls at 138 deg without lockout
    seq = [
        168.0, 148.0, 143.0, 147.0, 144.0, 165.0, 168.0,
        138.0, 110.0, 88.0, 84.0, 112.0, 140.0, 168.0, 170.0,
        138.0, 110.0, 88.0, 84.0, 112.0, 138.0
    ]
    samples = [_sample(i, fps, _make_push_up_kps(a)) for i, a in enumerate(seq)]
    res = analyzer.analyze_video_sequence(samples, fps=fps)
    assert res["validReps"] == 1
    assert res["incompleteReps"] == 1
    assert "INCOMPLETE_LOCKOUT" in res["reps"][1]["issues"]


def test_pull_up_side_view_extract_measurements_and_eye_fallback():
    analyzer = PullUpAnalyzer()
    # Unilateral left arm at 60 deg (right arm occluded) must NOT average with 180 deg!
    kps = _make_pull_up_kps(60.0, chin_over_bar=True, occlude_right_arm=True)
    m = analyzer.extract_measurements(kps, side_view=True)
    assert abs(m["elbow_angle"] - 60.0) < 2.0
    assert "elbow_symmetry_diff" not in m

    # Eye fallback when nose is occluded
    kps_eye = dict(kps)
    nose_y = kps_eye["nose"].y
    kps_eye["nose"] = _kp("nose", 210.0, nose_y, conf=0.05, visible=False)
    kps_eye["left_eye"] = _kp("left_eye", 205.0, nose_y - 3.0, conf=0.90, visible=True)
    geo = extract_pull_up_geometry(kps_eye, side_view=True)
    assert geo["chin_to_bar_clearance"] is not None
    assert geo["chin_to_bar_clearance"] > 0.0

    # Jitter around pulling boundary (140 deg) -> 0 false reps
    fps = 10.0
    jitter_seq = [170.0, 143.0, 137.0, 142.0, 138.0, 155.0, 170.0]
    res = analyzer.analyze_video_sequence(
        [_sample(i, fps, _make_pull_up_kps(a)) for i, a in enumerate(jitter_seq)],
        fps=fps,
    )
    assert res["repCount"] == 0
    assert res["incompleteReps"] == 0

