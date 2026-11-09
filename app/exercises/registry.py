import logging
from typing import Any, Dict, List, Optional
from app.exercises.base import BaseExerciseAnalyzer
from app.exercises.dips import DipsAnalyzer
from app.exercises.front_lever import FrontLeverAnalyzer
from app.exercises.handstand import HandstandAnalyzer
from app.exercises.l_sit import LSitAnalyzer
from app.exercises.muscle_up import MuscleUpAnalyzer
from app.exercises.pike_push_up import PikePushUpAnalyzer
from app.exercises.planche import PlancheAnalyzer
from app.exercises.pull_up import PullUpAnalyzer
from app.exercises.push_up import PushUpAnalyzer
from app.exercises.squat import SquatAnalyzer

logger = logging.getLogger(__name__)


class ExerciseRegistry:
    """Registry managing exercise analyzers for real-time live evaluation."""

    def __init__(self):
        self._analyzers: Dict[str, BaseExerciseAnalyzer] = {}
        self._alias_map: Dict[str, BaseExerciseAnalyzer] = {}
        self._register_defaults()

    def register(self, analyzer: BaseExerciseAnalyzer, aliases: Optional[List[str]] = None) -> None:
        """Register an exercise analyzer under its primary name and aliases."""
        key = analyzer.name.upper()
        self._analyzers[key] = analyzer
        self._alias_map[self._normalize(key)] = analyzer

        if aliases:
            for alias in aliases:
                self._alias_map[self._normalize(alias)] = analyzer

    def _normalize(self, name: str) -> str:
        """Normalize exercise string for robust matching."""
        return str(name).strip().upper().replace(" ", "_").replace("-", "_")

    def _register_defaults(self) -> None:
        # 1. Planche
        planche = PlancheAnalyzer()
        self.register(
            planche,
            [
                "PLANCHE", "FULL_PLANCHE", "STRADDLE_PLANCHE",
                "ADV_TUCK_PLANCHE", "ADVANCED_TUCK_PLANCHE", "TUCK_PLANCHE"
            ]
        )

        # 2. Front Lever
        front_lever = FrontLeverAnalyzer()
        self.register(
            front_lever,
            [
                "FRONT_LEVER", "FULL_FRONT_LEVER", "STRADDLE_FRONT_LEVER",
                "ADV_TUCK_FRONT_LEVER", "ADVANCED_TUCK_FRONT_LEVER", "TUCK_FRONT_LEVER"
            ]
        )

        # 3. Handstand & HSPU
        handstand = HandstandAnalyzer()
        self.register(
            handstand,
            [
                "HANDSTAND", "HANDSTAND_PUSHUP", "HSPU",
                "HANDSTAND_PUSH_UP"
            ]
        )

        # 4. Pull-Up
        pull_up = PullUpAnalyzer()
        self.register(
            pull_up,
            [
                "PULL_UP", "PULLUP", "CHIN_UP", "CHINUP"
            ]
        )

        # 5. Dips
        dips = DipsAnalyzer()
        self.register(
            dips,
            [
                "DIPS", "DIP", "PARALLEL_BAR_DIPS"
            ]
        )

        # 6. Muscle-Up
        muscle_up = MuscleUpAnalyzer()
        self.register(
            muscle_up,
            [
                "MUSCLE_UP", "MUSCLEUP"
            ]
        )

        # 7. Push-Up
        push_up = PushUpAnalyzer()
        self.register(
            push_up,
            [
                "PUSH_UP", "PUSHUP", "PRESS_UP", "STANDARD_PUSH_UP"
            ]
        )

        # 8. Squat
        squat = SquatAnalyzer()
        self.register(
            squat,
            [
                "SQUAT", "SQUATS", "BODYWEIGHT_SQUAT", "AIR_SQUAT"
            ]
        )

        # 9. Pike Push-Up
        pike = PikePushUpAnalyzer()
        self.register(
            pike,
            [
                "PIKE_PUSH_UP", "PIKE_PUSHUP", "PIKE_PRESS"
            ]
        )

        # 10. L-Sit
        l_sit = LSitAnalyzer()
        self.register(
            l_sit,
            [
                "L_SIT", "LSIT", "L_SIT_HOLD", "TUCK_L_SIT"
            ]
        )

    def get_analyzer(self, exercise_name: Optional[str]) -> BaseExerciseAnalyzer:
        """
        Resolve an exercise analyzer by name or code.
        Defaults to PlancheAnalyzer if not matched, or raises ValueError if strictly unknown.
        """
        if not exercise_name:
            return self._analyzers["PLANCHE"]

        clean = self._normalize(exercise_name)

        # Direct match in alias map
        if clean in self._alias_map:
            return self._alias_map[clean]

        # Substring heuristics
        if "PIKE" in clean:
            return self._alias_map["PIKE_PUSH_UP"]
        elif "PUSH_UP" in clean or "PUSHUP" in clean or "PRESS_UP" in clean:
            return self._alias_map["PUSH_UP"]
        elif "SQUAT" in clean:
            return self._alias_map["SQUAT"]
        elif "L_SIT" in clean or "LSIT" in clean:
            return self._alias_map["L_SIT"]
        elif "FRONT_LEVER" in clean or "FRONTLEVER" in clean:
            return self._alias_map["FRONT_LEVER"]
        elif "HANDSTAND" in clean or "HSPU" in clean:
            return self._alias_map["HANDSTAND"]
        elif "PULL_UP" in clean or "PULLUP" in clean or "CHIN" in clean:
            return self._alias_map["PULL_UP"]
        elif "DIP" in clean:
            return self._alias_map["DIPS"]
        elif "MUSCLE" in clean:
            return self._alias_map["MUSCLE_UP"]
        elif "PLANCHE" in clean:
            return self._alias_map["PLANCHE"]

        logger.warning("Unrecognized exercise '%s', defaulting to PlancheAnalyzer", exercise_name)
        return self._analyzers["PLANCHE"]

    def is_supported(self, exercise_name: Optional[str]) -> bool:
        if not exercise_name:
            return False
        clean = self._normalize(exercise_name)
        return clean in self._alias_map or any(
            k in clean for k in [
                "PLANCHE", "FRONT_LEVER", "HANDSTAND", "PULL_UP", "DIP",
                "MUSCLE", "PUSH_UP", "PUSHUP", "SQUAT", "PIKE", "L_SIT", "LSIT"
            ]
        )

    def list_supported_exercises(self) -> List[Dict[str, Any]]:
        return [
            {
                "exercise": "PUSH_UP",
                "name": "Push-up",
                "type": "REPETITION",
                "variations": ["PUSH_UP"],
                "camera_view": "Side-on (45° or 90°) view at chest height"
            },
            {
                "exercise": "SQUAT",
                "name": "Squat",
                "type": "REPETITION",
                "variations": ["SQUAT"],
                "camera_view": "Side or 45-degree angle showing full body"
            },
            {
                "exercise": "PULL_UP",
                "name": "Pull-Up",
                "type": "REPETITION",
                "variations": ["PULL_UP", "CHIN_UP"],
                "camera_view": "Front or side view"
            },
            {
                "exercise": "DIPS",
                "name": "Parallel Bar Dips",
                "type": "REPETITION",
                "variations": ["DIPS"],
                "camera_view": "Side-on view"
            },
            {
                "exercise": "HANDSTAND",
                "name": "Handstand & Handstand Push-up",
                "type": "HOLD_AND_REPETITION",
                "variations": ["HANDSTAND", "HANDSTAND_PUSHUP"],
                "camera_view": "Side or front-diagonal view"
            },
            {
                "exercise": "PLANCHE",
                "name": "Planche",
                "type": "ISOMETRIC_HOLD",
                "variations": ["TUCK_PLANCHE", "ADV_TUCK_PLANCHE", "STRADDLE_PLANCHE", "FULL_PLANCHE"],
                "camera_view": "Side-on (profile) view recommended"
            },
            {
                "exercise": "FRONT_LEVER",
                "name": "Front Lever",
                "type": "ISOMETRIC_HOLD",
                "variations": ["TUCK_FRONT_LEVER", "ADV_TUCK_FRONT_LEVER", "STRADDLE_FRONT_LEVER", "FULL_FRONT_LEVER"],
                "camera_view": "Side-on (profile) view recommended"
            },
            {
                "exercise": "PIKE_PUSH_UP",
                "name": "Pike Push-up",
                "type": "REPETITION",
                "variations": ["PIKE_PUSH_UP"],
                "camera_view": "Side-on view showing hips, shoulders, and wrists"
            },
            {
                "exercise": "L_SIT",
                "name": "L-sit",
                "type": "ISOMETRIC_HOLD",
                "variations": ["L_SIT", "TUCK_L_SIT"],
                "camera_view": "Side-on (profile) view at floor height"
            },
            {
                "exercise": "MUSCLE_UP",
                "name": "Muscle-Up",
                "type": "REPETITION",
                "variations": ["MUSCLE_UP"],
                "camera_view": "Front or side view"
            }
        ]


# Global singleton registry
exercise_registry = ExerciseRegistry()
