"""Stable contracts shared by perception, mission logic, and flight adapters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Protocol, Sequence


class MissionState(str, Enum):
    SEARCH = "SEARCH"
    LOCK = "LOCK"
    TRACK = "TRACK"
    LOST = "LOST"
    HOLD = "HOLD"
    RTL_LAND = "RTL_LAND"


class Action(str, Enum):
    SEARCH = "SEARCH"
    LOCK = "LOCK"
    TRACK = "TRACK"
    HOLD = "HOLD"
    RTL = "RTL"
    LAND = "LAND"


@dataclass(frozen=True)
class BoundingBox:
    """Normalized xyxy bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("bounding-box coordinates must be normalized to [0, 1]")
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError("bounding box must have positive area")

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass(frozen=True)
class TargetObservation:
    label: str
    box: BoundingBox
    detector_confidence: float
    tracker_confidence: Optional[float] = None
    relation_verified: bool = False
    color_confidence: Optional[float] = None
    distance_m: Optional[float] = None
    track_id: Optional[int] = None
    class_id: Optional[int] = None

    def __post_init__(self) -> None:
        confidence_values = (
            self.detector_confidence,
            self.tracker_confidence,
            self.color_confidence,
        )
        if any(value is not None and not 0.0 <= value <= 1.0 for value in confidence_values):
            raise ValueError("confidence values must be in [0, 1]")


@dataclass(frozen=True)
class SceneObject:
    """Object category and optional attributes reported by a VLM."""

    name: str
    attributes: tuple[str, ...] = ()
    count: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("scene-object name must not be empty")
        if self.count is not None and self.count < 0:
            raise ValueError("scene-object count must be non-negative")


@dataclass(frozen=True)
class GroundingTask:
    """Structured target description passed from a VLM to a grounding detector."""

    user_query: str
    yolo_world_prompts: tuple[str, ...]
    required_attributes: tuple[str, ...] = ()
    relation: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.user_query.strip():
            raise ValueError("grounding user_query must not be empty")
        if not self.yolo_world_prompts or any(
            not prompt.strip() for prompt in self.yolo_world_prompts
        ):
            raise ValueError("grounding requires at least one non-empty YOLO-World prompt")


@dataclass(frozen=True)
class VlmSceneAnalysis:
    """Provider-neutral, structured output from one low-frequency VLM call."""

    summary: str
    objects: tuple[SceneObject, ...]
    grounding: GroundingTask

    def __post_init__(self) -> None:
        if not self.summary.strip():
            raise ValueError("VLM scene summary must not be empty")


@dataclass(frozen=True)
class PerceptionSnapshot:
    frame_id: int
    timestamp_s: float
    target: Optional[TargetObservation] = None


@dataclass(frozen=True)
class VehicleTelemetry:
    battery_percent: float = 100.0
    localization_ok: bool = True
    obstacle_distance_m: Optional[float] = None
    operator_hold: bool = False


@dataclass(frozen=True)
class Decision:
    state: MissionState
    requested_action: Action
    safe_action: Action
    reason: str


class Detector(Protocol):
    def detect(
        self, frame: Any, prompts: Sequence[str] = ()
    ) -> Sequence[TargetObservation]: ...


class VisionLanguageModel(Protocol):
    """Low-frequency scene understanding; never controls the vehicle directly."""

    def analyze(self, frame: Any, instruction: str) -> VlmSceneAnalysis: ...


class Tracker(Protocol):
    def initialize(self, frame: Any, target: TargetObservation) -> None: ...

    def update(self, frame: Any) -> Optional[TargetObservation]: ...


class FlightGateway(Protocol):
    """Only high-level, safety-reviewed actions cross this boundary."""

    def execute(self, action: Action, context: Mapping[str, Any]) -> None: ...
