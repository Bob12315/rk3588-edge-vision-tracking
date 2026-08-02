"""Persistent single-target selection over multi-object tracker observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .contracts import TargetObservation


@dataclass(frozen=True)
class TargetSelectorConfig:
    """Weights for acquiring one target and grace time before an ID switch."""

    miss_tolerance: int = 5
    detector_weight: float = 0.50
    attribute_weight: float = 0.30
    area_weight: float = 0.15
    center_weight: float = 0.05
    allow_switch: bool = True

    def __post_init__(self) -> None:
        if self.miss_tolerance < 0:
            raise ValueError("miss_tolerance must be non-negative")
        weights = (
            self.detector_weight,
            self.attribute_weight,
            self.area_weight,
            self.center_weight,
        )
        if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
            raise ValueError("selector weights must be non-negative with a positive sum")


@dataclass(frozen=True)
class SelectionResult:
    observation: TargetObservation | None
    active_track_id: int | None
    event: str
    reason: str


class PersistentTargetSelector:
    """Keep one tracker identity and only switch after a configurable absence."""

    def __init__(self, config: TargetSelectorConfig | None = None) -> None:
        self.config = config or TargetSelectorConfig()
        self.active_track_id: int | None = None
        self.missed_frames = 0

    def reset(self) -> None:
        self.active_track_id = None
        self.missed_frames = 0

    def lock(self, track_id: int) -> None:
        """Pin a detector-provided identity before the next tracking frame."""

        if track_id < 0:
            raise ValueError("track_id must be non-negative")
        self.active_track_id = track_id
        self.missed_frames = 0

    def select(self, candidates: Sequence[TargetObservation]) -> SelectionResult:
        if self.active_track_id is not None:
            retained = next(
                (item for item in candidates if item.track_id == self.active_track_id),
                None,
            )
            if retained is not None:
                self.missed_frames = 0
                return SelectionResult(
                    retained,
                    self.active_track_id,
                    "retained",
                    "retained active tracker identity",
                )

            self.missed_frames += 1
            if self.missed_frames <= self.config.miss_tolerance:
                return SelectionResult(
                    None,
                    self.active_track_id,
                    "missing",
                    "active identity absent inside switch grace window",
                )
            if not self.config.allow_switch:
                return SelectionResult(
                    None,
                    self.active_track_id,
                    "lost",
                    "locked tracker identity is lost; automatic switching is disabled",
                )

        tracked = [item for item in candidates if item.track_id is not None]
        if not tracked:
            event = "missing" if self.active_track_id is not None else "empty"
            return SelectionResult(
                None,
                self.active_track_id,
                event,
                "no tracked candidate is available",
            )

        previous_track_id = self.active_track_id
        selected = max(tracked, key=self._rank)
        self.active_track_id = selected.track_id
        self.missed_frames = 0
        if previous_track_id is None:
            event = "acquired"
            reason = "acquired highest-ranked tracked candidate"
        else:
            event = "switched"
            reason = "previous identity exceeded miss tolerance; selected a new track"
        return SelectionResult(selected, self.active_track_id, event, reason)

    def _rank(self, observation: TargetObservation) -> tuple[float, float, int]:
        box = observation.box
        area = (box.x2 - box.x1) * (box.y2 - box.y1)
        center_x, center_y = box.center
        center_score = max(0.0, 1.0 - ((center_x - 0.5) ** 2 + (center_y - 0.5) ** 2) ** 0.5)
        attribute_score = observation.color_confidence or 0.0
        score = (
            self.config.detector_weight * observation.detector_confidence
            + self.config.attribute_weight * attribute_score
            + self.config.area_weight * min(1.0, area * 4.0)
            + self.config.center_weight * center_score
        )
        # Lower IDs win exact score ties so replay is deterministic.
        return score, observation.detector_confidence, -(observation.track_id or 0)
