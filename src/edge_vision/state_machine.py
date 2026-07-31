"""Deterministic mission state transitions for target acquisition and tracking."""

from __future__ import annotations

from dataclasses import dataclass

from .config import StateMachineConfig
from .contracts import Action, MissionState, PerceptionSnapshot, TargetObservation


@dataclass(frozen=True)
class Transition:
    state: MissionState
    action: Action
    reason: str


class TrackingStateMachine:
    def __init__(self, config: StateMachineConfig) -> None:
        self.config = config
        self.state = MissionState.SEARCH
        self._lock_hits = 0
        self._lock_misses = 0
        self._track_misses = 0
        self._lost_frames = 0

    def reset(self) -> None:
        self.state = MissionState.SEARCH
        self._reset_counters()

    def force_rtl_land(self) -> None:
        self.state = MissionState.RTL_LAND
        self._reset_counters()

    def step(self, snapshot: PerceptionSnapshot) -> Transition:
        if self.state is MissionState.RTL_LAND:
            return self._result(Action.HOLD, "terminal flight-safety state")
        if self.state is MissionState.HOLD:
            return self._result(Action.HOLD, "manual review or explicit reset required")

        target = snapshot.target
        if self.state is MissionState.SEARCH:
            return self._search(target)
        if self.state is MissionState.LOCK:
            return self._lock(target)
        if self.state is MissionState.TRACK:
            return self._track(target)
        return self._lost(target)

    def _search(self, target: TargetObservation | None) -> Transition:
        if self._detectable(target):
            self.state = MissionState.LOCK
            self._lock_hits = 0
            self._lock_misses = 0
            return self._result(Action.LOCK, "candidate detected; starting identity confirmation")
        return self._result(Action.SEARCH, "no candidate above detection threshold")

    def _lock(self, target: TargetObservation | None) -> Transition:
        if not self._detectable(target):
            self._lock_misses += 1
            if self._lock_misses >= self.config.lock_miss_limit:
                self.state = MissionState.SEARCH
                self._lock_hits = 0
                self._lock_misses = 0
                return self._result(Action.SEARCH, "candidate disappeared during lock")
            return self._result(Action.LOCK, "waiting for candidate to reappear")

        self._lock_misses = 0
        verified = target is not None and (
            target.relation_verified or not self.config.require_relation_verification
        )
        if not verified:
            self._lock_hits = 0
            return self._result(Action.LOCK, "candidate requires semantic or relation verification")

        self._lock_hits += 1
        if self._lock_hits >= self.config.lock_confirmations:
            self.state = MissionState.TRACK
            self._track_misses = 0
            return self._result(Action.TRACK, "target identity confirmed")
        return self._result(Action.LOCK, "collecting consecutive lock confirmations")

    def _track(self, target: TargetObservation | None) -> Transition:
        tracker_ok = (
            target is not None
            and target.tracker_confidence is not None
            and target.tracker_confidence >= self.config.tracker_threshold
        )
        if tracker_ok:
            self._track_misses = 0
            return self._result(Action.TRACK, "tracker confidence healthy")

        self._track_misses += 1
        if self._track_misses >= self.config.track_miss_limit:
            self.state = MissionState.LOST
            self._lost_frames = 0
            return self._result(Action.SEARCH, "tracker confidence lost; starting reacquisition")
        return self._result(Action.TRACK, "tracker degraded; awaiting periodic redetection")

    def _lost(self, target: TargetObservation | None) -> Transition:
        if self._detectable(target):
            self.state = MissionState.LOCK
            self._lock_hits = 0
            self._lock_misses = 0
            return self._result(Action.LOCK, "candidate reacquired; revalidating identity")

        self._lost_frames += 1
        if self._lost_frames >= self.config.lost_search_limit:
            self.state = MissionState.HOLD
            return self._result(Action.HOLD, "reacquisition timed out")
        return self._result(Action.SEARCH, "expanding search after target loss")

    def _detectable(self, target: TargetObservation | None) -> bool:
        return target is not None and target.detector_confidence >= self.config.detection_threshold

    def _result(self, action: Action, reason: str) -> Transition:
        return Transition(state=self.state, action=action, reason=reason)

    def _reset_counters(self) -> None:
        self._lock_hits = 0
        self._lock_misses = 0
        self._track_misses = 0
        self._lost_frames = 0
