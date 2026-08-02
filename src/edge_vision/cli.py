"""Small deterministic simulator for validating orchestration before model integration."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .config import load_config
from .contracts import BoundingBox, PerceptionSnapshot, TargetObservation, VehicleTelemetry
from .pipeline import MissionPipeline
from .safety import SafetyArbiter
from .state_machine import TrackingStateMachine


def _target(detector: float, tracker: float | None, verified: bool = True) -> TargetObservation:
    return TargetObservation(
        label="person carrying blue box",
        box=BoundingBox(0.35, 0.2, 0.65, 0.85),
        detector_confidence=detector,
        tracker_confidence=tracker,
        relation_verified=verified,
        distance_m=12.0,
        track_id=1,
    )


def _scenario(name: str) -> Iterable[PerceptionSnapshot]:
    if name == "nominal":
        targets = [None, _target(0.82, None), _target(0.88, 0.70), _target(0.91, 0.76)]
    elif name == "lost":
        targets = [
            _target(0.82, None),
            _target(0.88, 0.70),
            _target(0.91, 0.76),
            None,
            None,
            None,
            _target(0.86, None),
            _target(0.90, 0.72),
            _target(0.92, 0.80),
        ]
    else:
        raise ValueError(f"unknown scenario: {name}")
    for frame_id, target in enumerate(targets):
        yield PerceptionSnapshot(frame_id, frame_id / 10.0, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    parser.add_argument("--scenario", choices=("nominal", "lost"), default="nominal")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = MissionPipeline(
        TrackingStateMachine(config.state_machine),
        SafetyArbiter(config.safety),
    )
    telemetry = VehicleTelemetry()
    for snapshot in _scenario(args.scenario):
        decision = pipeline.process(snapshot, telemetry)
        record = {"frame_id": snapshot.frame_id, **asdict(decision)}
        print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
