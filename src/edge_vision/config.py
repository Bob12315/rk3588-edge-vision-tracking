"""Configuration loading without a runtime dependency on a YAML parser."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class StateMachineConfig:
    detection_threshold: float = 0.65
    tracker_threshold: float = 0.35
    lock_confirmations: int = 2
    lock_miss_limit: int = 2
    track_miss_limit: int = 3
    lost_search_limit: int = 10
    require_relation_verification: bool = True


@dataclass(frozen=True)
class SafetyConfig:
    battery_rtl_percent: float = 20.0
    battery_land_percent: float = 10.0
    minimum_obstacle_distance_m: float = 3.0


@dataclass(frozen=True)
class AppConfig:
    state_machine: StateMachineConfig
    safety: SafetyConfig

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "AppConfig":
        return cls(
            state_machine=StateMachineConfig(**data.get("state_machine", {})),
            safety=SafetyConfig(**data.get("safety", {})),
        )


def load_config(path: Path) -> AppConfig:
    with path.open("r", encoding="utf-8") as handle:
        return AppConfig.from_mapping(json.load(handle))
