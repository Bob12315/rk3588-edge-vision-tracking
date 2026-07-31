"""Safety arbitration that has authority over perception-requested actions."""

from __future__ import annotations

from dataclasses import dataclass

from .config import SafetyConfig
from .contracts import Action, VehicleTelemetry


@dataclass(frozen=True)
class SafetyDecision:
    action: Action
    reason: str
    overridden: bool


class SafetyArbiter:
    def __init__(self, config: SafetyConfig) -> None:
        if config.battery_land_percent > config.battery_rtl_percent:
            raise ValueError("battery_land_percent must not exceed battery_rtl_percent")
        self.config = config

    def arbitrate(self, requested: Action, telemetry: VehicleTelemetry) -> SafetyDecision:
        if telemetry.battery_percent <= self.config.battery_land_percent:
            return self._override(Action.LAND, "battery at critical landing threshold", requested)
        if telemetry.operator_hold:
            return self._override(Action.HOLD, "operator hold is active", requested)
        if not telemetry.localization_ok:
            return self._override(Action.HOLD, "localization is unhealthy; RTL is unsafe", requested)
        if telemetry.battery_percent <= self.config.battery_rtl_percent:
            return self._override(Action.RTL, "battery at return-to-launch threshold", requested)
        if (
            telemetry.obstacle_distance_m is not None
            and telemetry.obstacle_distance_m < self.config.minimum_obstacle_distance_m
            and requested in {Action.SEARCH, Action.LOCK, Action.TRACK}
        ):
            return self._override(Action.HOLD, "obstacle inside minimum safety distance", requested)
        return SafetyDecision(requested, "requested action passed safety checks", False)

    @staticmethod
    def _override(action: Action, reason: str, requested: Action) -> SafetyDecision:
        return SafetyDecision(action, reason, action is not requested)
