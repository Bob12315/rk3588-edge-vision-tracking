"""Mission orchestration independent of concrete model and autopilot SDKs."""

from __future__ import annotations

from .contracts import Action, Decision, PerceptionSnapshot, VehicleTelemetry
from .safety import SafetyArbiter
from .state_machine import TrackingStateMachine


class MissionPipeline:
    def __init__(self, state_machine: TrackingStateMachine, safety: SafetyArbiter) -> None:
        self.state_machine = state_machine
        self.safety = safety

    def process(
        self, snapshot: PerceptionSnapshot, telemetry: VehicleTelemetry
    ) -> Decision:
        transition = self.state_machine.step(snapshot)
        safe = self.safety.arbitrate(transition.action, telemetry)
        if safe.action in {Action.RTL, Action.LAND}:
            self.state_machine.force_rtl_land()
        reason = safe.reason if safe.overridden else transition.reason
        return Decision(
            state=self.state_machine.state,
            requested_action=transition.action,
            safe_action=safe.action,
            reason=reason,
            transition_reason=transition.reason,
            safety_reason=safe.reason,
            safety_overridden=safe.overridden,
        )
