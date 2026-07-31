import unittest

from edge_vision.config import SafetyConfig, StateMachineConfig
from edge_vision.contracts import Action, PerceptionSnapshot, VehicleTelemetry
from edge_vision.pipeline import MissionPipeline
from edge_vision.safety import SafetyArbiter
from edge_vision.state_machine import TrackingStateMachine


class PipelineTests(unittest.TestCase):
    def test_safety_override_is_reflected_in_decision(self) -> None:
        pipeline = MissionPipeline(
            TrackingStateMachine(StateMachineConfig()),
            SafetyArbiter(SafetyConfig()),
        )
        decision = pipeline.process(
            PerceptionSnapshot(0, 0.0),
            VehicleTelemetry(operator_hold=True),
        )
        self.assertEqual(decision.requested_action, Action.SEARCH)
        self.assertEqual(decision.safe_action, Action.HOLD)


if __name__ == "__main__":
    unittest.main()
