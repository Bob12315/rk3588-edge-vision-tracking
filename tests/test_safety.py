import unittest

from edge_vision.config import SafetyConfig
from edge_vision.contracts import Action, VehicleTelemetry
from edge_vision.safety import SafetyArbiter


class SafetyArbiterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.arbiter = SafetyArbiter(SafetyConfig())

    def test_critical_battery_forces_land(self) -> None:
        decision = self.arbiter.arbitrate(Action.TRACK, VehicleTelemetry(battery_percent=9))
        self.assertEqual(decision.action, Action.LAND)

    def test_low_battery_forces_rtl(self) -> None:
        decision = self.arbiter.arbitrate(Action.TRACK, VehicleTelemetry(battery_percent=18))
        self.assertEqual(decision.action, Action.RTL)

    def test_bad_localization_holds_instead_of_rtl(self) -> None:
        telemetry = VehicleTelemetry(battery_percent=18, localization_ok=False)
        decision = self.arbiter.arbitrate(Action.TRACK, telemetry)
        self.assertEqual(decision.action, Action.HOLD)

    def test_near_obstacle_holds_motion(self) -> None:
        telemetry = VehicleTelemetry(obstacle_distance_m=1.5)
        decision = self.arbiter.arbitrate(Action.SEARCH, telemetry)
        self.assertEqual(decision.action, Action.HOLD)


if __name__ == "__main__":
    unittest.main()
