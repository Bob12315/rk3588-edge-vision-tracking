import unittest

from edge_vision.config import StateMachineConfig
from edge_vision.contracts import BoundingBox, MissionState, PerceptionSnapshot, TargetObservation
from edge_vision.state_machine import TrackingStateMachine


def snapshot(frame_id: int, detector: float, tracker=None, verified=True):
    return PerceptionSnapshot(
        frame_id=frame_id,
        timestamp_s=frame_id / 10,
        target=TargetObservation(
            label="target",
            box=BoundingBox(0.2, 0.2, 0.5, 0.6),
            detector_confidence=detector,
            tracker_confidence=tracker,
            relation_verified=verified,
        ),
    )


class TrackingStateMachineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.machine = TrackingStateMachine(StateMachineConfig())

    def test_candidate_requires_two_verified_lock_frames(self) -> None:
        self.assertEqual(self.machine.step(snapshot(0, 0.9)).state, MissionState.LOCK)
        self.assertEqual(self.machine.step(snapshot(1, 0.9)).state, MissionState.LOCK)
        self.assertEqual(self.machine.step(snapshot(2, 0.9, tracker=0.8)).state, MissionState.TRACK)

    def test_unverified_relation_does_not_lock(self) -> None:
        self.machine.step(snapshot(0, 0.9, verified=False))
        transition = self.machine.step(snapshot(1, 0.9, verified=False))
        self.assertEqual(transition.state, MissionState.LOCK)

    def test_tracker_loss_enters_lost_then_reacquires(self) -> None:
        self.machine.step(snapshot(0, 0.9))
        self.machine.step(snapshot(1, 0.9))
        self.machine.step(snapshot(2, 0.9, tracker=0.8))
        empty = PerceptionSnapshot(3, 0.3, None)
        self.machine.step(empty)
        self.machine.step(empty)
        self.assertEqual(self.machine.step(empty).state, MissionState.LOST)
        self.assertEqual(self.machine.step(snapshot(4, 0.9)).state, MissionState.LOCK)


if __name__ == "__main__":
    unittest.main()
