import unittest

from edge_vision.config import StateMachineConfig
from edge_vision.contracts import BoundingBox, MissionState, PerceptionSnapshot, TargetObservation
from edge_vision.state_machine import TrackingStateMachine


def snapshot(frame_id: int, detector: float, tracker=None, verified=True, track_id=None):
    return PerceptionSnapshot(
        frame_id=frame_id,
        timestamp_s=frame_id / 10,
        target=TargetObservation(
            label="target",
            box=BoundingBox(0.2, 0.2, 0.5, 0.6),
            detector_confidence=detector,
            tracker_confidence=tracker,
            relation_verified=verified,
            track_id=track_id,
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

    def test_bytetrack_id_uses_detector_score_as_health_signal(self) -> None:
        self.machine.step(snapshot(0, 0.9))
        self.machine.step(snapshot(1, 0.9))
        self.assertEqual(
            self.machine.step(snapshot(2, 0.9, track_id=17)).state,
            MissionState.TRACK,
        )
        transition = self.machine.step(snapshot(3, 0.8, track_id=17))
        self.assertEqual(transition.state, MissionState.TRACK)
        self.assertIn("track ID", transition.reason)

    def test_untracked_detection_is_not_healthy_in_track_state(self) -> None:
        self.machine.step(snapshot(0, 0.9))
        self.machine.step(snapshot(1, 0.9))
        self.machine.step(snapshot(2, 0.9, track_id=17))
        self.machine.step(snapshot(3, 0.9))
        self.machine.step(snapshot(4, 0.9))
        self.assertEqual(self.machine.step(snapshot(5, 0.9)).state, MissionState.LOST)


if __name__ == "__main__":
    unittest.main()
