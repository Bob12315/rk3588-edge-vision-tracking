import unittest

from edge_vision.contracts import BoundingBox, TargetObservation
from edge_vision.target_selection import PersistentTargetSelector, TargetSelectorConfig


def observation(
    track_id: int | None,
    confidence: float,
    *,
    color: float = 0.5,
    box: BoundingBox | None = None,
) -> TargetObservation:
    return TargetObservation(
        label="person wearing white clothes",
        box=box or BoundingBox(0.2, 0.2, 0.5, 0.7),
        detector_confidence=confidence,
        color_confidence=color,
        relation_verified=True,
        track_id=track_id,
    )


class PersistentTargetSelectorTests(unittest.TestCase):
    def test_acquires_highest_ranked_tracked_candidate(self) -> None:
        selector = PersistentTargetSelector()
        result = selector.select([observation(4, 0.4), observation(8, 0.8)])
        self.assertEqual(result.active_track_id, 8)
        self.assertEqual(result.event, "acquired")

    def test_retains_identity_when_a_stronger_candidate_appears(self) -> None:
        selector = PersistentTargetSelector()
        selector.select([observation(4, 0.7)])
        result = selector.select([observation(4, 0.2), observation(8, 0.95)])
        self.assertEqual(result.observation.track_id, 4)
        self.assertEqual(result.event, "retained")

    def test_switches_only_after_missing_grace_window(self) -> None:
        selector = PersistentTargetSelector(TargetSelectorConfig(miss_tolerance=2))
        selector.select([observation(4, 0.7)])
        self.assertIsNone(selector.select([observation(8, 0.9)]).observation)
        self.assertIsNone(selector.select([observation(8, 0.9)]).observation)
        result = selector.select([observation(8, 0.9)])
        self.assertEqual(result.observation.track_id, 8)
        self.assertEqual(result.event, "switched")

    def test_does_not_acquire_observation_without_tracker_identity(self) -> None:
        selector = PersistentTargetSelector()
        result = selector.select([observation(None, 0.99)])
        self.assertIsNone(result.observation)
        self.assertEqual(result.event, "empty")

    def test_locked_identity_never_silently_switches_to_another_person(self) -> None:
        selector = PersistentTargetSelector(
            TargetSelectorConfig(miss_tolerance=1, allow_switch=False)
        )
        selector.lock(4)
        self.assertEqual(selector.select([observation(4, 0.7)]).event, "retained")
        self.assertEqual(selector.select([observation(8, 0.9)]).event, "missing")
        result = selector.select([observation(8, 0.9)])
        self.assertIsNone(result.observation)
        self.assertEqual(result.active_track_id, 4)
        self.assertEqual(result.event, "lost")

    def test_lock_rejects_negative_track_id(self) -> None:
        selector = PersistentTargetSelector()
        with self.assertRaises(ValueError):
            selector.lock(-1)


if __name__ == "__main__":
    unittest.main()
