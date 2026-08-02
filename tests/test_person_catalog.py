import unittest

try:
    import numpy as np
except ImportError:
    np = None

from edge_vision.contracts import BoundingBox, TargetObservation
from edge_vision.person_catalog import (
    PersonAttributes,
    PersonScanCollector,
    catalog_descriptions,
    person_attributes_from_mapping,
)


def attributes(**overrides):
    values = {
        "upper_color": "brown",
        "upper_type": "jacket",
        "lower_color": "black",
        "lower_type": "pants",
        "headwear": "helmet",
        "headwear_color": "white",
        "carried_object": "none",
        "carried_object_color": "unknown",
        "safety_vest": "no",
        "action": "riding",
        "upper_visibility": "clear",
        "lower_visibility": "partial",
        "head_visibility": "clear",
        "confidence": 0.88,
    }
    values.update(overrides)
    return PersonAttributes(**values)


class PersonAttributeTests(unittest.TestCase):
    def test_parses_strict_controlled_attributes(self) -> None:
        payload = attributes().to_mapping()
        parsed = person_attributes_from_mapping(payload)
        self.assertEqual(parsed.upper_color, "brown")
        self.assertEqual(parsed.action, "riding")

    def test_rejects_free_form_attribute_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported upper_color"):
            attributes(upper_color="rust colored")

    def test_recommends_shortest_distinctive_visible_phrase(self) -> None:
        descriptions = catalog_descriptions(
            [
                attributes(),
                attributes(upper_color="white", headwear="none", action="standing"),
            ]
        )
        self.assertTrue(descriptions[0]["unique_in_scan"])
        self.assertEqual(descriptions[0]["recommended_label"], "person · brown jacket")
        self.assertEqual(descriptions[1]["recommended_label"], "person · white jacket")

    def test_marks_identical_people_as_not_unique(self) -> None:
        descriptions = catalog_descriptions([attributes(), attributes()])
        self.assertFalse(descriptions[0]["unique_in_scan"])
        self.assertEqual(descriptions[0]["recommended_label"], "person")


@unittest.skipIf(np is None, "optional NumPy dependency is not installed")
class PersonScanCollectorTests(unittest.TestCase):
    def test_requires_stable_tracks_and_retains_best_crop(self) -> None:
        collector = PersonScanCollector(minimum_samples=3, maximum_candidates=2)
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        first = TargetObservation(
            label="person",
            class_id=0,
            box=BoundingBox(0.1, 0.1, 0.4, 0.8),
            detector_confidence=0.7,
            track_id=7,
        )
        better = TargetObservation(
            label="person",
            class_id=0,
            box=BoundingBox(0.08, 0.05, 0.48, 0.9),
            detector_confidence=0.9,
            track_id=7,
        )
        collector.observe(frame, [first])
        collector.observe(frame, [first])
        self.assertEqual(collector.candidates(), ())
        collector.observe(frame, [better])
        candidate = collector.candidates()[0]
        self.assertEqual(candidate.track_id, 7)
        self.assertEqual(candidate.sample_count, 3)
        self.assertEqual(candidate.best_observation, better)
        self.assertGreater(candidate.best_crop.shape[0], 80)


if __name__ == "__main__":
    unittest.main()
