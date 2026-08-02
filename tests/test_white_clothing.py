import unittest

from edge_vision.contracts import BoundingBox, TargetObservation
from edge_vision.filters.white_clothing import WhiteClothingConfig, torso_roi_bounds


class WhiteClothingFilterTests(unittest.TestCase):
    def test_torso_roi_is_inside_person_box(self) -> None:
        observation = TargetObservation(
            label="person",
            class_id=0,
            box=BoundingBox(0.1, 0.2, 0.5, 0.8),
            detector_confidence=0.9,
        )
        bounds = torso_roi_bounds(
            observation,
            width=1000,
            height=500,
            config=WhiteClothingConfig(),
        )
        self.assertEqual(bounds, (172, 154, 428, 265))

    def test_invalid_white_ratio_threshold_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            WhiteClothingConfig(ratio_threshold=1.1)

    def test_invalid_minimum_box_height_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            WhiteClothingConfig(minimum_box_height_ratio=-0.1)


if __name__ == "__main__":
    unittest.main()
