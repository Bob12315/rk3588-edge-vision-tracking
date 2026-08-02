import unittest

try:
    import cv2  # noqa: F401
    import numpy as np
except ImportError:
    np = None

from edge_vision.contracts import BoundingBox, TargetObservation
from edge_vision.filters.clothing_color import (
    ClothingColorConfig,
    ClothingColorFilter,
    SUPPORTED_COLORS,
    clothing_color_from_text,
    estimate_clothing_color,
    matching_color_score,
    normalize_color_name,
)


class StaticDetector:
    class_ids = (0,)

    def __init__(self, observation: TargetObservation) -> None:
        self.observation = observation

    def detect(self, frame, prompts=()):
        return [self.observation]


class ClothingColorTests(unittest.TestCase):
    def test_normalizes_english_and_chinese_colors(self) -> None:
        self.assertEqual(normalize_color_name("a BLUE jacket"), "blue")
        self.assertEqual(normalize_color_name("穿红色衣服的人"), "red")
        self.assertEqual(normalize_color_name("粉红色上衣"), "pink")
        self.assertIsNone(normalize_color_name("person"))

    def test_requires_clothing_context(self) -> None:
        self.assertEqual(clothing_color_from_text("person in a green shirt"), "green")
        self.assertIsNone(clothing_color_from_text("green car"))
        self.assertIsNone(clothing_color_from_text("red laptop"))

    @unittest.skipIf(np is None, "optional OpenCV/NumPy dependencies are not installed")
    def test_classifies_common_solid_colors(self) -> None:
        samples = {
            "white": (240, 240, 240),
            "black": (20, 20, 20),
            "gray": (110, 110, 110),
            "red": (0, 0, 220),
            "green": (0, 200, 0),
            "blue": (220, 0, 0),
            "yellow": (0, 220, 220),
        }
        for expected, bgr in samples.items():
            with self.subTest(expected=expected):
                roi = np.full((32, 24, 3), bgr, dtype=np.uint8)
                estimate = estimate_clothing_color(roi)
                self.assertEqual(estimate.label, expected)
                self.assertGreater(estimate.confidence, 0.95)

    @unittest.skipIf(np is None, "optional OpenCV/NumPy dependencies are not installed")
    def test_temporal_history_delays_and_stabilizes_color_match(self) -> None:
        observation = TargetObservation(
            label="person",
            class_id=0,
            box=BoundingBox(0.1, 0.1, 0.9, 0.9),
            detector_confidence=0.9,
            track_id=7,
        )
        detector = ClothingColorFilter(
            StaticDetector(observation),
            ClothingColorConfig(
                target_color="red",
                minimum_box_height_ratio=0.0,
                temporal_min_samples=3,
            ),
        )
        red = np.full((100, 100, 3), (0, 0, 220), dtype=np.uint8)
        blue = np.full((100, 100, 3), (220, 0, 0), dtype=np.uint8)
        self.assertEqual(detector.detect(red), [])
        self.assertEqual(detector.detect(red), [])
        matched = detector.detect(red)
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0].color_label, "red")
        self.assertTrue(matched[0].relation_verified)
        self.assertEqual(len(detector.detect(blue)), 1)

    @unittest.skipIf(np is None, "optional OpenCV/NumPy dependencies are not installed")
    def test_single_color_flash_on_established_track_is_not_confirmed(self) -> None:
        observation = TargetObservation(
            label="person",
            class_id=0,
            box=BoundingBox(0.1, 0.1, 0.9, 0.9),
            detector_confidence=0.9,
            track_id=9,
        )
        detector = ClothingColorFilter(
            StaticDetector(observation),
            ClothingColorConfig(
                target_color="red",
                minimum_box_height_ratio=0.0,
                temporal_alpha=1.0,
                temporal_min_samples=3,
            ),
        )
        blue = np.full((100, 100, 3), (220, 0, 0), dtype=np.uint8)
        red = np.full((100, 100, 3), (0, 0, 220), dtype=np.uint8)
        for _ in range(3):
            self.assertEqual(detector.detect(blue), [])
        self.assertEqual(detector.detect(red), [])
        self.assertEqual(detector.detect(red), [])
        self.assertEqual(len(detector.detect(red)), 1)

    def test_rejects_unknown_target_color(self) -> None:
        with self.assertRaises(ValueError):
            ClothingColorConfig(target_color="transparent")
        with self.assertRaises(ValueError):
            ClothingColorConfig(target_color="red", minimum_dominance_ratio=1.1)
        with self.assertRaises(ValueError):
            ClothingColorConfig(target_color="red", minimum_detector_confidence=-0.1)

    def test_rust_orange_can_support_brown_without_making_pure_orange_brown(self) -> None:
        rust = {color: 0.0 for color in SUPPORTED_COLORS}
        rust.update({"brown": 0.12, "orange": 0.62, "red": 0.08})
        self.assertGreater(matching_color_score(rust, "brown"), 0.60)
        self.assertGreaterEqual(
            matching_color_score(rust, "brown"),
            0.85 * matching_color_score(rust, "orange"),
        )
        pure_orange = dict(rust)
        pure_orange.update({"brown": 0.0, "orange": 1.0, "red": 0.0})
        self.assertLess(
            matching_color_score(pure_orange, "brown"),
            0.85 * matching_color_score(pure_orange, "orange"),
        )


if __name__ == "__main__":
    unittest.main()
