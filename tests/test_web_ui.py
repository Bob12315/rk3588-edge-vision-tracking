import io
import unittest
from unittest.mock import patch

from edge_vision.contracts import GroundingTask, SceneObject, VlmSceneAnalysis
from edge_vision.web_ui import (
    WebUiConfig,
    _copy_limited,
    grounding_detection_prompts,
    performance_image_size,
    requested_clothing_color,
    requires_white_clothing_filter,
    safe_upload_filename,
    split_direct_prompts,
    timestamp_rate,
)


def analysis_with_attributes(*attributes: str) -> VlmSceneAnalysis:
    return VlmSceneAnalysis(
        summary="test scene",
        objects=(SceneObject("person", attributes, 1),),
        grounding=GroundingTask(
            user_query="target",
            yolo_world_prompts=("person",),
            required_attributes=attributes,
        ),
    )


class WebUiHelperTests(unittest.TestCase):
    def test_direct_prompts_accept_chinese_and_english_commas(self) -> None:
        self.assertEqual(
            split_direct_prompts("person， red car,water bottle"),
            ("person", "red car", "water bottle"),
        )

    def test_empty_direct_prompt_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            split_direct_prompts(" ， , ")

    @patch("edge_vision.web_ui.uuid.uuid4")
    def test_upload_filename_removes_path_and_unsafe_characters(self, uuid4) -> None:
        uuid4.return_value.hex = "1234567890abcdef"
        self.assertEqual(
            safe_upload_filename("../../my video (1).MP4"),
            "my_video__1-1234567890.mp4",
        )

    def test_non_video_upload_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            safe_upload_filename("payload.py")

    def test_white_clothing_plan_enables_attribute_filter(self) -> None:
        self.assertTrue(
            requires_white_clothing_filter(
                analysis_with_attributes("white clothing")
            )
        )
        self.assertFalse(
            requires_white_clothing_filter(analysis_with_attributes("red car"))
        )

    def test_general_clothing_color_is_extracted_without_colored_object_false_positive(self) -> None:
        self.assertEqual(
            requested_clothing_color(analysis_with_attributes("blue shirt")),
            "blue",
        )
        self.assertIsNone(
            requested_clothing_color(analysis_with_attributes("red car"))
        )
        self.assertEqual(requested_clothing_color(None, "穿黑色衣服的人"), "black")

    def test_clothing_attributes_are_not_sent_as_yolo_world_object_classes(self) -> None:
        self.assertEqual(
            grounding_detection_prompts(("person", "white", "pants"), "white"),
            ("person",),
        )
        self.assertEqual(
            grounding_detection_prompts(("red car",), None),
            ("red car",),
        )

    def test_limited_copy_rejects_oversized_payload(self) -> None:
        destination = io.BytesIO()
        with self.assertRaises(ValueError):
            _copy_limited(io.BytesIO(b"12345"), destination, 4)

    def test_web_config_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            WebUiConfig(confidence=1.1)
        with self.assertRaises(ValueError):
            WebUiConfig(jpeg_quality=0)
        with self.assertRaises(ValueError):
            WebUiConfig(clothing_color_threshold=1.1)
        with self.assertRaises(ValueError):
            WebUiConfig(color_minimum_dominance_ratio=-0.1)
        with self.assertRaises(ValueError):
            WebUiConfig(person_scan_frames=2, person_scan_minimum_samples=3)
        with self.assertRaises(ValueError):
            WebUiConfig(person_scan_maximum_candidates=0)

    def test_performance_modes_have_explicit_model_sizes(self) -> None:
        self.assertEqual(performance_image_size("realtime"), 384)
        self.assertEqual(performance_image_size("balanced"), 512)
        self.assertEqual(performance_image_size("quality"), 640)
        with self.assertRaises(ValueError):
            performance_image_size("turbo")

    def test_timestamp_rate_reports_end_to_end_frequency(self) -> None:
        self.assertEqual(timestamp_rate([]), 0.0)
        self.assertEqual(timestamp_rate([4.0]), 0.0)
        self.assertAlmostEqual(timestamp_rate([4.0, 4.1, 4.2]), 10.0)


if __name__ == "__main__":
    unittest.main()
