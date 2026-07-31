import io
import unittest
from unittest.mock import patch

from edge_vision.contracts import GroundingTask, SceneObject, VlmSceneAnalysis
from edge_vision.web_ui import (
    WebUiConfig,
    _copy_limited,
    requires_white_clothing_filter,
    safe_upload_filename,
    split_direct_prompts,
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

    def test_limited_copy_rejects_oversized_payload(self) -> None:
        destination = io.BytesIO()
        with self.assertRaises(ValueError):
            _copy_limited(io.BytesIO(b"12345"), destination, 4)

    def test_web_config_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            WebUiConfig(confidence=1.1)
        with self.assertRaises(ValueError):
            WebUiConfig(jpeg_quality=0)


if __name__ == "__main__":
    unittest.main()
