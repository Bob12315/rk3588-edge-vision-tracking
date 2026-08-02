import json
import tempfile
import unittest
from pathlib import Path

from edge_vision.identity_annotation import (
    IdentityAnnotationStore,
    annotation_from_mapping,
    load_prediction_suggestions,
)


class IdentityAnnotationTests(unittest.TestCase):
    def test_annotation_requires_valid_identity_and_video_bounds(self) -> None:
        annotation = annotation_from_mapping(
            {
                "identity_id": 3,
                "x": 10,
                "y": 20,
                "width": 30,
                "height": 40,
                "visibility": 0.75,
            },
            width=100,
            height=100,
        )
        self.assertEqual(annotation.identity_id, 3)
        self.assertEqual(annotation.visibility, 0.75)
        with self.assertRaises(ValueError):
            annotation_from_mapping(
                {"identity_id": 0, "x": 0, "y": 0, "width": 5, "height": 5},
                width=100,
                height=100,
            )
        with self.assertRaises(ValueError):
            annotation_from_mapping(
                {"identity_id": 1, "x": 90, "y": 0, "width": 20, "height": 5},
                width=100,
                height=100,
            )

    def test_reviewed_frames_persist_to_json_and_mot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "video.mp4"
            source.write_bytes(b"test")
            output = root / "annotations"
            store = IdentityAnnotationStore(
                output,
                source=source,
                frame_count=3,
                width=100,
                height=80,
                fps=30.0,
            )
            store.update_frame(
                1,
                [
                    {
                        "identity_id": 7,
                        "x": 10,
                        "y": 20,
                        "width": 30,
                        "height": 40,
                        "visibility": 0.5,
                    }
                ],
            )
            store.update_frame(2, [])
            reloaded = IdentityAnnotationStore(
                output,
                source=source,
                frame_count=3,
                width=100,
                height=80,
                fps=30.0,
            )
            project = reloaded.project()
            ground_truth = reloaded.ground_truth_path.read_text(encoding="utf-8")

        self.assertEqual(project["reviewed_frames"], [1, 2])
        self.assertEqual(project["reviewed_frame_count"], 2)
        self.assertEqual(project["annotation_count"], 1)
        self.assertTrue(reloaded.frame(2)["reviewed"])
        self.assertEqual(
            ground_truth,
            "1,7,10.000,20.000,30.000,40.000,1,1,0.500000\n",
        )

    def test_duplicate_identity_in_one_frame_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "video.mp4"
            source.write_bytes(b"test")
            store = IdentityAnnotationStore(
                root / "annotations",
                source=source,
                frame_count=1,
                width=100,
                height=80,
                fps=30.0,
            )
            box = {"identity_id": 1, "x": 0, "y": 0, "width": 10, "height": 10}
            with self.assertRaises(ValueError):
                store.update_frame(1, [box, {**box, "x": 20}])

    def test_predictions_remain_separate_unreviewed_suggestions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "detections.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "frame_id": 0,
                        "detections": [
                            {
                                "track_id": 4,
                                "confidence": 0.8,
                                "box_xyxy_pixels": [10, 10, 40, 50],
                            },
                            {
                                "track_id": 8,
                                "confidence": 0.2,
                                "box_xyxy_pixels": [95, 10, 110, 30],
                            },
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            suggestions = load_prediction_suggestions(
                path,
                frame_count=1,
                width=100,
                height=80,
            )
        self.assertEqual(len(suggestions[1]), 1)
        self.assertEqual(suggestions[1][0]["suggested_identity_id"], 4)
        self.assertEqual(suggestions[1][0]["source"], "model_prediction")


if __name__ == "__main__":
    unittest.main()
