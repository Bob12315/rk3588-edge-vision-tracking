import json
import tempfile
import unittest
from pathlib import Path

from edge_vision.tracking_evaluation import (
    MotDetection,
    box_iou,
    evaluate_mot,
    export_mot_jsonl,
    load_mot,
    load_reviewed_frame_ids,
    maximum_weight_pairs,
    tracking_diagnostics,
)


def detection(frame_id: int, track_id: int, x: float = 0.0) -> MotDetection:
    return MotDetection(frame_id, track_id, x, 0.0, 10.0, 10.0)


class TrackingEvaluationTests(unittest.TestCase):
    def test_box_iou(self) -> None:
        self.assertEqual(box_iou(detection(1, 1), detection(1, 2)), 1.0)
        self.assertAlmostEqual(
            box_iou(detection(1, 1), detection(1, 2, x=5.0)),
            1.0 / 3.0,
        )

    def test_assignment_is_global_instead_of_greedy(self) -> None:
        pairs = maximum_weight_pairs(((0.9, 0.8), (0.85, 0.1)))
        self.assertEqual(pairs, [(0, 1), (1, 0)])

    def test_diagnostics_marks_only_spatial_switches_as_proxy(self) -> None:
        report = tracking_diagnostics(
            (detection(1, 7), detection(2, 8)),
            processed_frames=2,
        )
        proxy = report["spatial_continuity_proxy"]
        self.assertEqual(proxy["consecutive_frame_matches"], 1)
        self.assertEqual(proxy["approximate_id_switches"], 1)
        self.assertIn("Proxy only", proxy["warning"])
        self.assertEqual(report["established_track_ids"], 0)
        self.assertEqual(report["short_track_ids"], [7, 8])
        self.assertEqual(report["tracks"][0]["first_frame"], 1)

    def test_perfect_mot_predictions_score_one(self) -> None:
        ground_truth = (detection(1, 1), detection(2, 1, x=1.0))
        predictions = (detection(1, 9), detection(2, 9, x=1.0))
        report = evaluate_mot(ground_truth, predictions)
        self.assertEqual(report["counts"]["id_switches"], 0)
        self.assertEqual(report["detection"]["mota"], 1.0)
        self.assertEqual(report["identity"]["idf1"], 1.0)

    def test_identity_switch_reduces_idf1_and_mota(self) -> None:
        ground_truth = (detection(1, 1), detection(2, 1))
        predictions = (detection(1, 9), detection(2, 10))
        report = evaluate_mot(ground_truth, predictions)
        self.assertEqual(report["counts"]["id_switches"], 1)
        self.assertEqual(report["detection"]["mota"], 0.5)
        self.assertEqual(report["identity"]["idf1"], 0.5)

    def test_mot_loader_filters_unmarked_and_non_person_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gt.txt"
            path.write_text(
                "1,1,0,0,10,10,1,1,1\n"
                "1,2,20,0,10,10,0,1,1\n"
                "1,3,40,0,10,10,1,3,1\n",
                encoding="utf-8",
            )
            loaded = load_mot(path, ground_truth=True)
        self.assertEqual([item.track_id for item in loaded], [1])

    def test_jsonl_export_uses_one_based_mot_frames_and_skips_untracked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "detections.jsonl"
            output = Path(directory) / "mot.txt"
            source.write_text(
                json.dumps(
                    {
                        "frame_id": 0,
                        "detections": [
                            {
                                "track_id": 4,
                                "confidence": 0.75,
                                "box_xyxy_pixels": [10, 20, 40, 70],
                            },
                            {
                                "track_id": None,
                                "confidence": 0.5,
                                "box_xyxy_pixels": [0, 0, 5, 5],
                            },
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            summary = export_mot_jsonl(source, output)
            exported = output.read_text(encoding="utf-8")
        self.assertEqual(summary["exported_detections"], 1)
        self.assertEqual(summary["skipped_untracked_observations"], 1)
        self.assertEqual(exported, "1,4,10.000,20.000,30.000,50.000,0.750000,-1,-1,-1\n")

    def test_annotation_project_defines_explicit_reviewed_frame_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "annotations.json"
            project.write_text(
                json.dumps(
                    {
                        "video": {"frame_count": 10},
                        "reviewed_frames": [1, 4, 10],
                    }
                ),
                encoding="utf-8",
            )
            reviewed = load_reviewed_frame_ids(project)
        self.assertEqual(reviewed, {1, 4, 10})

    def test_empty_annotation_project_cannot_define_evaluation_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "annotations.json"
            project.write_text(
                json.dumps(
                    {"video": {"frame_count": 10}, "reviewed_frames": []}
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_reviewed_frame_ids(project)


if __name__ == "__main__":
    unittest.main()
