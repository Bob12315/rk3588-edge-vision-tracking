import unittest

from edge_vision.contracts import BoundingBox, TargetObservation
from edge_vision.video_detection import (
    RunStatistics,
    observation_to_mapping,
    percentile,
    resolve_backend_defaults,
    resolve_tracker_settings,
)


class VideoDetectionUtilityTests(unittest.TestCase):
    def test_backend_defaults_select_yolo_world(self) -> None:
        model, confidence = resolve_backend_defaults("yolo-world", None, None)
        self.assertEqual(model, "yolov8s-worldv2.pt")
        self.assertEqual(confidence, 0.10)

    def test_backend_defaults_preserve_explicit_values(self) -> None:
        model, confidence = resolve_backend_defaults("yolo", "custom.pt", 0.4)
        self.assertEqual(model, "custom.pt")
        self.assertEqual(confidence, 0.4)

    def test_tracker_defaults_and_actual_yaml_type_are_reported(self) -> None:
        tracker, config = resolve_tracker_settings("botsort", None)
        self.assertEqual(tracker, "botsort")
        self.assertTrue(config.endswith("configs/botsort_reid.yaml"))
        tracker, config = resolve_tracker_settings(
            "bytetrack", "configs/botsort_reid.yaml"
        )
        self.assertEqual(tracker, "botsort")
        self.assertTrue(config.endswith("configs/botsort_reid.yaml"))
        self.assertEqual(resolve_tracker_settings("none", None), (None, None))

    def test_percentile_interpolates(self) -> None:
        self.assertEqual(percentile([10.0, 20.0, 30.0], 0.5), 20.0)
        self.assertAlmostEqual(percentile([10.0, 20.0], 0.95), 19.5)

    def test_statistics_include_tail_latency_and_counts(self) -> None:
        stats = RunStatistics()
        stats.record(2, 10.0)
        stats.record(4, 20.0)
        report = stats.to_mapping(wall_time_s=0.5)
        self.assertEqual(report["processed_frames"], 2)
        self.assertEqual(report["detections"]["total"], 6)
        self.assertEqual(report["performance"]["throughput_fps"], 4.0)
        self.assertAlmostEqual(report["performance"]["inference_ms"]["p95"], 19.5)

    def test_statistics_summarize_track_ids(self) -> None:
        stats = RunStatistics()
        stats.record(2, 10.0, [7, 8])
        stats.record(2, 11.0, [7, None])
        tracking = stats.tracking_mapping(enabled=True)
        self.assertTrue(tracking["enabled"])
        self.assertEqual(tracking["unique_track_ids"], 2)
        self.assertEqual(tracking["tracked_observations"], 3)
        self.assertEqual(tracking["untracked_observations"], 1)
        self.assertEqual(tracking["observations_per_track"]["max"], 2)

    def test_observation_serializes_normalized_and_pixel_boxes(self) -> None:
        observation = TargetObservation(
            label="person",
            class_id=0,
            box=BoundingBox(0.1, 0.2, 0.5, 0.8),
            detector_confidence=0.9,
            track_id=12,
        )
        record = observation_to_mapping(observation, width=1000, height=500)
        self.assertEqual(record["box_xyxy_pixels"], [100, 100, 500, 400])
        self.assertEqual(record["center_normalized"], [0.3, 0.5])
        self.assertEqual(record["track_id"], 12)
        self.assertIsNone(record["color_label"])
        self.assertIsNone(record["color_score"])


if __name__ == "__main__":
    unittest.main()
