import unittest

from edge_vision.contracts import BoundingBox, TargetObservation
from edge_vision.video_detection import RunStatistics, observation_to_mapping, percentile


class VideoDetectionUtilityTests(unittest.TestCase):
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

    def test_observation_serializes_normalized_and_pixel_boxes(self) -> None:
        observation = TargetObservation(
            label="person",
            class_id=0,
            box=BoundingBox(0.1, 0.2, 0.5, 0.8),
            detector_confidence=0.9,
        )
        record = observation_to_mapping(observation, width=1000, height=500)
        self.assertEqual(record["box_xyxy_pixels"], [100, 100, 500, 400])
        self.assertEqual(record["center_normalized"], [0.3, 0.5])


if __name__ == "__main__":
    unittest.main()
