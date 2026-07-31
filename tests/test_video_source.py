import tempfile
import unittest
from pathlib import Path

from edge_vision.video_source import resolve_capture_source


class ResolveCaptureSourceTests(unittest.TestCase):
    def test_numeric_source_is_camera_index(self) -> None:
        self.assertEqual(resolve_capture_source(" 2 "), (2, "camera"))

    def test_rtsp_source_is_network_stream(self) -> None:
        value = "rtsp://127.0.0.1/live"
        self.assertEqual(resolve_capture_source(value), (value, "stream"))

    def test_existing_file_is_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.mp4"
            path.touch()
            resolved, kind = resolve_capture_source(str(path))
        self.assertEqual(resolved, str(path.resolve()))
        self.assertEqual(kind, "file")

    def test_missing_file_is_rejected(self) -> None:
        with self.assertRaises(FileNotFoundError):
            resolve_capture_source("/definitely/missing/video.mp4")


if __name__ == "__main__":
    unittest.main()
