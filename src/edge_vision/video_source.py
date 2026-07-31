"""OpenCV input abstraction for files, camera indexes, and network streams."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


NETWORK_PREFIXES = ("rtsp://", "rtsps://", "http://", "https://", "udp://", "tcp://")


def resolve_capture_source(value: str) -> tuple[int | str, str]:
    """Resolve a CLI source into an OpenCV source and a stable source-kind label."""

    stripped = value.strip()
    if not stripped:
        raise ValueError("video source must not be empty")
    if stripped.isdecimal():
        return int(stripped), "camera"
    if stripped.casefold().startswith(NETWORK_PREFIXES):
        return stripped, "stream"
    path = Path(stripped).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path), "file"


@dataclass(frozen=True)
class VideoMetadata:
    kind: str
    source: str
    width: int
    height: int
    fps: float
    declared_frames: int

    @property
    def duration_s(self) -> float | None:
        if self.declared_frames <= 0:
            return None
        return self.declared_frames / self.fps


class OpenCvVideoSource:
    """Synchronous decoder; deterministic for files and low-latency for live input."""

    def __init__(self, source: str, *, fallback_fps: float = 30.0) -> None:
        if fallback_fps <= 0.0:
            raise ValueError("fallback_fps must be positive")
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for video input") from exc

        capture_source, kind = resolve_capture_source(source)
        capture = cv2.VideoCapture(capture_source)
        if not capture.isOpened():
            raise RuntimeError(f"cannot open {kind} source: {source}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if kind == "file" else 0
        if width <= 0 or height <= 0:
            capture.release()
            raise RuntimeError("video source returned invalid frame dimensions")
        if fps <= 0.0:
            fps = fallback_fps

        self._cv2 = cv2
        self._capture = capture
        self._next_frame_id = 0
        self.metadata = VideoMetadata(
            kind=kind,
            source=str(capture_source),
            width=width,
            height=height,
            fps=fps,
            declared_frames=declared_frames,
        )

    def read(self) -> tuple[int, float, Any] | None:
        ok, frame = self._capture.read()
        if not ok:
            return None
        frame_id = self._next_frame_id
        self._next_frame_id += 1
        timestamp_s = frame_id / self.metadata.fps
        if self.metadata.kind == "file":
            position_ms = float(self._capture.get(self._cv2.CAP_PROP_POS_MSEC))
            if position_ms > 0.0:
                timestamp_s = position_ms / 1000.0
        return frame_id, timestamp_s, frame

    def close(self) -> None:
        self._capture.release()

    def __enter__(self) -> "OpenCvVideoSource":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
