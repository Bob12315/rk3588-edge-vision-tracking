"""Run closed-set object detection over a video and export reproducible evidence."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .contracts import TargetObservation


class FrameDetector(Protocol):
    def detect(self, frame: Any, prompts: Sequence[str] = ()) -> Sequence[TargetObservation]: ...


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass
class RunStatistics:
    detection_counts: list[int] = field(default_factory=list)
    inference_ms: list[float] = field(default_factory=list)

    def record(self, detection_count: int, inference_ms: float) -> None:
        self.detection_counts.append(detection_count)
        self.inference_ms.append(inference_ms)

    def to_mapping(self, wall_time_s: float) -> Mapping[str, Any]:
        frame_count = len(self.detection_counts)
        total_detections = sum(self.detection_counts)
        return {
            "processed_frames": frame_count,
            "detections": {
                "total": total_detections,
                "per_frame_min": min(self.detection_counts, default=0),
                "per_frame_max": max(self.detection_counts, default=0),
                "per_frame_mean": total_detections / frame_count if frame_count else 0.0,
            },
            "performance": {
                "wall_time_s": wall_time_s,
                "throughput_fps": frame_count / wall_time_s if wall_time_s else 0.0,
                "inference_ms": {
                    "mean": statistics.fmean(self.inference_ms) if self.inference_ms else 0.0,
                    "p50": percentile(self.inference_ms, 0.50),
                    "p95": percentile(self.inference_ms, 0.95),
                    "p99": percentile(self.inference_ms, 0.99),
                },
            },
        }


def observation_to_mapping(observation: TargetObservation, width: int, height: int) -> dict:
    box = observation.box
    return {
        "class_id": observation.class_id,
        "label": observation.label,
        "confidence": observation.detector_confidence,
        "box_xyxy_normalized": [box.x1, box.y1, box.x2, box.y2],
        "box_xyxy_pixels": [
            round(box.x1 * width),
            round(box.y1 * height),
            round(box.x2 * width),
            round(box.y2 * height),
        ],
        "center_normalized": list(box.center),
    }


def _draw_observations(frame: Any, observations: Sequence[TargetObservation], cv2: Any) -> Any:
    height, width = frame.shape[:2]
    for observation in observations:
        x1 = round(observation.box.x1 * width)
        y1 = round(observation.box.y1 * height)
        x2 = round(observation.box.x2 * width)
        y2 = round(observation.box.y2 * height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 40), 2)
        label = f"{observation.label} {observation.detector_confidence:.2f}"
        text_y = max(20, y1 - 6)
        cv2.putText(
            frame,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 220, 40),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        f"persons: {len(observations)}",
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def run_video_detection(
    source: Path,
    output_dir: Path,
    detector: FrameDetector,
    *,
    model_name: str,
    confidence: float,
    image_size: int,
    device: str,
    max_frames: int | None = None,
) -> dict:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for video input and output") from exc

    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {source}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("video metadata is invalid")

    output_video = output_dir / "annotated.mp4"
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("cannot initialize MP4 output writer")

    records_path = output_dir / "detections.jsonl"
    preview_path = output_dir / "preview.jpg"
    summary_path = output_dir / "summary.json"
    stats = RunStatistics()
    best_preview = None
    best_preview_count = -1

    first_ok, first_frame = capture.read()
    if not first_ok:
        capture.release()
        writer.release()
        raise RuntimeError("video contains no decodable frames")
    warmup_started = time.perf_counter()
    detector.detect(first_frame)
    warmup_ms = (time.perf_counter() - warmup_started) * 1000.0
    started = time.perf_counter()

    try:
        with records_path.open("w", encoding="utf-8") as records:
            frame_id = 0
            while max_frames is None or frame_id < max_frames:
                if frame_id == 0:
                    frame = first_frame
                else:
                    ok, frame = capture.read()
                    if not ok:
                        break
                inference_started = time.perf_counter()
                observations = list(detector.detect(frame))
                inference_ms = (time.perf_counter() - inference_started) * 1000.0
                stats.record(len(observations), inference_ms)

                record = {
                    "frame_id": frame_id,
                    "timestamp_s": frame_id / fps,
                    "inference_ms": inference_ms,
                    "detections": [
                        observation_to_mapping(item, width, height) for item in observations
                    ],
                }
                records.write(json.dumps(record, ensure_ascii=False) + "\n")

                annotated = _draw_observations(frame.copy(), observations, cv2)
                writer.write(annotated)
                if len(observations) > best_preview_count:
                    best_preview = annotated
                    best_preview_count = len(observations)

                frame_id += 1
                if frame_id % 20 == 0:
                    print(f"processed {frame_id} frames", file=sys.stderr)
    finally:
        capture.release()
        writer.release()

    wall_time_s = time.perf_counter() - started
    if best_preview is not None:
        cv2.imwrite(str(preview_path), best_preview)

    summary = {
        "schema_version": 1,
        "source": str(source),
        "video": {
            "width": width,
            "height": height,
            "fps": fps,
            "declared_frames": declared_frames,
            "duration_s": declared_frames / fps,
        },
        "inference": {
            "backend": "ultralytics",
            "model": model_name,
            "device": device,
            "class_ids": [0],
            "class_names": ["person"],
            "confidence_threshold": confidence,
            "image_size": image_size,
            "warmup_ms": warmup_ms,
        },
        **stats.to_mapping(wall_time_s),
        "outputs": {
            "annotated_video": str(output_video),
            "detections_jsonl": str(records_path),
            "preview_image": str(preview_path),
        },
        "warnings": (
            ["decoded frame count differs from video metadata"]
            if max_frames is None and len(stats.detection_counts) != declared_frames
            else []
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/person_baseline"))
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    from .adapters.ultralytics_yolo import UltralyticsYoloDetector

    detector = UltralyticsYoloDetector(
        args.model,
        class_ids=(0,),
        confidence=args.confidence,
        image_size=args.image_size,
        device=args.device,
    )
    summary = run_video_detection(
        args.source,
        args.output_dir,
        detector,
        model_name=args.model,
        confidence=args.confidence,
        image_size=args.image_size,
        device=args.device,
        max_frames=args.max_frames,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
