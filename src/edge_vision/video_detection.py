"""Run YOLO or YOLO-World over a video and export reproducible evidence."""

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


def read_video_frame(source: Path, frame_index: int) -> Any:
    """Decode one keyframe for low-frequency VLM analysis."""

    if frame_index < 0:
        raise ValueError("VLM frame index must be non-negative")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to read a VLM keyframe") from exc

    capture = cv2.VideoCapture(str(source.expanduser().resolve()))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video for VLM analysis: {source}")
    try:
        if frame_index:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError(f"cannot decode VLM keyframe {frame_index} from {source}")
    return frame


def resolve_backend_defaults(
    backend: str, model: str | None, confidence: float | None
) -> tuple[str, float]:
    if backend == "yolo-world":
        return model or "yolov8s-worldv2.pt", 0.10 if confidence is None else confidence
    if backend == "yolo":
        return model or "yolo11n.pt", 0.25 if confidence is None else confidence
    raise ValueError(f"unsupported backend: {backend}")


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
    track_frame_counts: dict[int, int] = field(default_factory=dict)
    untracked_observations: int = 0

    def record(
        self,
        detection_count: int,
        inference_ms: float,
        track_ids: Sequence[int | None] = (),
    ) -> None:
        self.detection_counts.append(detection_count)
        self.inference_ms.append(inference_ms)
        for track_id in track_ids:
            if track_id is None:
                self.untracked_observations += 1
                continue
            self.track_frame_counts[track_id] = self.track_frame_counts.get(track_id, 0) + 1

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

    def tracking_mapping(self, enabled: bool) -> Mapping[str, Any]:
        frame_counts = list(self.track_frame_counts.values())
        tracked = sum(frame_counts)
        return {
            "enabled": enabled,
            "unique_track_ids": len(frame_counts),
            "tracked_observations": tracked,
            "untracked_observations": self.untracked_observations,
            "observations_per_track": {
                "min": min(frame_counts, default=0),
                "max": max(frame_counts, default=0),
                "mean": statistics.fmean(frame_counts) if frame_counts else 0.0,
            },
        }


def observation_to_mapping(observation: TargetObservation, width: int, height: int) -> dict:
    box = observation.box
    return {
        "class_id": observation.class_id,
        "label": observation.label,
        "track_id": observation.track_id,
        "confidence": observation.detector_confidence,
        "tracker_confidence": observation.tracker_confidence,
        "color_label": observation.color_label,
        "color_score": observation.color_confidence,
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
        if observation.track_id is not None:
            label = f"ID {observation.track_id} | {label}"
        if observation.color_confidence is not None:
            color_label = observation.color_label or "color"
            label += f" {color_label}={observation.color_confidence:.2f}"
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
        f"detections: {len(observations)}",
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
    backend_name: str,
    model_name: str,
    class_ids: Sequence[int],
    class_names: Sequence[str],
    confidence: float,
    image_size: int,
    device: str,
    postprocess: Mapping[str, Any] | None = None,
    tracker_name: str | None = None,
    tracker_config: str | None = None,
    vlm_context: Mapping[str, Any] | None = None,
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
    warmup = getattr(detector, "warmup", None)
    if callable(warmup):
        warmup(first_frame)
    else:
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
                stats.record(
                    len(observations),
                    inference_ms,
                    [item.track_id for item in observations],
                )

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
            "backend": backend_name,
            "model": model_name,
            "device": device,
            "class_ids": list(class_ids),
            "class_names": list(class_names),
            "confidence_threshold": confidence,
            "image_size": image_size,
            "warmup_ms": warmup_ms,
            "tracker": tracker_name,
            "tracker_config": tracker_config,
        },
        "vlm": dict(vlm_context or {}),
        "postprocess": dict(postprocess or {}),
        **stats.to_mapping(wall_time_s),
        "tracking": stats.tracking_mapping(tracker_name is not None),
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
    parser.add_argument("--backend", choices=("yolo-world", "yolo"), default="yolo-world")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--prompts", nargs="+", default=["person"])
    parser.add_argument("--class-ids", nargs="+", type=int, default=[0])
    parser.add_argument("--tracker", choices=("none", "bytetrack"), default="none")
    parser.add_argument("--tracker-config", default="configs/bytetrack.yaml")
    vlm_source = parser.add_mutually_exclusive_group()
    vlm_source.add_argument(
        "--vlm-plan",
        type=Path,
        help="validated, previously generated VLM scene-analysis JSON for deterministic replay",
    )
    vlm_source.add_argument(
        "--vlm-provider",
        choices=("ollama",),
        help="run a real local VLM on one keyframe before starting detection",
    )
    parser.add_argument("--vlm-model", default="qwen3-vl:2b")
    parser.add_argument("--vlm-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--vlm-frame-index", type=int, default=0)
    parser.add_argument(
        "--vlm-instruction",
        default=(
            "Describe the visible objects and create a grounding plan to find and track "
            "people wearing white clothes."
        ),
    )
    parser.add_argument("--white-clothing", action="store_true")
    parser.add_argument(
        "--clothing-color",
        choices=(
            "white",
            "black",
            "gray",
            "red",
            "orange",
            "yellow",
            "green",
            "cyan",
            "blue",
            "purple",
            "pink",
            "brown",
        ),
        help="verify a clothing color inside tracked person boxes",
    )
    parser.add_argument("--color-score-threshold", type=float)
    parser.add_argument("--color-dominance-ratio", type=float, default=0.85)
    parser.add_argument("--white-ratio-threshold", type=float, default=0.20)
    parser.add_argument("--white-saturation-max", type=int, default=60)
    parser.add_argument("--white-value-min", type=int, default=145)
    parser.add_argument("--minimum-person-height-ratio", type=float, default=0.10)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    vlm_context = None
    if args.vlm_plan is not None:
        from .vlm import load_scene_analysis, scene_analysis_to_mapping

        analysis = load_scene_analysis(args.vlm_plan)
        args.prompts = list(analysis.grounding.yolo_world_prompts)
        vlm_context = {
            "mode": "precomputed-replay",
            "source": str(args.vlm_plan.expanduser().resolve()),
            "analysis": scene_analysis_to_mapping(analysis),
        }
    elif args.vlm_provider == "ollama":
        from .adapters.ollama_vlm import OllamaVlm
        from .vlm import normalize_grounding_prompts, scene_analysis_to_mapping

        vlm = OllamaVlm(model=args.vlm_model, base_url=args.vlm_base_url)
        keyframe = read_video_frame(args.source, args.vlm_frame_index)
        analysis = normalize_grounding_prompts(
            vlm.analyze(keyframe, args.vlm_instruction)
        )
        args.prompts = list(analysis.grounding.yolo_world_prompts)
        vlm_context = {
            "mode": "local-live",
            "provider": "ollama",
            "model": args.vlm_model,
            "base_url": args.vlm_base_url,
            "frame_index": args.vlm_frame_index,
            "instruction": args.vlm_instruction,
            "analysis": scene_analysis_to_mapping(analysis),
            "performance": dict(vlm.last_metrics),
        }

    tracker_name = None if args.tracker == "none" else args.tracker
    tracker_config = None
    if tracker_name is not None:
        configured_path = Path(args.tracker_config).expanduser()
        tracker_config = (
            str(configured_path.resolve())
            if configured_path.is_file()
            else args.tracker_config
        )

    model, confidence = resolve_backend_defaults(args.backend, args.model, args.confidence)
    if args.output_dir is None:
        suffix = "yolo_world" if args.backend == "yolo-world" else "yolo"
        args.output_dir = Path(f"outputs/person_{suffix}")

    if args.backend == "yolo-world":
        from .adapters.ultralytics_yolo_world import UltralyticsYoloWorldDetector

        detector = UltralyticsYoloWorldDetector(
            model,
            prompts=args.prompts,
            confidence=confidence,
            image_size=args.image_size,
            device=args.device,
            tracker_config=tracker_config,
        )
        class_ids = detector.class_ids
        class_names = detector.class_names
    else:
        from .adapters.ultralytics_yolo import UltralyticsYoloDetector

        detector = UltralyticsYoloDetector(
            model,
            class_ids=args.class_ids,
            confidence=confidence,
            image_size=args.image_size,
            device=args.device,
            tracker_config=tracker_config,
        )
        class_ids = detector.class_ids
        class_names = detector.class_names

    postprocess = None
    clothing_color = args.clothing_color or ("white" if args.white_clothing else None)
    if clothing_color is not None:
        from .filters.clothing_color import ClothingColorConfig, ClothingColorFilter

        color_config = ClothingColorConfig(
            target_color=clothing_color,
            score_threshold=(
                args.color_score_threshold
                if args.color_score_threshold is not None
                else args.white_ratio_threshold
            ),
            minimum_box_height_ratio=args.minimum_person_height_ratio,
            white_saturation_max=args.white_saturation_max,
            white_value_min=args.white_value_min,
            minimum_dominance_ratio=args.color_dominance_ratio,
        )
        detector = ClothingColorFilter(detector, color_config)
        class_ids = detector.class_ids
        class_names = detector.class_names
        postprocess = {
            "name": "tracked-clothing-color-filter",
            "target_color": color_config.target_color,
            "score_threshold": color_config.score_threshold,
            "temporal_alpha": color_config.temporal_alpha,
            "temporal_min_samples": color_config.temporal_min_samples,
            "minimum_dominance_ratio": color_config.minimum_dominance_ratio,
            "white_saturation_max": color_config.white_saturation_max,
            "white_value_min": color_config.white_value_min,
            "minimum_box_height_ratio": color_config.minimum_box_height_ratio,
            "roi_xy_fractions": [
                color_config.roi_x_start,
                color_config.roi_y_start,
                color_config.roi_x_end,
                color_config.roi_y_end,
            ],
        }

    summary = run_video_detection(
        args.source,
        args.output_dir,
        detector,
        backend_name=args.backend,
        model_name=model,
        class_ids=class_ids,
        class_names=class_names,
        confidence=confidence,
        image_size=args.image_size,
        device=args.device,
        postprocess=postprocess,
        tracker_name=tracker_name,
        tracker_config=tracker_config,
        vlm_context=vlm_context,
        max_frames=args.max_frames,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
