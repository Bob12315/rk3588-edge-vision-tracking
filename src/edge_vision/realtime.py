"""Unified VLM -> YOLO-World -> ByteTrack -> mission-state runtime."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import load_config
from .contracts import (
    Action,
    MissionState,
    PerceptionSnapshot,
    TargetObservation,
    VehicleTelemetry,
    VisionLanguageModel,
    VlmSceneAnalysis,
)
from .pipeline import MissionPipeline
from .safety import SafetyArbiter
from .state_machine import TrackingStateMachine
from .target_selection import (
    PersistentTargetSelector,
    SelectionResult,
    TargetSelectorConfig,
)
from .video_detection import observation_to_mapping, percentile
from .video_source import OpenCvVideoSource
from .vlm import normalize_grounding_prompts, scene_analysis_to_mapping


DEFAULT_INSTRUCTION = (
    "Describe the visible objects and create a grounding plan to find and track "
    "people wearing white clothes."
)


@dataclass
class RuntimeStatistics:
    inference_ms: list[float] = field(default_factory=list)
    candidate_counts: list[int] = field(default_factory=list)
    state_counts: Counter[str] = field(default_factory=Counter)
    requested_actions: Counter[str] = field(default_factory=Counter)
    safe_actions: Counter[str] = field(default_factory=Counter)
    selected_track_ids: Counter[int] = field(default_factory=Counter)
    selection_events: Counter[str] = field(default_factory=Counter)
    transitions: list[dict[str, Any]] = field(default_factory=list)

    def to_mapping(self, wall_time_s: float) -> dict[str, Any]:
        frame_count = len(self.inference_ms)
        return {
            "processed_frames": frame_count,
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
            "candidates": {
                "total": sum(self.candidate_counts),
                "frames_with_candidates": sum(count > 0 for count in self.candidate_counts),
                "per_frame_max": max(self.candidate_counts, default=0),
            },
            "mission": {
                "state_frame_counts": dict(self.state_counts),
                "requested_action_counts": dict(self.requested_actions),
                "safe_action_counts": dict(self.safe_actions),
                "transitions": self.transitions,
            },
            "selection": {
                "track_frame_counts": {
                    str(track_id): count for track_id, count in self.selected_track_ids.items()
                },
                "events": dict(self.selection_events),
            },
        }


class VlmCoordinator:
    """Run a VLM initially and at bounded target-loss events."""

    def __init__(
        self,
        provider: VisionLanguageModel | None,
        instruction: str,
        *,
        cooldown_s: float = 10.0,
        max_triggers: int = 2,
    ) -> None:
        if cooldown_s < 0.0:
            raise ValueError("VLM cooldown must be non-negative")
        if max_triggers < 0:
            raise ValueError("VLM max_triggers must be non-negative")
        self.provider = provider
        self.instruction = instruction
        self.cooldown_s = cooldown_s
        self.max_triggers = max_triggers
        self.events: list[dict[str, Any]] = []
        self.last_trigger_timestamp_s: float | None = None

    def analyze(
        self,
        frame: Any,
        *,
        cause: str,
        frame_id: int,
        timestamp_s: float,
        force: bool = False,
        raise_on_error: bool = False,
    ) -> VlmSceneAnalysis | None:
        if self.provider is None or len(self.events) >= self.max_triggers:
            return None
        if (
            not force
            and self.last_trigger_timestamp_s is not None
            and timestamp_s - self.last_trigger_timestamp_s < self.cooldown_s
        ):
            return None

        started = time.perf_counter()
        try:
            analysis = normalize_grounding_prompts(
                self.provider.analyze(frame, self.instruction)
            )
        except Exception as exc:
            elapsed = time.perf_counter() - started
            self.events.append(
                {
                    "cause": cause,
                    "frame_id": frame_id,
                    "timestamp_s": timestamp_s,
                    "wall_time_s": elapsed,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            self.last_trigger_timestamp_s = timestamp_s
            if raise_on_error:
                raise
            return None
        elapsed = time.perf_counter() - started
        metrics = dict(getattr(self.provider, "last_metrics", {}))
        self.events.append(
            {
                "cause": cause,
                "frame_id": frame_id,
                "timestamp_s": timestamp_s,
                "wall_time_s": elapsed,
                "metrics": metrics,
                "analysis": scene_analysis_to_mapping(analysis),
            }
        )
        self.last_trigger_timestamp_s = timestamp_s
        return analysis

    def add_replay_analysis(
        self, analysis: VlmSceneAnalysis, *, source: Path
    ) -> None:
        self.events.append(
            {
                "cause": "initial_replay",
                "frame_id": 0,
                "timestamp_s": 0.0,
                "source": str(source.expanduser().resolve()),
                "analysis": scene_analysis_to_mapping(analysis),
            }
        )


def _draw_runtime_overlay(
    frame: Any,
    candidates: Sequence[TargetObservation],
    selection: SelectionResult,
    decision: Any,
    cv2: Any,
) -> Any:
    height, width = frame.shape[:2]
    for observation in candidates:
        selected = (
            selection.observation is not None
            and observation.track_id == selection.observation.track_id
        )
        color = (255, 60, 220) if selected else (40, 220, 40)
        thickness = 3 if selected else 2
        x1 = round(observation.box.x1 * width)
        y1 = round(observation.box.y1 * height)
        x2 = round(observation.box.x2 * width)
        y2 = round(observation.box.y2 * height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        track = "?" if observation.track_id is None else str(observation.track_id)
        label = f"ID {track} {observation.detector_confidence:.2f}"
        if observation.color_confidence is not None:
            color_label = observation.color_label or "color"
            label += f" {color_label}={observation.color_confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(22, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    active_id = "none" if selection.active_track_id is None else str(selection.active_track_id)
    lines = (
        f"state={decision.state.value} action={decision.safe_action.value}",
        f"target_id={active_id} candidates={len(candidates)} event={selection.event}",
    )
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (18, 34 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return frame


def _decision_mapping(decision: Any) -> dict[str, Any]:
    return {
        "state": decision.state.value,
        "requested_action": decision.requested_action.value,
        "safe_action": decision.safe_action.value,
        "reason": decision.reason,
        "transition_reason": decision.transition_reason,
        "safety_reason": decision.safety_reason,
        "safety_overridden": decision.safety_overridden,
    }


def run_mission_runtime(
    source: OpenCvVideoSource,
    first_sample: tuple[int, float, Any],
    output_dir: Path,
    detector: Any,
    base_detector: Any,
    pipeline: MissionPipeline,
    selector: PersistentTargetSelector,
    telemetry: VehicleTelemetry,
    vlm: VlmCoordinator,
    *,
    model_name: str,
    confidence: float,
    image_size: int,
    device: str,
    tracker_config: str,
    max_frames: int | None = None,
    display: bool = False,
) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for the unified runtime") from exc

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.jsonl"
    output_video = output_dir / "annotated.mp4"
    preview_path = output_dir / "preview.jpg"
    summary_path = output_dir / "summary.json"
    metadata = source.metadata
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        metadata.fps,
        (metadata.width, metadata.height),
    )
    if not writer.isOpened():
        raise RuntimeError("cannot initialize runtime MP4 writer")

    warmup_started = time.perf_counter()
    warmup = getattr(detector, "warmup", None)
    if callable(warmup):
        warmup(first_sample[2])
    warmup_ms = (time.perf_counter() - warmup_started) * 1000.0
    stats = RuntimeStatistics()
    previous_state = pipeline.state_machine.state
    best_preview = None
    best_preview_score = -1
    stop_reason = "end_of_stream"
    started = time.perf_counter()
    sample: tuple[int, float, Any] | None = first_sample

    try:
        with events_path.open("w", encoding="utf-8") as events:
            while sample is not None:
                frame_id, timestamp_s, frame = sample
                if max_frames is not None and len(stats.inference_ms) >= max_frames:
                    stop_reason = "max_frames"
                    break

                inference_started = time.perf_counter()
                candidates = list(detector.detect(frame))
                inference_ms = (time.perf_counter() - inference_started) * 1000.0
                selection = selector.select(candidates)
                snapshot = PerceptionSnapshot(
                    frame_id=frame_id,
                    timestamp_s=timestamp_s,
                    target=selection.observation,
                )
                decision = pipeline.process(snapshot, telemetry)
                entered_lost = (
                    decision.state is MissionState.LOST
                    and previous_state is not MissionState.LOST
                )
                vlm_event = None
                if entered_lost:
                    event_count = len(vlm.events)
                    analysis = vlm.analyze(
                        frame,
                        cause="target_lost",
                        frame_id=frame_id,
                        timestamp_s=timestamp_s,
                    )
                    if analysis is not None:
                        base_detector.set_prompts(analysis.grounding.yolo_world_prompts)
                    if len(vlm.events) > event_count:
                        vlm_event = vlm.events[event_count]

                if decision.state is not previous_state:
                    safety_transition = (
                        decision.state is MissionState.RTL_LAND
                        and decision.safety_overridden
                    )
                    stats.transitions.append(
                        {
                            "frame_id": frame_id,
                            "timestamp_s": timestamp_s,
                            "from": previous_state.value,
                            "to": decision.state.value,
                            "cause": "safety" if safety_transition else "mission",
                            "reason": (
                                decision.safety_reason
                                if safety_transition
                                else decision.transition_reason
                            ),
                        }
                    )
                previous_state = decision.state

                stats.inference_ms.append(inference_ms)
                stats.candidate_counts.append(len(candidates))
                stats.state_counts[decision.state.value] += 1
                stats.requested_actions[decision.requested_action.value] += 1
                stats.safe_actions[decision.safe_action.value] += 1
                stats.selection_events[selection.event] += 1
                if selection.observation is not None and selection.observation.track_id is not None:
                    stats.selected_track_ids[selection.observation.track_id] += 1

                record = {
                    "schema_version": 1,
                    "frame_id": frame_id,
                    "timestamp_s": timestamp_s,
                    "inference_ms": inference_ms,
                    "candidates": [
                        observation_to_mapping(item, metadata.width, metadata.height)
                        for item in candidates
                    ],
                    "selection": {
                        "active_track_id": selection.active_track_id,
                        "event": selection.event,
                        "reason": selection.reason,
                        "selected": (
                            observation_to_mapping(
                                selection.observation, metadata.width, metadata.height
                            )
                            if selection.observation is not None
                            else None
                        ),
                    },
                    "mission": _decision_mapping(decision),
                    "vlm_event": vlm_event,
                }
                events.write(json.dumps(record, ensure_ascii=False) + "\n")

                annotated = _draw_runtime_overlay(
                    frame.copy(), candidates, selection, decision, cv2
                )
                writer.write(annotated)
                preview_score = len(candidates) * 10 + int(selection.observation is not None)
                if preview_score > best_preview_score:
                    best_preview = annotated
                    best_preview_score = preview_score

                if display:
                    cv2.imshow("RK3588 edge vision runtime", annotated)
                    if cv2.waitKey(1) & 0xFF in {ord("q"), 27}:
                        stop_reason = "operator_exit"
                        break
                if len(stats.inference_ms) % 20 == 0:
                    print(
                        f"processed {len(stats.inference_ms)} frames; "
                        f"state={decision.state.value}; target={selection.active_track_id}",
                        file=sys.stderr,
                    )
                sample = source.read()
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
    finally:
        writer.release()
        if display:
            cv2.destroyAllWindows()

    wall_time_s = time.perf_counter() - started
    if best_preview is not None:
        cv2.imwrite(str(preview_path), best_preview)
    summary = {
        "schema_version": 1,
        "source": asdict(metadata),
        "inference": {
            "backend": "yolo-world",
            "model": model_name,
            "device": device,
            "confidence_threshold": confidence,
            "image_size": image_size,
            "tracker": "bytetrack",
            "tracker_config": tracker_config,
            "warmup_ms": warmup_ms,
        },
        "vlm": {
            "instruction": vlm.instruction,
            "cooldown_s": vlm.cooldown_s,
            "max_triggers": vlm.max_triggers,
            "events": vlm.events,
        },
        "telemetry": asdict(telemetry),
        "stop_reason": stop_reason,
        **stats.to_mapping(wall_time_s),
        "final": {
            "state": pipeline.state_machine.state.value,
            "active_track_id": selector.active_track_id,
        },
        "outputs": {
            "annotated_video": str(output_video),
            "events_jsonl": str(events_path),
            "preview_image": str(preview_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="video file, camera index such as 0, or RTSP/HTTP URL")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mission_runtime"))
    parser.add_argument("--config", type=Path, default=Path("configs/white_person_runtime.json"))
    parser.add_argument("--model", default="yolov8s-worldv2.pt")
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tracker-config", default="configs/bytetrack.yaml")
    parser.add_argument("--selector-miss-tolerance", type=int, default=5)
    parser.add_argument("--white-ratio-threshold", type=float, default=0.20)
    parser.add_argument("--white-saturation-max", type=int, default=60)
    parser.add_argument("--white-value-min", type=int, default=145)
    parser.add_argument("--minimum-person-height-ratio", type=float, default=0.10)

    vlm_source = parser.add_mutually_exclusive_group()
    vlm_source.add_argument("--vlm-provider", choices=("ollama",), default=None)
    vlm_source.add_argument("--vlm-plan", type=Path)
    vlm_source.add_argument("--no-vlm", action="store_true")
    parser.add_argument("--vlm-model", default="qwen3-vl:2b")
    parser.add_argument("--vlm-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--vlm-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--vlm-instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--vlm-cooldown-seconds", type=float, default=10.0)
    parser.add_argument("--vlm-max-triggers", type=int, default=2)

    parser.add_argument("--battery-percent", type=float, default=100.0)
    parser.add_argument("--localization-unhealthy", action="store_true")
    parser.add_argument("--obstacle-distance-m", type=float)
    parser.add_argument("--operator-hold", action="store_true")
    parser.add_argument("--fallback-fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--display", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    app_config = load_config(args.config)
    tracker_path = Path(args.tracker_config).expanduser()
    tracker_config = (
        str(tracker_path.resolve()) if tracker_path.is_file() else args.tracker_config
    )

    with OpenCvVideoSource(args.source, fallback_fps=args.fallback_fps) as source:
        first_sample = source.read()
        if first_sample is None:
            raise RuntimeError("video source contains no decodable frame")

        provider: VisionLanguageModel | None = None
        replay_analysis = None
        if args.vlm_plan is not None:
            from .vlm import load_scene_analysis

            replay_analysis = normalize_grounding_prompts(load_scene_analysis(args.vlm_plan))
        elif not args.no_vlm:
            from .adapters.ollama_vlm import OllamaVlm

            provider = OllamaVlm(
                model=args.vlm_model,
                base_url=args.vlm_base_url,
                timeout_seconds=args.vlm_timeout_seconds,
            )

        vlm = VlmCoordinator(
            provider,
            args.vlm_instruction,
            cooldown_s=args.vlm_cooldown_seconds,
            max_triggers=args.vlm_max_triggers,
        )
        if replay_analysis is not None:
            analysis = replay_analysis
            vlm.add_replay_analysis(analysis, source=args.vlm_plan)
        elif provider is not None:
            analysis = vlm.analyze(
                first_sample[2],
                cause="initial_scene_analysis",
                frame_id=first_sample[0],
                timestamp_s=first_sample[1],
                force=True,
                raise_on_error=True,
            )
        else:
            analysis = None
        prompts = (
            analysis.grounding.yolo_world_prompts if analysis is not None else ("person",)
        )

        from .adapters.ultralytics_yolo_world import UltralyticsYoloWorldDetector
        from .filters.white_clothing import WhiteClothingConfig, WhiteClothingFilter

        base_detector = UltralyticsYoloWorldDetector(
            args.model,
            prompts=prompts,
            confidence=args.confidence,
            image_size=args.image_size,
            device=args.device,
            tracker_config=tracker_config,
        )
        detector = WhiteClothingFilter(
            base_detector,
            WhiteClothingConfig(
                saturation_max=args.white_saturation_max,
                value_min=args.white_value_min,
                ratio_threshold=args.white_ratio_threshold,
                minimum_box_height_ratio=args.minimum_person_height_ratio,
            ),
        )
        selector = PersistentTargetSelector(
            TargetSelectorConfig(miss_tolerance=args.selector_miss_tolerance)
        )
        pipeline = MissionPipeline(
            TrackingStateMachine(app_config.state_machine),
            SafetyArbiter(app_config.safety),
        )
        telemetry = VehicleTelemetry(
            battery_percent=args.battery_percent,
            localization_ok=not args.localization_unhealthy,
            obstacle_distance_m=args.obstacle_distance_m,
            operator_hold=args.operator_hold,
        )
        summary = run_mission_runtime(
            source,
            first_sample,
            args.output_dir,
            detector,
            base_detector,
            pipeline,
            selector,
            telemetry,
            vlm,
            model_name=args.model,
            confidence=args.confidence,
            image_size=args.image_size,
            device=args.device,
            tracker_config=tracker_config,
            max_frames=args.max_frames,
            display=args.display,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
