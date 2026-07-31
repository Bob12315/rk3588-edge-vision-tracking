"""Local web UI for VLM scene analysis and YOLO-World + ByteTrack tracking."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
import time
import uuid
import webbrowser
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping, Sequence
from urllib.parse import urlparse

from .contracts import TargetObservation, VisionLanguageModel, VlmSceneAnalysis
from .filters.clothing_color import clothing_color_from_text, normalize_color_name
from .video_detection import observation_to_mapping
from .vlm import normalize_grounding_prompts, scene_analysis_to_mapping


SCENE_INSTRUCTION = (
    "Describe the visible scene in concise English and list the important visible object "
    "categories and attributes in English. In yolo_world_prompts, return up to four short "
    "English physical-object categories that are visibly present and suitable for "
    "YOLO-World bounding boxes."
)
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
PERFORMANCE_IMAGE_SIZES = {
    "realtime": 384,
    "balanced": 512,
    "quality": 640,
}


def split_direct_prompts(value: str) -> tuple[str, ...]:
    """Parse comma-separated direct YOLO-World prompts."""

    normalized = value.replace("，", ",")
    prompts = tuple(item.strip() for item in normalized.split(",") if item.strip())
    if not prompts:
        raise ValueError("检测目标不能为空")
    return prompts


def safe_upload_filename(filename: str) -> str:
    """Return a local-only, collision-resistant upload name."""

    basename = Path(filename).name.strip()
    suffix = Path(basename).suffix.casefold()
    if not basename or suffix not in VIDEO_EXTENSIONS:
        supported = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise ValueError(f"不支持该视频格式，可用格式：{supported}")
    stem = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in Path(basename).stem
    ).strip("_")
    stem = stem[:80] or "video"
    return f"{stem}-{uuid.uuid4().hex[:10]}{suffix}"


def requested_clothing_color(
    analysis: VlmSceneAnalysis | None,
    query: str = "",
) -> str | None:
    """Extract a requested garment color without treating every colored object as clothing."""

    candidates = [query]
    person_prompt = False
    if analysis is not None:
        candidates.extend(analysis.grounding.required_attributes)
        candidates.append(analysis.grounding.user_query)
        person_prompt = any(
            prompt.casefold() in {"person", "people", "man", "woman"}
            for prompt in analysis.grounding.yolo_world_prompts
        )
    for candidate in candidates:
        color = clothing_color_from_text(candidate)
        if color is not None:
            return color
        if person_prompt:
            normalized = normalize_color_name(candidate)
            words = candidate.casefold().replace("color", "").strip(" -_")
            if normalized is not None and words in {normalized, "grey"}:
                return normalized
    return None


def requires_white_clothing_filter(analysis: VlmSceneAnalysis | None) -> bool:
    """Compatibility helper retained for callers of the first white-only version."""

    return requested_clothing_color(analysis) == "white"


def performance_image_size(mode: str) -> int:
    try:
        return PERFORMANCE_IMAGE_SIZES[mode]
    except KeyError as exc:
        choices = ", ".join(PERFORMANCE_IMAGE_SIZES)
        raise ValueError(f"未知性能模式，可选：{choices}") from exc


def grounding_detection_prompts(
    prompts: Sequence[str], clothing_color: str | None
) -> tuple[str, ...]:
    """Keep clothing attributes out of YOLO-World's object-class vocabulary."""

    if clothing_color is not None:
        return ("person",)
    return tuple(prompts)


def timestamp_rate(timestamps: Sequence[float]) -> float:
    if len(timestamps) < 2:
        return 0.0
    elapsed = timestamps[-1] - timestamps[0]
    return (len(timestamps) - 1) / elapsed if elapsed > 0.0 else 0.0


@dataclass(frozen=True)
class WebUiConfig:
    model: str = "yolov8s-worldv2.pt"
    confidence: float = 0.05
    image_size: int = 384
    device: str = "cpu"
    tracker_config: str = "configs/bytetrack_balanced.yaml"
    clothing_color_threshold: float = 0.20
    color_minimum_dominance_ratio: float = 0.85
    color_temporal_min_samples: int = 3
    minimum_person_height_ratio: float = 0.10
    jpeg_quality: int = 82

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if not 0.0 <= self.clothing_color_threshold <= 1.0:
            raise ValueError("clothing_color_threshold must be in [0, 1]")
        if not 0.0 <= self.color_minimum_dominance_ratio <= 1.0:
            raise ValueError("color_minimum_dominance_ratio must be in [0, 1]")
        if self.color_temporal_min_samples < 1:
            raise ValueError("color_temporal_min_samples must be positive")
        if not 0.0 <= self.minimum_person_height_ratio <= 1.0:
            raise ValueError("minimum_person_height_ratio must be in [0, 1]")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")


DetectorFactory = Callable[[Sequence[str], str | None, int], tuple[Any, Any]]


class VisionWebSession:
    """Thread-safe capture, VLM, detection, and tracking session shared by HTTP clients."""

    def __init__(
        self,
        config: WebUiConfig,
        vlm: VisionLanguageModel,
        *,
        detector_factory: DetectorFactory | None = None,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("网页 UI 需要 OpenCV，请安装 requirements-pc.txt") from exc

        self.config = config
        self.vlm = vlm
        self._cv2 = cv2
        self._detector_factory = detector_factory or self._build_detector
        self._lock = threading.RLock()
        self._operation_lock = threading.Lock()
        self._vlm_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._capture: Any | None = None
        self._source_kind: str | None = None
        self._source_label: str | None = None
        self._source_width = 0
        self._source_height = 0
        self._source_fps = 0.0
        self._generation = 0
        self._raw_frame: Any | None = None
        self._jpeg: bytes | None = None
        self._detector: Any | None = None
        self._base_detector: Any | None = None
        self._tracking = False
        self._phase = "idle"
        self._error: str | None = None
        self._frame_id = 0
        self._detections: list[TargetObservation] = []
        self._seen_track_ids: set[int] = set()
        self._matched_frame_count = 0
        self._target_query = ""
        self._prompts: tuple[str, ...] = ()
        self._attribute_filter: str | None = None
        self._target_color: str | None = None
        self._performance_mode = "realtime"
        self._active_image_size = config.image_size
        self._scene_analysis: Mapping[str, Any] | None = None
        self._grounding_analysis: Mapping[str, Any] | None = None
        self._vlm_metrics: Mapping[str, Any] = {}
        self._vlm_busy = False
        self._detector_loading = False
        self._inference_ms: deque[float] = deque(maxlen=60)
        self._processed_timestamps: deque[float] = deque(maxlen=60)
        self._read_failures = 0
        self._worker = threading.Thread(
            target=self._capture_loop,
            name="edge-vision-web-capture",
            daemon=True,
        )
        self._worker.start()

    def use_camera(self, index: int) -> Mapping[str, Any]:
        if index < 0:
            raise ValueError("摄像头编号不能为负数")
        capture = self._cv2.VideoCapture(index)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"无法打开 USB 摄像头 {index}")
        capture.set(self._cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"USB 摄像头 {index} 无法读取画面")
        self._replace_source(capture, "camera", f"USB 摄像头 {index}", frame)
        return self.status()

    def use_video(self, path: Path, *, display_name: str | None = None) -> Mapping[str, Any]:
        source = path.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        capture = self._cv2.VideoCapture(str(source))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError("无法打开上传的视频")
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError("上传的视频没有可解码画面")
        self._replace_source(capture, "video", display_name or source.name, frame)
        return self.status()

    def _replace_source(
        self, capture: Any, source_kind: str, source_label: str, first_frame: Any
    ) -> None:
        width = int(capture.get(self._cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(self._cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(self._cv2.CAP_PROP_FPS))
        if width <= 0 or height <= 0:
            capture.release()
            raise RuntimeError("视频源返回了无效分辨率")
        if fps <= 0.0:
            fps = 30.0

        with self._lock:
            if self._capture is not None:
                self._capture.release()
            self._capture = capture
            self._source_kind = source_kind
            self._source_label = source_label
            self._source_width = width
            self._source_height = height
            self._source_fps = fps
            self._generation += 1
            self._detector = None
            self._base_detector = None
            self._tracking = False
            self._phase = "preview"
            self._error = None
            self._frame_id = 0
            self._detections = []
            self._seen_track_ids.clear()
            self._matched_frame_count = 0
            self._target_query = ""
            self._prompts = ()
            self._attribute_filter = None
            self._target_color = None
            self._performance_mode = "realtime"
            self._active_image_size = self.config.image_size
            self._grounding_analysis = None
            self._scene_analysis = None
            self._vlm_metrics = {}
            self._inference_ms.clear()
            self._processed_timestamps.clear()
            self._read_failures = 0
            self._store_frame_locked(first_frame, first_frame)

    def analyze_scene(self, instruction: str | None = None) -> Mapping[str, Any]:
        frame = self._frame_snapshot()
        with self._operation_lock:
            with self._lock:
                self._vlm_busy = True
                self._error = None
            try:
                with self._vlm_lock:
                    analysis = normalize_grounding_prompts(
                        self.vlm.analyze(frame, instruction or SCENE_INSTRUCTION)
                    )
                mapping = scene_analysis_to_mapping(analysis)
                with self._lock:
                    self._scene_analysis = mapping
                    self._vlm_metrics = dict(getattr(self.vlm, "last_metrics", {}))
                return mapping
            except Exception as exc:
                self._set_error(f"VLM 分析失败：{exc}")
                raise
            finally:
                with self._lock:
                    self._vlm_busy = False

    def start_tracking(
        self,
        target_query: str,
        *,
        use_vlm_grounding: bool = True,
        performance_mode: str = "realtime",
    ) -> Mapping[str, Any]:
        query = target_query.strip()
        if not query:
            raise ValueError("请输入要识别和跟踪的目标")
        if len(query) > 500:
            raise ValueError("目标描述不能超过 500 个字符")
        image_size = performance_image_size(performance_mode)
        frame = self._frame_snapshot()

        with self._operation_lock:
            with self._lock:
                self._detector_loading = True
                self._error = None
            try:
                grounding_analysis: VlmSceneAnalysis | None = None
                if use_vlm_grounding:
                    instruction = (
                        "The user wants to detect and track this target: "
                        f"{query}. Inspect the image, describe the scene briefly, and create a "
                        "grounding plan specifically for that target. Use short English object "
                        "nouns for YOLO-World prompts and put colors, clothing, and relations in "
                        "required_attributes or relation. Express clothing colors using one of "
                        "white, black, gray, red, orange, yellow, green, cyan, blue, purple, "
                        "pink, or brown."
                    )
                    with self._lock:
                        self._vlm_busy = True
                    try:
                        with self._vlm_lock:
                            grounding_analysis = normalize_grounding_prompts(
                                self.vlm.analyze(frame, instruction)
                            )
                    finally:
                        with self._lock:
                            self._vlm_busy = False
                    prompts = grounding_analysis.grounding.yolo_world_prompts
                else:
                    prompts = split_direct_prompts(query)

                clothing_color = requested_clothing_color(grounding_analysis, query)
                prompts = grounding_detection_prompts(prompts, clothing_color)
                detector, base_detector = self._detector_factory(
                    prompts, clothing_color, image_size
                )
                warmup = getattr(detector, "warmup", None)
                if callable(warmup):
                    warmup(frame)

                with self._lock:
                    if self._capture is None:
                        raise RuntimeError("视频源已断开，请重新选择")
                    if self._source_kind == "video":
                        self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, 0)
                    self._generation += 1
                    self._detector = detector
                    self._base_detector = base_detector
                    self._tracking = True
                    self._phase = "tracking"
                    self._frame_id = 0
                    self._detections = []
                    self._seen_track_ids.clear()
                    self._matched_frame_count = 0
                    self._target_query = query
                    self._prompts = tuple(prompts)
                    self._attribute_filter = (
                        f"clothing-color:{clothing_color}"
                        if clothing_color is not None
                        else None
                    )
                    self._target_color = clothing_color
                    self._performance_mode = performance_mode
                    self._active_image_size = image_size
                    self._grounding_analysis = (
                        scene_analysis_to_mapping(grounding_analysis)
                        if grounding_analysis is not None
                        else None
                    )
                    if grounding_analysis is not None:
                        self._vlm_metrics = dict(
                            getattr(self.vlm, "last_metrics", {})
                        )
                    self._inference_ms.clear()
                    self._processed_timestamps.clear()
                    self._read_failures = 0
                return self.status()
            except Exception as exc:
                self._set_error(f"启动检测跟踪失败：{exc}")
                raise
            finally:
                with self._lock:
                    self._detector_loading = False

    def stop_tracking(self) -> Mapping[str, Any]:
        with self._lock:
            self._generation += 1
            self._tracking = False
            self._detector = None
            self._base_detector = None
            self._detections = []
            self._phase = "preview" if self._capture is not None else "idle"
            self._error = None
        return self.status()

    def status(self) -> Mapping[str, Any]:
        with self._lock:
            detections = [
                observation_to_mapping(item, self._source_width, self._source_height)
                for item in self._detections
            ]
            inference_values = list(self._inference_ms)
            processed_timestamps = list(self._processed_timestamps)
            mean_inference = (
                sum(inference_values) / len(inference_values)
                if inference_values
                else 0.0
            )
            track_ids = sorted(
                {
                    item.track_id
                    for item in self._detections
                    if item.track_id is not None
                }
            )
            return {
                "phase": self._phase,
                "error": self._error,
                "busy": self._vlm_busy or self._detector_loading,
                "vlm_busy": self._vlm_busy,
                "detector_loading": self._detector_loading,
                "source": {
                    "connected": self._capture is not None,
                    "kind": self._source_kind,
                    "label": self._source_label,
                    "width": self._source_width,
                    "height": self._source_height,
                    "fps": self._source_fps,
                },
                "scene_analysis": self._scene_analysis,
                "grounding_analysis": self._grounding_analysis,
                "vlm_metrics": dict(self._vlm_metrics),
                "tracking": {
                    "active": self._tracking,
                    "target_query": self._target_query,
                    "prompts": list(self._prompts),
                    "attribute_filter": self._attribute_filter,
                    "target_color": self._target_color,
                    "performance_mode": self._performance_mode,
                    "image_size": self._active_image_size,
                    "frame_id": self._frame_id,
                    "detections": detections,
                    "detection_count": len(detections),
                    "track_ids": track_ids,
                    "seen_track_ids": sorted(self._seen_track_ids),
                    "matched_frame_count": self._matched_frame_count,
                    "mean_inference_ms": mean_inference,
                    "model_fps": 1000.0 / mean_inference if mean_inference else 0.0,
                    "processing_fps": timestamp_rate(processed_timestamps),
                },
            }

    def jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def close(self) -> None:
        self._stop_event.set()
        self._worker.join(timeout=2.0)
        with self._lock:
            if self._capture is not None:
                self._capture.release()
                self._capture = None

    def _frame_snapshot(self) -> Any:
        with self._lock:
            if self._raw_frame is None:
                raise RuntimeError("请先连接 USB 摄像头或上传视频")
            return self._raw_frame.copy()

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                capture = self._capture
                source_kind = self._source_kind
                tracking = self._tracking
                detector = self._detector
                generation = self._generation
                if capture is None or (source_kind == "video" and not tracking):
                    should_read = False
                else:
                    should_read = True
                    ok, frame = capture.read()

            if not should_read:
                self._stop_event.wait(0.04)
                continue
            if not ok:
                if source_kind == "video":
                    with self._lock:
                        if generation == self._generation:
                            self._tracking = False
                            self._detector = None
                            self._base_detector = None
                            self._phase = "finished"
                    self._stop_event.wait(0.04)
                    continue
                self._read_failures += 1
                if self._read_failures >= 15:
                    self._set_error("USB 摄像头连续读取失败，请重新连接")
                    with self._lock:
                        self._tracking = False
                        self._phase = "error"
                self._stop_event.wait(0.05)
                continue

            self._read_failures = 0
            observations: list[TargetObservation] = []
            inference_ms = 0.0
            display_frame = frame
            if tracking and detector is not None:
                started = time.perf_counter()
                try:
                    observations = list(detector.detect(frame))
                except Exception as exc:
                    self._set_error(f"检测跟踪运行失败：{exc}")
                    with self._lock:
                        self._tracking = False
                        self._phase = "error"
                    continue
                inference_ms = (time.perf_counter() - started) * 1000.0
                display_frame = draw_tracking_frame(
                    frame.copy(),
                    observations,
                    self._target_query,
                    tuple(self._prompts),
                    self._cv2,
                )

            with self._lock:
                if generation != self._generation:
                    continue
                self._frame_id += 1
                self._detections = observations
                self._seen_track_ids.update(
                    item.track_id for item in observations if item.track_id is not None
                )
                if observations:
                    self._matched_frame_count += 1
                if inference_ms:
                    self._inference_ms.append(inference_ms)
                self._store_frame_locked(frame, display_frame)
                if inference_ms:
                    self._processed_timestamps.append(time.perf_counter())
            self._stop_event.wait(0.003)

    def _store_frame_locked(self, raw_frame: Any, display_frame: Any) -> None:
        ok, encoded = self._cv2.imencode(
            ".jpg",
            display_frame,
            [self._cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality],
        )
        self._raw_frame = raw_frame.copy()
        if ok:
            self._jpeg = encoded.tobytes()

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._error = message

    def _build_detector(
        self, prompts: Sequence[str], clothing_color: str | None, image_size: int
    ) -> tuple[Any, Any]:
        from .adapters.ultralytics_yolo_world import UltralyticsYoloWorldDetector

        base_detector = UltralyticsYoloWorldDetector(
            self.config.model,
            prompts=prompts,
            confidence=self.config.confidence,
            image_size=image_size,
            device=self.config.device,
            tracker_config=self.config.tracker_config,
        )
        if clothing_color is None:
            return base_detector, base_detector

        from .filters.clothing_color import ClothingColorConfig, ClothingColorFilter

        detector = ClothingColorFilter(
            base_detector,
            ClothingColorConfig(
                target_color=clothing_color,
                score_threshold=self.config.clothing_color_threshold,
                minimum_dominance_ratio=self.config.color_minimum_dominance_ratio,
                minimum_box_height_ratio=self.config.minimum_person_height_ratio,
                temporal_min_samples=self.config.color_temporal_min_samples,
            ),
        )
        return detector, base_detector


def draw_tracking_frame(
    frame: Any,
    observations: Sequence[TargetObservation],
    target_query: str,
    prompts: Sequence[str],
    cv2: Any,
) -> Any:
    height, width = frame.shape[:2]
    for observation in observations:
        track_id = observation.track_id
        color_seed = 0 if track_id is None else track_id
        color = (
            60 + (color_seed * 47) % 180,
            80 + (color_seed * 73) % 160,
            90 + (color_seed * 97) % 150,
        )
        x1 = round(observation.box.x1 * width)
        y1 = round(observation.box.y1 * height)
        x2 = round(observation.box.x2 * width)
        y2 = round(observation.box.y2 * height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        identity = "?" if track_id is None else str(track_id)
        label = f"ID {identity} | {observation.label} {observation.detector_confidence:.2f}"
        if observation.color_confidence is not None:
            color_label = observation.color_label or "color"
            label += f" | {color_label} {observation.color_confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(24, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 88), (12, 18, 30), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.putText(
        frame,
        f"Target: {target_query}",
        (18, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (235, 240, 250),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"YOLO-World: {', '.join(prompts)} | detections: {len(observations)}",
        (18, 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (70, 220, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


class VisionHttpServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def build_handler(
    session: VisionWebSession,
    *,
    static_dir: Path,
    upload_dir: Path,
    max_upload_bytes: int,
) -> type[BaseHTTPRequestHandler]:
    static_files = {
        "/": (static_dir / "index.html", "text/html; charset=utf-8"),
        "/app.js": (static_dir / "app.js", "text/javascript; charset=utf-8"),
        "/styles.css": (static_dir / "styles.css", "text/css; charset=utf-8"),
        "/favicon.svg": (static_dir / "favicon.svg", "image/svg+xml"),
    }

    class Handler(BaseHTTPRequestHandler):
        server_version = "EdgeVisionUI/0.1"

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path in static_files:
                file_path, content_type = static_files[path]
                self._send_bytes(file_path.read_bytes(), content_type)
                return
            if path == "/api/status":
                self._send_json(session.status())
                return
            if path == "/stream":
                self._stream_mjpeg()
                return
            self._send_json({"error": "not found"}, status=404)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                if path == "/api/source/camera":
                    payload = self._read_json()
                    self._send_json(session.use_camera(int(payload.get("index", 0))))
                    return
                if path == "/api/source/upload":
                    uploaded_path, display_name = self._save_upload()
                    self._send_json(
                        session.use_video(uploaded_path, display_name=display_name)
                    )
                    return
                if path == "/api/analyze":
                    payload = self._read_json()
                    instruction = str(payload.get("instruction", "")).strip() or None
                    self._send_json({"analysis": session.analyze_scene(instruction)})
                    return
                if path == "/api/tracking/start":
                    payload = self._read_json()
                    target = str(payload.get("target", ""))
                    use_vlm = bool(payload.get("use_vlm_grounding", True))
                    performance_mode = str(
                        payload.get("performance_mode", "realtime")
                    )
                    self._send_json(
                        session.start_tracking(
                            target,
                            use_vlm_grounding=use_vlm,
                            performance_mode=performance_mode,
                        )
                    )
                    return
                if path == "/api/tracking/stop":
                    self._read_json()
                    self._send_json(session.stop_tracking())
                    return
                self._send_json({"error": "not found"}, status=404)
            except (ValueError, FileNotFoundError) as exc:
                self._send_json({"error": str(exc)}, status=400)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)

        def _read_json(self) -> Mapping[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length > 64 * 1024:
                raise ValueError("JSON 请求过大")
            if length == 0:
                return {}
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("JSON 请求必须是对象")
            return payload

        def _save_upload(self) -> tuple[Path, str]:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0:
                raise ValueError("没有收到视频文件")
            if length > max_upload_bytes:
                raise ValueError(
                    f"视频超过上传限制 {max_upload_bytes // (1024 * 1024)} MB"
                )
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                raise ValueError("上传请求必须使用 multipart/form-data")

            import cgi

            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                    "CONTENT_LENGTH": str(length),
                },
            )
            if "file" not in form:
                raise ValueError("上传表单缺少 file 字段")
            item = form["file"]
            if isinstance(item, list) or not item.filename or item.file is None:
                raise ValueError("一次只能上传一个视频文件")
            destination = upload_dir / safe_upload_filename(item.filename)
            upload_dir.mkdir(parents=True, exist_ok=True)
            try:
                with destination.open("wb") as output:
                    _copy_limited(item.file, output, max_upload_bytes)
            except Exception:
                destination.unlink(missing_ok=True)
                raise
            return destination, Path(item.filename).name

        def _stream_mjpeg(self) -> None:
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            last_frame: bytes | None = None
            try:
                while True:
                    frame = session.jpeg()
                    if frame is not None and frame is not last_frame:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                        last_frame = frame
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                return

        def _send_json(self, payload: Mapping[str, Any], status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send_bytes(data, "application/json; charset=utf-8", status)

        def _send_bytes(
            self, data: bytes, content_type: str, status: int = 200
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' blob:; "
                "style-src 'self'; script-src 'self'",
            )
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt: str, *args: object) -> None:
            if urlparse(self.path).path in {"/api/status", "/stream"}:
                return
            print(f"web-ui {self.address_string()} {fmt % args}", file=sys.stderr)

    return Handler


def _copy_limited(source: BinaryIO, destination: BinaryIO, limit: int) -> None:
    copied = 0
    while True:
        chunk = source.read(min(1024 * 1024, limit - copied + 1))
        if not chunk:
            return
        copied += len(chunk)
        if copied > limit:
            raise ValueError("视频超过上传限制")
        destination.write(chunk)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--model", default="yolov8s-worldv2.pt")
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--tracker-config", default="configs/bytetrack_balanced.yaml"
    )
    parser.add_argument("--vlm-model", default="qwen3-vl:2b")
    parser.add_argument("--vlm-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--vlm-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--upload-dir", type=Path, default=Path("artifacts/web-ui/uploads"))
    parser.add_argument("--max-upload-mb", type=int, default=2048)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in [1, 65535]")
    if args.max_upload_mb <= 0:
        raise ValueError("max-upload-mb must be positive")
    tracker_path = Path(args.tracker_config).expanduser()
    tracker_config = (
        str(tracker_path.resolve()) if tracker_path.is_file() else args.tracker_config
    )

    from .adapters.ollama_vlm import OllamaVlm

    vlm = OllamaVlm(
        model=args.vlm_model,
        base_url=args.vlm_base_url,
        timeout_seconds=args.vlm_timeout_seconds,
    )
    session = VisionWebSession(
        WebUiConfig(
            model=args.model,
            confidence=args.confidence,
            device=args.device,
            tracker_config=tracker_config,
        ),
        vlm,
    )
    static_dir = Path(__file__).with_name("web")
    handler = build_handler(
        session,
        static_dir=static_dir,
        upload_dir=args.upload_dir.expanduser().resolve(),
        max_upload_bytes=args.max_upload_mb * 1024 * 1024,
    )
    server = VisionHttpServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Edge Vision Web UI: {url}")
    print("Press Ctrl+C to stop.")
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print("Warning: the UI has no authentication; use a trusted private network only.")
    if args.open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        session.close()


if __name__ == "__main__":
    main()
