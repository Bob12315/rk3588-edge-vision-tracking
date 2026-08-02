"""Persistent, review-gated identity annotations for MOT evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Sequence

from .tracking_evaluation import load_detection_jsonl


@dataclass(frozen=True)
class IdentityAnnotation:
    identity_id: int
    x: float
    y: float
    width: float
    height: float
    visibility: float = 1.0

    def __post_init__(self) -> None:
        if self.identity_id < 1:
            raise ValueError("identity_id must be positive")
        if self.x < 0.0 or self.y < 0.0:
            raise ValueError("box coordinates must be non-negative")
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("box width and height must be positive")
        if not 0.0 <= self.visibility <= 1.0:
            raise ValueError("visibility must be in [0, 1]")

    def mot_row(self, frame_id: int) -> str:
        return (
            f"{frame_id},{self.identity_id},{self.x:.3f},{self.y:.3f},"
            f"{self.width:.3f},{self.height:.3f},1,1,{self.visibility:.6f}"
        )


def annotation_from_mapping(
    value: Mapping[str, Any], *, width: int, height: int
) -> IdentityAnnotation:
    try:
        annotation = IdentityAnnotation(
            identity_id=int(value["identity_id"]),
            x=float(value["x"]),
            y=float(value["y"]),
            width=float(value["width"]),
            height=float(value["height"]),
            visibility=float(value.get("visibility", 1.0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid identity annotation") from exc
    tolerance = 1e-6
    if annotation.x + annotation.width > width + tolerance:
        raise ValueError("annotation extends beyond video width")
    if annotation.y + annotation.height > height + tolerance:
        raise ValueError("annotation extends beyond video height")
    return annotation


def load_prediction_suggestions(
    path: Path,
    *,
    frame_count: int,
    width: int,
    height: int,
) -> dict[int, list[dict[str, Any]]]:
    """Load model boxes as untrusted suggestions, never as reviewed ground truth."""

    detections, _, _ = load_detection_jsonl(path)
    suggestions: dict[int, list[dict[str, Any]]] = {}
    for item in detections:
        if item.frame_id > frame_count:
            continue
        if item.x < 0.0 or item.y < 0.0:
            continue
        if item.x + item.width > width or item.y + item.height > height:
            continue
        suggestions.setdefault(item.frame_id, []).append(
            {
                "suggested_identity_id": item.track_id,
                "x": item.x,
                "y": item.y,
                "width": item.width,
                "height": item.height,
                "confidence": item.confidence,
                "source": "model_prediction",
            }
        )
    for frame in suggestions.values():
        frame.sort(key=lambda item: int(item["suggested_identity_id"]))
    return suggestions


class IdentityAnnotationStore:
    """Thread-safe reviewed-frame store with atomic JSON and MOT snapshots."""

    def __init__(
        self,
        output_dir: Path,
        *,
        source: Path,
        frame_count: int,
        width: int,
        height: int,
        fps: float,
    ) -> None:
        if frame_count < 1 or width < 1 or height < 1 or fps <= 0.0:
            raise ValueError("invalid video metadata")
        self.output_dir = output_dir.expanduser().resolve()
        self.source = source.expanduser().resolve()
        self.frame_count = frame_count
        self.width = width
        self.height = height
        self.fps = fps
        self.project_path = self.output_dir / "annotations.json"
        self.ground_truth_path = self.output_dir / "gt.txt"
        self._lock = RLock()
        self._reviewed_frames: set[int] = set()
        self._annotations: dict[int, tuple[IdentityAnnotation, ...]] = {}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.project_path.is_file():
            self._load()

    def frame(self, frame_id: int) -> Mapping[str, Any]:
        self._validate_frame_id(frame_id)
        with self._lock:
            return {
                "frame_id": frame_id,
                "reviewed": frame_id in self._reviewed_frames,
                "annotations": [
                    asdict(item) for item in self._annotations.get(frame_id, ())
                ],
            }

    def update_frame(
        self, frame_id: int, values: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        self._validate_frame_id(frame_id)
        annotations = tuple(
            annotation_from_mapping(value, width=self.width, height=self.height)
            for value in values
        )
        identities = [item.identity_id for item in annotations]
        if len(identities) != len(set(identities)):
            raise ValueError("an identity may appear only once in one frame")
        with self._lock:
            self._annotations[frame_id] = annotations
            self._reviewed_frames.add(frame_id)
            self._persist_locked()
            return self.frame(frame_id)

    def project(self) -> Mapping[str, Any]:
        with self._lock:
            identities = sorted(
                {
                    item.identity_id
                    for annotations in self._annotations.values()
                    for item in annotations
                }
            )
            return {
                "schema_version": 1,
                "video": {
                    "source": str(self.source),
                    "frame_count": self.frame_count,
                    "width": self.width,
                    "height": self.height,
                    "fps": self.fps,
                    "duration_s": self.frame_count / self.fps,
                },
                "reviewed_frames": sorted(self._reviewed_frames),
                "reviewed_frame_count": len(self._reviewed_frames),
                "annotation_count": sum(
                    len(items) for items in self._annotations.values()
                ),
                "identity_ids": identities,
                "outputs": {
                    "project_json": str(self.project_path),
                    "mot_ground_truth": str(self.ground_truth_path),
                },
            }

    def _load(self) -> None:
        try:
            payload = json.loads(self.project_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read annotation project: {self.project_path}") from exc
        video = payload.get("video", {})
        if Path(str(video.get("source", ""))).expanduser().resolve() != self.source:
            raise ValueError("annotation project belongs to a different source video")
        expected = (self.frame_count, self.width, self.height)
        actual = (
            int(video.get("frame_count", 0)),
            int(video.get("width", 0)),
            int(video.get("height", 0)),
        )
        if actual != expected:
            raise ValueError("annotation project video metadata does not match")
        reviewed = {int(frame_id) for frame_id in payload.get("reviewed_frames", [])}
        for frame_id in reviewed:
            self._validate_frame_id(frame_id)
        annotations: dict[int, tuple[IdentityAnnotation, ...]] = {}
        raw_annotations = payload.get("annotations", {})
        if not isinstance(raw_annotations, Mapping):
            raise ValueError("annotation project annotations must be an object")
        for raw_frame_id, values in raw_annotations.items():
            frame_id = int(raw_frame_id)
            self._validate_frame_id(frame_id)
            if not isinstance(values, list):
                raise ValueError("frame annotations must be a list")
            parsed = tuple(
                annotation_from_mapping(value, width=self.width, height=self.height)
                for value in values
            )
            identities = [item.identity_id for item in parsed]
            if len(identities) != len(set(identities)):
                raise ValueError("annotation project contains duplicate frame identities")
            annotations[frame_id] = parsed
        if not set(annotations).issubset(reviewed):
            raise ValueError("annotation boxes exist on an unreviewed frame")
        self._reviewed_frames = reviewed
        self._annotations = annotations

    def _validate_frame_id(self, frame_id: int) -> None:
        if not 1 <= frame_id <= self.frame_count:
            raise ValueError(f"frame_id must be in [1, {self.frame_count}]")

    def _persist_locked(self) -> None:
        project = {
            "schema_version": 1,
            "video": {
                "source": str(self.source),
                "frame_count": self.frame_count,
                "width": self.width,
                "height": self.height,
                "fps": self.fps,
            },
            "reviewed_frames": sorted(self._reviewed_frames),
            "annotations": {
                str(frame_id): [asdict(item) for item in self._annotations[frame_id]]
                for frame_id in sorted(self._annotations)
                if frame_id in self._reviewed_frames
            },
        }
        project_text = json.dumps(project, ensure_ascii=False, indent=2) + "\n"
        mot_lines = [
            item.mot_row(frame_id)
            for frame_id in sorted(self._reviewed_frames)
            for item in self._annotations.get(frame_id, ())
        ]
        self._atomic_write(self.project_path, project_text)
        self._atomic_write(
            self.ground_truth_path,
            "\n".join(mot_lines) + ("\n" if mot_lines else ""),
        )

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
