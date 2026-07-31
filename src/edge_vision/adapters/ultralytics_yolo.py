"""Ultralytics YOLO adapter used by the x86 development baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from edge_vision.contracts import BoundingBox, TargetObservation


def result_to_observations(result: Any, width: int, height: int) -> list[TargetObservation]:
    if result.boxes is None:
        return []

    observations = []
    xyxy_values = result.boxes.xyxy.cpu().tolist()
    confidence_values = result.boxes.conf.cpu().tolist()
    class_values = result.boxes.cls.cpu().tolist()
    box_ids = getattr(result.boxes, "id", None)
    track_ids = (
        [None] * len(xyxy_values)
        if box_ids is None
        else box_ids.int().cpu().tolist()
    )
    for xyxy, confidence, class_value, track_id in zip(
        xyxy_values, confidence_values, class_values, track_ids
    ):
        x1, y1, x2, y2 = xyxy
        normalized = (
            max(0.0, min(1.0, x1 / width)),
            max(0.0, min(1.0, y1 / height)),
            max(0.0, min(1.0, x2 / width)),
            max(0.0, min(1.0, y2 / height)),
        )
        if normalized[0] >= normalized[2] or normalized[1] >= normalized[3]:
            continue
        class_id = int(class_value)
        observations.append(
            TargetObservation(
                label=str(result.names[class_id]),
                class_id=class_id,
                track_id=None if track_id is None else int(track_id),
                box=BoundingBox(*normalized),
                detector_confidence=float(confidence),
            )
        )
    return observations


class UltralyticsYoloDetector:
    def __init__(
        self,
        model: str | Path = "yolo11n.pt",
        *,
        class_ids: Sequence[int] = (0,),
        confidence: float = 0.25,
        image_size: int = 640,
        device: str = "cpu",
        tracker_config: str | Path | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is required for PC inference; install requirements-pc.txt"
            ) from exc

        self.model_path = str(model)
        self.class_ids = tuple(class_ids)
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.tracker_config = None if tracker_config is None else str(tracker_config)
        self._model = YOLO(self.model_path)
        self.class_names = tuple(str(self._model.names[item]) for item in self.class_ids)

    def detect(
        self, frame: Any, prompts: Sequence[str] = ()
    ) -> Sequence[TargetObservation]:
        del prompts  # Closed-set YOLO uses configured class IDs, not runtime text prompts.
        height, width = frame.shape[:2]
        arguments = {
            "source": frame,
            "classes": list(self.class_ids),
            "conf": self.confidence,
            "imgsz": self.image_size,
            "device": self.device,
            "verbose": False,
        }
        if self.tracker_config is None:
            result = self._model.predict(**arguments)[0]
        else:
            result = self._model.track(
                **arguments,
                persist=True,
                tracker=self.tracker_config,
            )[0]
        return result_to_observations(result, width, height)

    def warmup(self, frame: Any) -> None:
        """Warm model kernels without advancing the stateful tracker."""

        self._model.predict(
            source=frame,
            classes=list(self.class_ids),
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
