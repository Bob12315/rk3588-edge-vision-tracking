"""Ultralytics YOLO-World adapter with runtime text prompts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from edge_vision.contracts import TargetObservation

from .ultralytics_yolo import result_to_observations


class UltralyticsYoloWorldDetector:
    def __init__(
        self,
        model: str | Path = "yolov8s-worldv2.pt",
        *,
        prompts: Sequence[str] = ("person",),
        confidence: float = 0.10,
        image_size: int = 640,
        device: str = "cpu",
    ) -> None:
        try:
            from ultralytics import YOLOWorld
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is required for YOLO-World; install requirements-pc.txt"
            ) from exc

        self.model_path = str(model)
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self._model = YOLOWorld(self.model_path)
        self._prompts: tuple[str, ...] = ()
        self.set_prompts(prompts)

    @property
    def prompts(self) -> tuple[str, ...]:
        return self._prompts

    @property
    def class_ids(self) -> tuple[int, ...]:
        return tuple(range(len(self._prompts)))

    @property
    def class_names(self) -> tuple[str, ...]:
        return self._prompts

    def set_prompts(self, prompts: Sequence[str]) -> None:
        normalized = tuple(prompt.strip() for prompt in prompts if prompt.strip())
        if not normalized:
            raise ValueError("YOLO-World requires at least one non-empty text prompt")
        if normalized == self._prompts:
            return
        self._model.set_classes(list(normalized))
        self._prompts = normalized

    def detect(
        self, frame: Any, prompts: Sequence[str] = ()
    ) -> Sequence[TargetObservation]:
        if prompts:
            self.set_prompts(prompts)
        height, width = frame.shape[:2]
        result = self._model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )[0]
        return result_to_observations(result, width, height)
