"""Filter person detections using a white-pixel ratio in the torso region."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

from edge_vision.contracts import TargetObservation


@dataclass(frozen=True)
class WhiteClothingConfig:
    saturation_max: int = 60
    value_min: int = 145
    ratio_threshold: float = 0.20
    minimum_box_height_ratio: float = 0.10
    roi_x_start: float = 0.18
    roi_x_end: float = 0.82
    roi_y_start: float = 0.18
    roi_y_end: float = 0.55

    def __post_init__(self) -> None:
        if not 0 <= self.saturation_max <= 255 or not 0 <= self.value_min <= 255:
            raise ValueError("HSV thresholds must be in [0, 255]")
        if not 0.0 <= self.ratio_threshold <= 1.0:
            raise ValueError("ratio_threshold must be in [0, 1]")
        if not 0.0 <= self.minimum_box_height_ratio <= 1.0:
            raise ValueError("minimum_box_height_ratio must be in [0, 1]")
        if not 0.0 <= self.roi_x_start < self.roi_x_end <= 1.0:
            raise ValueError("horizontal ROI fractions are invalid")
        if not 0.0 <= self.roi_y_start < self.roi_y_end <= 1.0:
            raise ValueError("vertical ROI fractions are invalid")


def torso_roi_bounds(
    observation: TargetObservation,
    width: int,
    height: int,
    config: WhiteClothingConfig,
) -> tuple[int, int, int, int]:
    box = observation.box
    x1 = round(box.x1 * width)
    y1 = round(box.y1 * height)
    x2 = round(box.x2 * width)
    y2 = round(box.y2 * height)
    box_width = x2 - x1
    box_height = y2 - y1
    return (
        max(0, x1 + round(config.roi_x_start * box_width)),
        max(0, y1 + round(config.roi_y_start * box_height)),
        min(width, x1 + round(config.roi_x_end * box_width)),
        min(height, y1 + round(config.roi_y_end * box_height)),
    )


class WhiteClothingFilter:
    """Wrap a detector and retain people whose torso region is predominantly white."""

    class_names = ("person wearing white clothes",)

    def __init__(self, detector: Any, config: WhiteClothingConfig | None = None) -> None:
        self.detector = detector
        self.config = config or WhiteClothingConfig()
        self.class_ids = tuple(getattr(detector, "class_ids", (0,)))

    def detect(
        self, frame: Any, prompts: Sequence[str] = ()
    ) -> Sequence[TargetObservation]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for white-clothing verification") from exc

        height, width = frame.shape[:2]
        verified = []
        for observation in self.detector.detect(frame, prompts):
            box_height_ratio = observation.box.y2 - observation.box.y1
            if box_height_ratio < self.config.minimum_box_height_ratio:
                continue
            x1, y1, x2, y2 = torso_roi_bounds(
                observation, width, height, self.config
            )
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            white_mask = (
                (hsv[:, :, 1] <= self.config.saturation_max)
                & (hsv[:, :, 2] >= self.config.value_min)
            )
            white_ratio = float(white_mask.mean())
            if white_ratio < self.config.ratio_threshold:
                continue
            verified.append(
                replace(
                    observation,
                    label=self.class_names[0],
                    color_confidence=white_ratio,
                    relation_verified=True,
                )
            )
        return verified

    def warmup(self, frame: Any) -> None:
        """Delegate model warmup without accidentally advancing tracker state."""

        warmup = getattr(self.detector, "warmup", None)
        if callable(warmup):
            warmup(frame)
        else:
            self.detector.detect(frame)
