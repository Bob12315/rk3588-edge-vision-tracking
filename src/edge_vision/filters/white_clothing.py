"""Backward-compatible white-clothing wrapper around the generic color filter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from edge_vision.contracts import TargetObservation

from .clothing_color import (
    ClothingColorConfig,
    ClothingColorFilter,
    clothing_roi_bounds,
)


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
    return clothing_roi_bounds(
        observation,
        width,
        height,
        ClothingColorConfig(
            target_color="white",
            score_threshold=config.ratio_threshold,
            minimum_box_height_ratio=config.minimum_box_height_ratio,
            roi_x_start=config.roi_x_start,
            roi_x_end=config.roi_x_end,
            roi_y_start=config.roi_y_start,
            roi_y_end=config.roi_y_end,
            white_saturation_max=config.saturation_max,
            white_value_min=config.value_min,
        ),
    )


class WhiteClothingFilter(ClothingColorFilter):
    """Compatibility name for the temporally stabilized white-clothing filter."""

    def __init__(self, detector: Any, config: WhiteClothingConfig | None = None) -> None:
        white = config or WhiteClothingConfig()
        super().__init__(
            detector,
            ClothingColorConfig(
                target_color="white",
                score_threshold=white.ratio_threshold,
                minimum_box_height_ratio=white.minimum_box_height_ratio,
                roi_x_start=white.roi_x_start,
                roi_x_end=white.roi_x_end,
                roi_y_start=white.roi_y_start,
                roi_y_end=white.roi_y_end,
                white_saturation_max=white.saturation_max,
                white_value_min=white.value_min,
            ),
        )
