"""Lightweight clothing-color verification with per-track temporal smoothing."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Mapping, Sequence

from edge_vision.contracts import TargetObservation


SUPPORTED_COLORS = (
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
)

_COLOR_ALIASES: Mapping[str, tuple[str, ...]] = {
    "white": ("white", "白色", "白衣", "白"),
    "black": ("black", "黑色", "黑衣", "黑"),
    "gray": ("gray", "grey", "灰色", "灰衣", "灰"),
    "red": ("red", "红色", "红衣", "红"),
    "orange": ("orange", "橙色", "橘色", "橙", "橘"),
    "yellow": ("yellow", "黄色", "黄衣", "黄"),
    "green": ("green", "绿色", "绿衣", "绿"),
    "cyan": ("cyan", "青色", "青衣", "青"),
    "blue": ("blue", "蓝色", "蓝衣", "蓝"),
    "purple": ("purple", "violet", "紫色", "紫衣", "紫"),
    "pink": ("pink", "粉色", "粉红色", "粉衣", "粉"),
    "brown": ("brown", "棕色", "褐色", "棕", "褐"),
}

_CLOTHING_PATTERN = re.compile(
    r"\b(?:clothes?|clothing|shirts?|jerseys?|jackets?|coats?|uniforms?|tops?|"
    r"trousers?|pants|shorts|dresses|wears?|wearing|dressed)\b"
)
_CLOTHING_CHARACTERS = ("衣", "服", "裤", "裙", "穿")

_COLOR_AFFINITY: Mapping[str, Mapping[str, float]] = {
    "white": {"white": 1.0, "gray": 0.20},
    "black": {"black": 1.0, "gray": 0.15},
    "gray": {"gray": 1.0, "white": 0.20, "black": 0.15},
    "red": {"red": 1.0, "orange": 0.10, "pink": 0.10, "brown": 0.10},
    "orange": {"orange": 1.0, "brown": 0.35, "yellow": 0.10},
    "yellow": {"yellow": 1.0, "orange": 0.10},
    "green": {"green": 1.0, "cyan": 0.10},
    "cyan": {"cyan": 1.0, "blue": 0.15, "green": 0.10},
    "blue": {"blue": 1.0, "cyan": 0.15, "purple": 0.10},
    "purple": {"purple": 1.0, "pink": 0.15, "blue": 0.10},
    "pink": {"pink": 1.0, "purple": 0.15, "red": 0.10},
    "brown": {"brown": 1.0, "orange": 0.80, "red": 0.15},
}


def normalize_color_name(value: str) -> str | None:
    """Return a supported canonical color found in free-form text."""

    text = value.casefold().strip()
    if not text:
        return None
    matches = []
    for color, aliases in _COLOR_ALIASES.items():
        for alias in aliases:
            if alias.isascii():
                match = re.search(rf"\b{re.escape(alias)}\b", text)
                if match is not None:
                    matches.append((match.start(), -len(alias), color))
            else:
                position = text.find(alias)
                if position >= 0:
                    matches.append((position, -len(alias), color))
    return min(matches, default=(0, 0, None))[2]


def clothing_color_from_text(value: str) -> str | None:
    """Extract a color only when the text describes clothing or something worn."""

    text = value.casefold()
    if _CLOTHING_PATTERN.search(text) is None and not any(
        marker in text for marker in _CLOTHING_CHARACTERS
    ):
        return None
    return normalize_color_name(text)


@dataclass(frozen=True)
class ClothingColorConfig:
    target_color: str
    score_threshold: float = 0.20
    minimum_detector_confidence: float = 0.10
    minimum_box_height_ratio: float = 0.08
    roi_x_start: float = 0.20
    roi_x_end: float = 0.80
    roi_y_start: float = 0.20
    roi_y_end: float = 0.58
    white_saturation_max: int = 70
    white_value_min: int = 145
    temporal_alpha: float = 0.45
    temporal_min_samples: int = 3
    stale_after_frames: int = 45
    minimum_dominance_ratio: float = 0.85
    adaptive_upper_roi: bool = True
    upper_roi_x_start: float = 0.15
    upper_roi_x_end: float = 0.85
    upper_roi_y_start: float = 0.10
    upper_roi_y_end: float = 0.43

    def __post_init__(self) -> None:
        if self.target_color not in SUPPORTED_COLORS:
            raise ValueError(f"unsupported clothing color: {self.target_color}")
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be in [0, 1]")
        if not 0.0 <= self.minimum_detector_confidence <= 1.0:
            raise ValueError("minimum_detector_confidence must be in [0, 1]")
        if not 0.0 <= self.minimum_box_height_ratio <= 1.0:
            raise ValueError("minimum_box_height_ratio must be in [0, 1]")
        if not 0.0 <= self.roi_x_start < self.roi_x_end <= 1.0:
            raise ValueError("horizontal ROI fractions are invalid")
        if not 0.0 <= self.roi_y_start < self.roi_y_end <= 1.0:
            raise ValueError("vertical ROI fractions are invalid")
        if not 0 <= self.white_saturation_max <= 255:
            raise ValueError("white_saturation_max must be in [0, 255]")
        if not 0 <= self.white_value_min <= 255:
            raise ValueError("white_value_min must be in [0, 255]")
        if not 0.0 < self.temporal_alpha <= 1.0:
            raise ValueError("temporal_alpha must be in (0, 1]")
        if self.temporal_min_samples < 1:
            raise ValueError("temporal_min_samples must be positive")
        if self.stale_after_frames < 1:
            raise ValueError("stale_after_frames must be positive")
        if not 0.0 <= self.minimum_dominance_ratio <= 1.0:
            raise ValueError("minimum_dominance_ratio must be in [0, 1]")
        if not 0.0 <= self.upper_roi_x_start < self.upper_roi_x_end <= 1.0:
            raise ValueError("upper horizontal ROI fractions are invalid")
        if not 0.0 <= self.upper_roi_y_start < self.upper_roi_y_end <= 1.0:
            raise ValueError("upper vertical ROI fractions are invalid")


@dataclass(frozen=True)
class ClothingColorEstimate:
    label: str
    confidence: float
    scores: Mapping[str, float]


@dataclass
class _TrackColorState:
    scores: dict[str, float]
    samples: int
    last_frame: int
    confirmations: int = 0


def matching_color_score(scores: Mapping[str, float], target_color: str) -> float:
    """Score a requested semantic color while retaining controlled neighbor affinity."""

    if target_color not in _COLOR_AFFINITY:
        raise ValueError(f"unsupported clothing color: {target_color}")
    return min(
        1.0,
        sum(
            float(scores.get(source_color, 0.0)) * weight
            for source_color, weight in _COLOR_AFFINITY[target_color].items()
        ),
    )


@lru_cache(maxsize=1)
def _lab_color_prototypes() -> tuple[Any, tuple[str, ...]]:
    """Build several illumination variants for each color in OpenCV Lab space."""

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("OpenCV and NumPy are required for clothing color") from exc

    samples = (
        ("white", (245, 245, 245)),
        ("white", (190, 190, 190)),
        ("black", (15, 15, 15)),
        ("black", (45, 45, 45)),
        ("gray", (90, 90, 90)),
        ("gray", (135, 135, 135)),
        ("red", (0, 0, 220)),
        ("red", (0, 0, 140)),
        ("orange", (0, 130, 245)),
        ("orange", (0, 85, 190)),
        ("yellow", (0, 220, 220)),
        ("yellow", (0, 150, 150)),
        ("green", (0, 190, 0)),
        ("green", (0, 110, 0)),
        ("cyan", (210, 210, 0)),
        ("cyan", (130, 130, 0)),
        ("blue", (220, 0, 0)),
        ("blue", (130, 0, 0)),
        ("purple", (170, 50, 150)),
        ("purple", (100, 25, 90)),
        ("pink", (180, 105, 235)),
        ("pink", (135, 80, 190)),
        ("brown", (35, 80, 145)),
        ("brown", (25, 55, 100)),
    )
    bgr = np.asarray([[sample] for _, sample in samples], dtype=np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return lab, tuple(color for color, _ in samples)


def clothing_roi_bounds(
    observation: TargetObservation,
    width: int,
    height: int,
    config: ClothingColorConfig,
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


def upper_clothing_roi_bounds(
    observation: TargetObservation,
    width: int,
    height: int,
    config: ClothingColorConfig,
) -> tuple[int, int, int, int]:
    box = observation.box
    x1 = round(box.x1 * width)
    y1 = round(box.y1 * height)
    x2 = round(box.x2 * width)
    y2 = round(box.y2 * height)
    box_width = x2 - x1
    box_height = y2 - y1
    return (
        max(0, x1 + round(config.upper_roi_x_start * box_width)),
        max(0, y1 + round(config.upper_roi_y_start * box_height)),
        min(width, x1 + round(config.upper_roi_x_end * box_width)),
        min(height, y1 + round(config.upper_roi_y_end * box_height)),
    )


def estimate_clothing_color(
    roi: Any,
    *,
    white_saturation_max: int = 70,
    white_value_min: int = 145,
) -> ClothingColorEstimate:
    """Classify a BGR torso crop using weighted HSV and Lab evidence.

    An elliptical center weighting suppresses box-edge background without adding a
    second neural network. The returned scores are visible-pixel ratios and therefore
    remain inspectable and cheap enough to run once per tracked box.
    """

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV and NumPy are required for clothing color") from exc

    if roi is None or getattr(roi, "size", 0) == 0:
        raise ValueError("clothing ROI must not be empty")
    roi_height, roi_width = roi.shape[:2]
    scale = min(1.0, 32.0 / max(roi_height, roi_width))
    sampled = (
        cv2.resize(
            roi,
            (max(2, round(roi_width * scale)), max(2, round(roi_height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        if scale < 1.0
        else roi
    )
    blurred = cv2.GaussianBlur(sampled, (3, 3), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    return estimate_clothing_color_spaces(
        hsv,
        lab,
        white_saturation_max=white_saturation_max,
        white_value_min=white_value_min,
    )


def estimate_clothing_color_spaces(
    hsv: Any,
    lab: Any,
    *,
    white_saturation_max: int = 70,
    white_value_min: int = 145,
) -> ClothingColorEstimate:
    """Classify aligned HSV/Lab crops, allowing one frame conversion for many boxes."""

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("OpenCV and NumPy are required for clothing color") from exc

    if hsv is None or lab is None or getattr(hsv, "size", 0) == 0:
        raise ValueError("clothing color-space ROI must not be empty")
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    roi_height, roi_width = hue.shape
    yy, xx = np.mgrid[0:roi_height, 0:roi_width]
    nx = (xx - (roi_width - 1) / 2.0) / max(roi_width / 2.0, 1.0)
    ny = (yy - (roi_height - 1) / 2.0) / max(roi_height / 2.0, 1.0)
    ellipse = (nx * nx + ny * ny) <= 1.0
    weights = np.exp(-(1.4 * nx * nx + 0.55 * ny * ny)) * ellipse
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        raise ValueError("clothing ROI is too small")

    black = value < 65
    white = (
        (saturation <= white_saturation_max)
        & (value >= white_value_min)
    )
    gray = (
        (saturation <= max(80, white_saturation_max + 10))
        & (value >= 65)
        & ~white
    )
    chromatic = (saturation > 55) & (value >= 55) & ~black
    brown = chromatic & (hue >= 5) & (hue < 22) & (value < 150)
    masks = {
        "white": white,
        "black": black,
        "gray": gray,
        "red": chromatic & ~brown & ((hue < 9) | (hue >= 171)),
        "orange": chromatic & ~brown & (hue >= 9) & (hue < 22),
        "yellow": chromatic & (hue >= 22) & (hue < 38),
        "green": chromatic & (hue >= 38) & (hue < 84),
        "cyan": chromatic & (hue >= 84) & (hue < 100),
        "blue": chromatic & (hue >= 100) & (hue < 130),
        "purple": chromatic & (hue >= 130) & (hue < 154),
        "pink": chromatic & (hue >= 154) & (hue < 171),
        "brown": brown,
    }
    hsv_scores = {
        color: float(weights[mask].sum()) / total_weight
        for color, mask in masks.items()
    }
    ranked_hsv = sorted(hsv_scores.values(), reverse=True)
    if ranked_hsv[0] >= 0.55 and ranked_hsv[0] - ranked_hsv[1] >= 0.18:
        label = max(SUPPORTED_COLORS, key=lambda item: hsv_scores[item])
        return ClothingColorEstimate(label, hsv_scores[label], hsv_scores)

    lab_width = min(12, roi_width)
    lab_height = min(12, roi_height)
    lab_source = cv2.resize(
        lab,
        (lab_width, lab_height),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    lab_yy, lab_xx = np.mgrid[0:lab_height, 0:lab_width]
    lab_nx = (lab_xx - (lab_width - 1) / 2.0) / max(lab_width / 2.0, 1.0)
    lab_ny = (lab_yy - (lab_height - 1) / 2.0) / max(lab_height / 2.0, 1.0)
    lab_ellipse = (lab_nx * lab_nx + lab_ny * lab_ny) <= 1.0
    lab_weights = (
        np.exp(-(1.4 * lab_nx * lab_nx + 0.55 * lab_ny * lab_ny))
        * lab_ellipse
    )
    lab_total_weight = float(lab_weights.sum())
    prototypes, prototype_labels = _lab_color_prototypes()
    delta = lab_source[:, :, None, :] - prototypes[None, None, :, :]
    delta[:, :, :, 0] *= 0.55
    nearest = np.argmin(np.sum(delta * delta, axis=3), axis=2)
    lab_scores = {
        color: float(
            lab_weights[
                np.isin(
                    nearest,
                    [
                        index
                        for index, prototype_color in enumerate(prototype_labels)
                        if prototype_color == color
                    ],
                )
            ].sum()
        )
        / lab_total_weight
        for color in SUPPORTED_COLORS
    }
    scores = {
        color: 0.70 * hsv_scores[color] + 0.30 * lab_scores[color]
        for color in SUPPORTED_COLORS
    }
    label = max(SUPPORTED_COLORS, key=lambda item: scores[item])
    return ClothingColorEstimate(label, scores[label], scores)


class ClothingColorFilter:
    """Retain detections whose torso color matches the configured target color."""

    def __init__(self, detector: Any, config: ClothingColorConfig) -> None:
        self.detector = detector
        self.config = config
        self.class_ids = tuple(getattr(detector, "class_ids", (0,)))
        self.class_names = (
            f"person wearing {self.config.target_color} clothing",
        )
        self._frame_index = 0
        self._track_states: dict[int, _TrackColorState] = {}

    def detect(
        self, frame: Any, prompts: Sequence[str] = ()
    ) -> Sequence[TargetObservation]:
        height, width = frame.shape[:2]
        self._frame_index += 1
        verified = []
        observations = tuple(self.detector.detect(frame, prompts))
        if not observations:
            self._expire_stale_tracks()
            return verified
        for observation in observations:
            if observation.detector_confidence < self.config.minimum_detector_confidence:
                continue
            if observation.box.y2 - observation.box.y1 < self.config.minimum_box_height_ratio:
                continue
            bounds = [clothing_roi_bounds(observation, width, height, self.config)]
            if self.config.adaptive_upper_roi:
                bounds.append(
                    upper_clothing_roi_bounds(observation, width, height, self.config)
                )
            estimates = []
            for x1, y1, x2, y2 in bounds:
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                estimates.append(
                    estimate_clothing_color(
                        roi,
                        white_saturation_max=self.config.white_saturation_max,
                        white_value_min=self.config.white_value_min,
                    )
                )
            if not estimates:
                continue
            estimate = max(
                estimates,
                key=lambda item: matching_color_score(
                    item.scores, self.config.target_color
                ),
            )
            scores, state = self._smoothed_scores(observation.track_id, estimate.scores)
            target_score = matching_color_score(scores, self.config.target_color)
            dominant_score = max(
                matching_color_score(scores, color) for color in SUPPORTED_COLORS
            )
            color_matches = (
                target_score >= self.config.score_threshold
                and target_score
                >= dominant_score * self.config.minimum_dominance_ratio
            )
            if state is not None:
                state.confirmations = state.confirmations + 1 if color_matches else 0
                confirmations = state.confirmations
            else:
                confirmations = self.config.temporal_min_samples if color_matches else 0
            if confirmations < self.config.temporal_min_samples:
                continue
            verified.append(
                replace(
                    observation,
                    label=self.class_names[0],
                    color_label=self.config.target_color,
                    color_confidence=min(1.0, target_score),
                    relation_verified=True,
                )
            )
        self._expire_stale_tracks()
        return verified

    def _smoothed_scores(
        self, track_id: int | None, scores: Mapping[str, float]
    ) -> tuple[dict[str, float], _TrackColorState | None]:
        current = {color: float(scores[color]) for color in SUPPORTED_COLORS}
        if track_id is None:
            return current, None
        state = self._track_states.get(track_id)
        if state is None:
            state = _TrackColorState(
                current, 1, self._frame_index
            )
            self._track_states[track_id] = state
            return current, state
        alpha = self.config.temporal_alpha
        state.scores = {
            color: alpha * current[color] + (1.0 - alpha) * state.scores[color]
            for color in SUPPORTED_COLORS
        }
        state.samples += 1
        state.last_frame = self._frame_index
        return dict(state.scores), state

    def _expire_stale_tracks(self) -> None:
        oldest = self._frame_index - self.config.stale_after_frames
        stale = [
            track_id
            for track_id, state in self._track_states.items()
            if state.last_frame < oldest
        ]
        for track_id in stale:
            del self._track_states[track_id]

    def warmup(self, frame: Any) -> None:
        """Warm the detector without updating either tracker or color history."""

        warmup = getattr(self.detector, "warmup", None)
        if callable(warmup):
            warmup(frame)
        else:
            self.detector.detect(frame)
