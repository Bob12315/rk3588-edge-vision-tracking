"""Controlled per-person attributes and stable crop collection for the Web UI."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

from .contracts import TargetObservation


CLOTHING_COLORS = (
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
    "unknown",
)
UPPER_TYPES = (
    "shirt",
    "t-shirt",
    "jacket",
    "hoodie",
    "vest",
    "coat",
    "dress",
    "uniform",
    "unknown",
)
LOWER_TYPES = ("pants", "shorts", "jeans", "skirt", "dress", "unknown")
HEADWEAR_TYPES = ("none", "hat", "helmet", "unknown")
CARRIED_OBJECT_TYPES = ("none", "backpack", "handbag", "umbrella", "unknown")
ACTIONS = (
    "standing",
    "sitting",
    "walking",
    "running",
    "riding",
    "lying_down",
    "unknown",
)
TERNARY_VALUES = ("yes", "no", "unknown")
VISIBILITY_VALUES = ("clear", "partial", "hidden", "unknown")


@dataclass(frozen=True)
class PersonAttributes:
    """A VLM result constrained to attributes that downstream code can reason about."""

    upper_color: str
    upper_type: str
    lower_color: str
    lower_type: str
    headwear: str
    headwear_color: str
    carried_object: str
    carried_object_color: str
    safety_vest: str
    action: str
    upper_visibility: str
    lower_visibility: str
    head_visibility: str
    confidence: float

    def __post_init__(self) -> None:
        _require_choice("upper_color", self.upper_color, CLOTHING_COLORS)
        _require_choice("upper_type", self.upper_type, UPPER_TYPES)
        _require_choice("lower_color", self.lower_color, CLOTHING_COLORS)
        _require_choice("lower_type", self.lower_type, LOWER_TYPES)
        _require_choice("headwear", self.headwear, HEADWEAR_TYPES)
        _require_choice("headwear_color", self.headwear_color, CLOTHING_COLORS)
        _require_choice("carried_object", self.carried_object, CARRIED_OBJECT_TYPES)
        _require_choice("carried_object_color", self.carried_object_color, CLOTHING_COLORS)
        _require_choice("safety_vest", self.safety_vest, TERNARY_VALUES)
        _require_choice("action", self.action, ACTIONS)
        _require_choice("upper_visibility", self.upper_visibility, VISIBILITY_VALUES)
        _require_choice("lower_visibility", self.lower_visibility, VISIBILITY_VALUES)
        _require_choice("head_visibility", self.head_visibility, VISIBILITY_VALUES)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("person attribute confidence must be in [0, 1]")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "upper_color": self.upper_color,
            "upper_type": self.upper_type,
            "lower_color": self.lower_color,
            "lower_type": self.lower_type,
            "headwear": self.headwear,
            "headwear_color": self.headwear_color,
            "carried_object": self.carried_object,
            "carried_object_color": self.carried_object_color,
            "safety_vest": self.safety_vest,
            "action": self.action,
            "visibility": {
                "upper_body": self.upper_visibility,
                "lower_body": self.lower_visibility,
                "head": self.head_visibility,
            },
            "confidence": self.confidence,
        }


def person_attributes_from_mapping(payload: Mapping[str, Any]) -> PersonAttributes:
    """Parse the strict person schema without accepting invented free-form labels."""

    visibility = payload.get("visibility")
    if not isinstance(visibility, Mapping):
        raise ValueError("person visibility must be an object")
    try:
        confidence = float(payload["confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("person confidence must be a number") from exc
    return PersonAttributes(
        upper_color=_string_value(payload, "upper_color"),
        upper_type=_string_value(payload, "upper_type"),
        lower_color=_string_value(payload, "lower_color"),
        lower_type=_string_value(payload, "lower_type"),
        headwear=_string_value(payload, "headwear"),
        headwear_color=_string_value(payload, "headwear_color"),
        carried_object=_string_value(payload, "carried_object"),
        carried_object_color=_string_value(payload, "carried_object_color"),
        safety_vest=_string_value(payload, "safety_vest"),
        action=_string_value(payload, "action"),
        upper_visibility=_string_value(visibility, "upper_body"),
        lower_visibility=_string_value(visibility, "lower_body"),
        head_visibility=_string_value(visibility, "head"),
        confidence=confidence,
    )


def attribute_phrases(attributes: PersonAttributes) -> tuple[str, ...]:
    """Return short, visible English phrases suitable for a person card."""

    phrases: list[str] = []
    if attributes.upper_visibility != "hidden":
        upper = _garment_phrase(
            attributes.upper_color,
            attributes.upper_type,
            fallback="upper clothing",
        )
        if upper:
            phrases.append(upper)
    if attributes.lower_visibility != "hidden":
        lower = _garment_phrase(
            attributes.lower_color,
            attributes.lower_type,
            fallback="lower clothing",
        )
        if lower:
            phrases.append(lower)
    if attributes.headwear not in {"none", "unknown"}:
        phrases.append(
            _colored_object_phrase(attributes.headwear_color, attributes.headwear)
        )
    if attributes.carried_object not in {"none", "unknown"}:
        phrases.append(
            _colored_object_phrase(
                attributes.carried_object_color, attributes.carried_object
            )
        )
    if attributes.safety_vest == "yes" and "safety vest" not in phrases:
        phrases.append("safety vest")
    if attributes.action != "unknown":
        phrases.append(attributes.action.replace("_", " "))
    return tuple(dict.fromkeys(phrases))


def catalog_descriptions(
    people: Sequence[PersonAttributes],
) -> tuple[dict[str, Any], ...]:
    """Choose the shortest one- or two-attribute description unique in this scan."""

    phrase_sets = [attribute_phrases(item) for item in people]
    descriptions: list[dict[str, Any]] = []
    for index, phrases in enumerate(phrase_sets):
        unique: tuple[str, ...] = ()
        for size in (1, 2):
            for choice in combinations(phrases, size):
                if all(
                    other_index == index
                    or not set(choice).issubset(set(other_phrases))
                    for other_index, other_phrases in enumerate(phrase_sets)
                ):
                    unique = choice
                    break
            if unique:
                break
        recommended = "person"
        if unique:
            recommended = "person · " + " · ".join(unique)
        descriptions.append(
            {
                "phrases": list(phrases),
                "recommended_label": recommended,
                "unique_in_scan": bool(unique),
            }
        )
    return tuple(descriptions)


@dataclass
class CollectedPerson:
    track_id: int
    best_observation: TargetObservation
    latest_observation: TargetObservation
    best_crop: Any
    sample_count: int
    best_quality: float


class PersonScanCollector:
    """Accumulate stable tracked people and retain each track's clearest large crop."""

    def __init__(
        self,
        *,
        minimum_samples: int = 3,
        maximum_candidates: int = 6,
        minimum_confidence: float = 0.10,
        minimum_height_ratio: float = 0.10,
        crop_padding_ratio: float = 0.03,
    ) -> None:
        if minimum_samples < 1:
            raise ValueError("minimum_samples must be positive")
        if maximum_candidates < 1:
            raise ValueError("maximum_candidates must be positive")
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in [0, 1]")
        if not 0.0 <= minimum_height_ratio <= 1.0:
            raise ValueError("minimum_height_ratio must be in [0, 1]")
        if not 0.0 <= crop_padding_ratio <= 0.25:
            raise ValueError("crop_padding_ratio must be in [0, 0.25]")
        self.minimum_samples = minimum_samples
        self.maximum_candidates = maximum_candidates
        self.minimum_confidence = minimum_confidence
        self.minimum_height_ratio = minimum_height_ratio
        self.crop_padding_ratio = crop_padding_ratio
        self._people: dict[int, CollectedPerson] = {}

    def observe(
        self, frame: Any, observations: Sequence[TargetObservation]
    ) -> None:
        height, width = frame.shape[:2]
        for observation in observations:
            if observation.track_id is None:
                continue
            box = observation.box
            box_height = box.y2 - box.y1
            if observation.detector_confidence < self.minimum_confidence:
                continue
            if box_height < self.minimum_height_ratio:
                continue
            crop = self._crop(frame, observation, width, height)
            if crop is None or crop.size == 0:
                continue
            area = (box.x2 - box.x1) * box_height
            quality = observation.detector_confidence + min(1.0, area * 4.0) * 0.25
            existing = self._people.get(observation.track_id)
            if existing is None:
                self._people[observation.track_id] = CollectedPerson(
                    track_id=observation.track_id,
                    best_observation=observation,
                    latest_observation=observation,
                    best_crop=crop.copy(),
                    sample_count=1,
                    best_quality=quality,
                )
                continue
            existing.latest_observation = observation
            existing.sample_count += 1
            if quality > existing.best_quality:
                existing.best_observation = observation
                existing.best_crop = crop.copy()
                existing.best_quality = quality

    def candidates(self) -> tuple[CollectedPerson, ...]:
        stable = [
            item
            for item in self._people.values()
            if item.sample_count >= self.minimum_samples
        ]
        selected = sorted(
            stable,
            key=lambda item: (item.sample_count, item.best_quality),
            reverse=True,
        )[: self.maximum_candidates]
        return tuple(sorted(selected, key=lambda item: item.latest_observation.box.center[0]))

    def _crop(
        self,
        frame: Any,
        observation: TargetObservation,
        width: int,
        height: int,
    ) -> Any | None:
        box = observation.box
        pad_x = (box.x2 - box.x1) * self.crop_padding_ratio
        pad_y = (box.y2 - box.y1) * self.crop_padding_ratio
        x1 = max(0, round((box.x1 - pad_x) * width))
        y1 = max(0, round((box.y1 - pad_y) * height))
        x2 = min(width, round((box.x2 + pad_x) * width))
        y2 = min(height, round((box.y2 + pad_y) * height))
        if x1 >= x2 or y1 >= y2:
            return None
        return frame[y1:y2, x1:x2]


def _string_value(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"person attribute {key} must be a non-empty string")
    return value


def _require_choice(name: str, value: str, choices: Sequence[str]) -> None:
    if value not in choices:
        raise ValueError(f"unsupported {name}: {value}")


def _garment_phrase(color: str, garment: str, *, fallback: str) -> str:
    known_color = color != "unknown"
    known_garment = garment != "unknown"
    if not known_color and not known_garment:
        return ""
    return " ".join(
        item
        for item in (color if known_color else "", garment if known_garment else fallback)
        if item
    )


def _colored_object_phrase(color: str, name: str) -> str:
    return f"{color} {name}" if color != "unknown" else name
