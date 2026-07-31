"""Strict VLM response parsing independent of any local or cloud provider."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import GroundingTask, SceneObject, VlmSceneAnalysis


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be an array of strings")
    items = tuple(str(item).strip() for item in value)
    if any(not item for item in items):
        raise ValueError(f"{field_name} must not contain empty strings")
    return items


def scene_analysis_from_mapping(payload: Mapping[str, Any]) -> VlmSceneAnalysis:
    """Validate a provider response before it can configure YOLO-World."""

    raw_objects = payload.get("objects")
    if not isinstance(raw_objects, Sequence) or isinstance(raw_objects, (str, bytes)):
        raise ValueError("objects must be an array")

    objects = []
    for index, item in enumerate(raw_objects):
        if not isinstance(item, Mapping):
            raise ValueError(f"objects[{index}] must be an object")
        raw_count = item.get("count")
        count = None if raw_count is None else int(raw_count)
        objects.append(
            SceneObject(
                name=str(item.get("name", "")).strip(),
                attributes=_string_tuple(
                    item.get("attributes", ()), f"objects[{index}].attributes"
                ),
                count=count,
            )
        )

    raw_grounding = payload.get("grounding")
    if not isinstance(raw_grounding, Mapping):
        raise ValueError("grounding must be an object")
    relation_value = raw_grounding.get("relation")
    relation = None if relation_value is None else str(relation_value).strip() or None
    grounding = GroundingTask(
        user_query=str(raw_grounding.get("user_query", "")).strip(),
        yolo_world_prompts=_string_tuple(
            raw_grounding.get("yolo_world_prompts", ()),
            "grounding.yolo_world_prompts",
        ),
        required_attributes=_string_tuple(
            raw_grounding.get("required_attributes", ()),
            "grounding.required_attributes",
        ),
        relation=relation,
    )
    return VlmSceneAnalysis(
        summary=str(payload.get("summary", "")).strip(),
        objects=tuple(objects),
        grounding=grounding,
    )


def load_scene_analysis(path: str | Path) -> VlmSceneAnalysis:
    """Load a saved VLM JSON response for deterministic replay and integration tests."""

    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("VLM response root must be an object")
    return scene_analysis_from_mapping(payload)


def scene_analysis_to_mapping(analysis: VlmSceneAnalysis) -> dict[str, Any]:
    """Serialize validated VLM output into run evidence."""

    return {
        "summary": analysis.summary,
        "objects": [
            {
                "name": item.name,
                "attributes": list(item.attributes),
                "count": item.count,
            }
            for item in analysis.objects
        ],
        "grounding": {
            "user_query": analysis.grounding.user_query,
            "yolo_world_prompts": list(analysis.grounding.yolo_world_prompts),
            "required_attributes": list(analysis.grounding.required_attributes),
            "relation": analysis.grounding.relation,
        },
    }
