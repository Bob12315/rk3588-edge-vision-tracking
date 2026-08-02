"""Local Ollama vision-language adapter using only Python's standard HTTP client."""

from __future__ import annotations

import base64
import json
from time import perf_counter
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from edge_vision.contracts import VlmSceneAnalysis
from edge_vision.person_catalog import (
    ACTIONS,
    CARRIED_OBJECT_TYPES,
    CLOTHING_COLORS,
    HEADWEAR_TYPES,
    LOWER_TYPES,
    TERNARY_VALUES,
    UPPER_TYPES,
    VISIBILITY_VALUES,
    PersonAttributes,
    person_attributes_from_mapping,
)
from edge_vision.vlm import scene_analysis_from_mapping


VLM_SCENE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "objects": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "attributes": {"type": "array", "items": {"type": "string"}},
                    "count": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                },
                "required": ["name", "attributes", "count"],
                "additionalProperties": False,
            },
        },
        "grounding": {
            "type": "object",
            "properties": {
                "user_query": {"type": "string"},
                "yolo_world_prompts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 4,
                },
                "required_attributes": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "relation": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required": [
                "user_query",
                "yolo_world_prompts",
                "required_attributes",
                "relation",
            ],
            "additionalProperties": False,
        },
    },
    "required": ["summary", "objects", "grounding"],
    "additionalProperties": False,
}

PERSON_ATTRIBUTE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "upper_color": {"type": "string", "enum": list(CLOTHING_COLORS)},
        "upper_type": {"type": "string", "enum": list(UPPER_TYPES)},
        "lower_color": {"type": "string", "enum": list(CLOTHING_COLORS)},
        "lower_type": {"type": "string", "enum": list(LOWER_TYPES)},
        "headwear": {"type": "string", "enum": list(HEADWEAR_TYPES)},
        "headwear_color": {"type": "string", "enum": list(CLOTHING_COLORS)},
        "carried_object": {"type": "string", "enum": list(CARRIED_OBJECT_TYPES)},
        "carried_object_color": {"type": "string", "enum": list(CLOTHING_COLORS)},
        "safety_vest": {"type": "string", "enum": list(TERNARY_VALUES)},
        "action": {"type": "string", "enum": list(ACTIONS)},
        "visibility": {
            "type": "object",
            "properties": {
                "upper_body": {"type": "string", "enum": list(VISIBILITY_VALUES)},
                "lower_body": {"type": "string", "enum": list(VISIBILITY_VALUES)},
                "head": {"type": "string", "enum": list(VISIBILITY_VALUES)},
            },
            "required": ["upper_body", "lower_body", "head"],
            "additionalProperties": False,
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": [
        "upper_color",
        "upper_type",
        "lower_color",
        "lower_type",
        "headwear",
        "headwear_color",
        "carried_object",
        "carried_object_color",
        "safety_vest",
        "action",
        "visibility",
        "confidence",
    ],
    "additionalProperties": False,
}


Transport = Callable[[Request, float], bytes]


def _default_transport(request: Request, timeout_seconds: float) -> bytes:
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.read()


class OllamaVlm:
    """Qwen3-VL-compatible local scene analyzer backed by an Ollama REST server."""

    def __init__(
        self,
        model: str = "qwen3-vl:2b",
        *,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 300.0,
        max_new_tokens: int = 384,
        context_length: int = 4096,
        transport: Transport | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_new_tokens = max_new_tokens
        self.context_length = context_length
        self._transport = transport or _default_transport
        self.last_metrics: dict[str, Any] = {}

    def analyze(self, frame: Any, instruction: str) -> VlmSceneAnalysis:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required to encode VLM image input") from exc

        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("failed to encode VLM keyframe as JPEG")
        return self.analyze_image_bytes(encoded.tobytes(), instruction)

    def analyze_image_bytes(
        self, image_bytes: bytes, instruction: str
    ) -> VlmSceneAnalysis:
        if not image_bytes:
            raise ValueError("VLM image must not be empty")
        if not instruction.strip():
            raise ValueError("VLM instruction must not be empty")

        schema_text = json.dumps(VLM_SCENE_SCHEMA, ensure_ascii=False, separators=(",", ":"))
        prompt = (
            f"Task: {instruction.strip()}\n"
            "Inspect the image and return only the requested JSON. Use one objects entry per "
            "category, with at most eight categories. For yolo_world_prompts, use short English "
            "object nouns suitable for open-vocabulary detection. If the target is a person with "
            "any clothing attribute, the prompt must be exactly 'person'; put colors, clothing "
            "and relations in required_attributes or relation. Normalize a clothing color to "
            "one of white, black, gray, red, orange, yellow, green, cyan, blue, purple, pink, "
            "or brown. "
            f"The exact JSON schema is: {schema_text}"
        )
        analysis_payload = self._request_structured_image(
            image_bytes, prompt, VLM_SCENE_SCHEMA
        )
        return scene_analysis_from_mapping(analysis_payload)

    def analyze_person(self, frame: Any) -> PersonAttributes:
        """Extract controlled visible attributes from one detector-provided person crop."""

        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required to encode VLM image input") from exc
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError("failed to encode person crop as JPEG")
        return self.analyze_person_image_bytes(encoded.tobytes())

    def analyze_person_image_bytes(self, image_bytes: bytes) -> PersonAttributes:
        if not image_bytes:
            raise ValueError("person image must not be empty")
        schema_text = json.dumps(
            PERSON_ATTRIBUTE_SCHEMA, ensure_ascii=False, separators=(",", ":")
        )
        prompt = (
            "The image is a detector crop containing one person. Describe only clearly visible "
            "attributes of that person. Choose every value exactly from the supplied enums. "
            "Use unknown whenever an attribute is occluded, too small, ambiguous, or not visible; "
            "do not infer age, gender, identity, or hidden clothing. For headwear_color and "
            "carried_object_color use unknown when the corresponding object is none or unknown. "
            "Return only JSON. "
            f"The exact JSON schema is: {schema_text}"
        )
        payload = self._request_structured_image(
            image_bytes, prompt, PERSON_ATTRIBUTE_SCHEMA
        )
        return person_attributes_from_mapping(payload)

    def _request_structured_image(
        self,
        image_bytes: bytes,
        prompt: str,
        schema: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(image_bytes).decode("ascii")],
                }
            ],
            "stream": False,
            "think": False,
            "format": schema,
            "options": {
                "temperature": 0,
                "num_predict": self.max_new_tokens,
                "num_ctx": self.context_length,
            },
            "keep_alive": "5m",
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = perf_counter()
        try:
            response_bytes = self._transport(request, self.timeout_seconds)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(
                f"cannot reach local Ollama server at {self.base_url}; start it first"
            ) from exc

        wall_time_s = perf_counter() - started
        response = json.loads(response_bytes.decode("utf-8"))
        if not isinstance(response, Mapping):
            raise RuntimeError("Ollama response root is not an object")
        message = response.get("message")
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
            raise RuntimeError("Ollama response does not contain message.content")
        done_reason = response.get("done_reason")
        if done_reason == "length":
            raise RuntimeError(
                "local VLM response reached its token limit before completing valid JSON"
            )
        content = message["content"].strip()
        response_field = "content"
        if not content and isinstance(message.get("thinking"), str):
            content = message["thinking"].strip()
            response_field = "thinking"
        if not content:
            raise RuntimeError("Ollama returned an empty VLM response")
        analysis_payload = json.loads(content)
        if not isinstance(analysis_payload, Mapping):
            raise RuntimeError("VLM structured response root is not an object")

        eval_count = int(response.get("eval_count", 0) or 0)
        eval_duration_ns = int(response.get("eval_duration", 0) or 0)
        self.last_metrics = {
            "wall_time_s": wall_time_s,
            "total_duration_s": int(response.get("total_duration", 0) or 0) / 1e9,
            "load_duration_s": int(response.get("load_duration", 0) or 0) / 1e9,
            "prompt_eval_count": int(response.get("prompt_eval_count", 0) or 0),
            "eval_count": eval_count,
            "generation_tokens_per_s": (
                eval_count / (eval_duration_ns / 1e9) if eval_duration_ns else 0.0
            ),
            "response_field": response_field,
        }
        return analysis_payload
