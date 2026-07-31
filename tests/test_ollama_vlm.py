import json
import unittest

from edge_vision.adapters.ollama_vlm import OllamaVlm


class OllamaVlmTests(unittest.TestCase):
    def test_parses_structured_local_response_and_records_metrics(self) -> None:
        captured = {}

        def transport(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            content = {
                "summary": "People standing on a sports field.",
                "objects": [
                    {
                        "name": "person",
                        "attributes": ["white clothing", "red clothing"],
                        "count": None,
                    }
                ],
                "grounding": {
                    "user_query": "find people wearing white clothes",
                    "yolo_world_prompts": ["person"],
                    "required_attributes": ["white clothing"],
                    "relation": None,
                },
            }
            return json.dumps(
                {
                    "message": {"role": "assistant", "content": json.dumps(content)},
                    "total_duration": 2_000_000_000,
                    "load_duration": 500_000_000,
                    "prompt_eval_count": 100,
                    "eval_count": 20,
                    "eval_duration": 1_000_000_000,
                }
            ).encode("utf-8")

        vlm = OllamaVlm(transport=transport, timeout_seconds=12)
        analysis = vlm.analyze_image_bytes(b"jpeg-data", "find white clothes")

        self.assertEqual(analysis.grounding.yolo_world_prompts, ("person",))
        self.assertEqual(analysis.grounding.required_attributes, ("white clothing",))
        self.assertEqual(vlm.last_metrics["generation_tokens_per_s"], 20.0)
        self.assertEqual(captured["timeout"], 12)
        self.assertFalse(captured["payload"]["stream"])
        self.assertEqual(captured["payload"]["format"]["type"], "object")
        self.assertTrue(captured["payload"]["messages"][0]["images"])

    def test_rejects_empty_image(self) -> None:
        vlm = OllamaVlm(transport=lambda request, timeout: b"{}")
        with self.assertRaisesRegex(ValueError, "image"):
            vlm.analyze_image_bytes(b"", "inspect")

    def test_uses_completed_thinking_field_when_content_is_empty(self) -> None:
        content = {
            "summary": "One person.",
            "objects": [{"name": "person", "attributes": [], "count": 1}],
            "grounding": {
                "user_query": "find person",
                "yolo_world_prompts": ["person"],
                "required_attributes": [],
                "relation": None,
            },
        }

        def transport(request, timeout):
            return json.dumps(
                {
                    "done_reason": "stop",
                    "message": {"content": "", "thinking": json.dumps(content)},
                }
            ).encode("utf-8")

        vlm = OllamaVlm(transport=transport)
        analysis = vlm.analyze_image_bytes(b"jpeg", "find person")
        self.assertEqual(analysis.objects[0].name, "person")
        self.assertEqual(vlm.last_metrics["response_field"], "thinking")


if __name__ == "__main__":
    unittest.main()
