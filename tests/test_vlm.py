import unittest

from edge_vision.vlm import scene_analysis_from_mapping, scene_analysis_to_mapping


class VlmContractTests(unittest.TestCase):
    def test_structured_response_supplies_grounding_prompts(self) -> None:
        payload = {
            "summary": "Two people near a blue box.",
            "objects": [
                {"name": "person", "attributes": ["white shirt"], "count": 2},
                {"name": "box", "attributes": ["blue"], "count": 1},
            ],
            "grounding": {
                "user_query": "the person next to the blue box",
                "yolo_world_prompts": ["person", "blue box"],
                "required_attributes": [],
                "relation": "person next to blue box",
            },
        }
        analysis = scene_analysis_from_mapping(payload)
        self.assertEqual(analysis.grounding.yolo_world_prompts, ("person", "blue box"))
        self.assertEqual(analysis.grounding.relation, "person next to blue box")
        self.assertEqual(scene_analysis_to_mapping(analysis)["objects"][0]["count"], 2)

    def test_rejects_missing_grounding_prompts(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            scene_analysis_from_mapping(
                {
                    "summary": "A scene.",
                    "objects": [],
                    "grounding": {
                        "user_query": "find target",
                        "yolo_world_prompts": [],
                    },
                }
            )


if __name__ == "__main__":
    unittest.main()
