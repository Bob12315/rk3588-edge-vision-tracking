import unittest

from edge_vision.contracts import GroundingTask, SceneObject, VlmSceneAnalysis
from edge_vision.realtime import VlmCoordinator


class FakeVlm:
    def __init__(self) -> None:
        self.calls = 0
        self.last_metrics = {"generation_tokens_per_s": 10.0}

    def analyze(self, frame, instruction):
        self.calls += 1
        return VlmSceneAnalysis(
            summary="people are visible",
            objects=(SceneObject("person", ("white clothes",), 1),),
            grounding=GroundingTask(
                user_query="white-clothed person",
                yolo_world_prompts=("person wearing white clothes",),
                required_attributes=("white clothing",),
            ),
        )


class FailingVlm:
    def analyze(self, frame, instruction):
        raise TimeoutError("model deadline exceeded")


class VlmCoordinatorTests(unittest.TestCase):
    def test_initial_and_cooldown_bounded_loss_trigger(self) -> None:
        provider = FakeVlm()
        coordinator = VlmCoordinator(
            provider,
            "find white clothes",
            cooldown_s=10.0,
            max_triggers=2,
        )
        initial = coordinator.analyze(
            object(), cause="initial", frame_id=0, timestamp_s=0.0, force=True
        )
        self.assertEqual(initial.grounding.yolo_world_prompts, ("person",))
        self.assertIsNone(
            coordinator.analyze(
                object(), cause="lost", frame_id=5, timestamp_s=5.0
            )
        )
        self.assertIsNotNone(
            coordinator.analyze(
                object(), cause="lost", frame_id=10, timestamp_s=10.0
            )
        )
        self.assertIsNone(
            coordinator.analyze(
                object(), cause="lost", frame_id=20, timestamp_s=20.0
            )
        )
        self.assertEqual(provider.calls, 2)
        self.assertEqual(len(coordinator.events), 2)

    def test_non_initial_vlm_failure_is_recorded_without_stopping_runtime(self) -> None:
        coordinator = VlmCoordinator(
            FailingVlm(), "find target", cooldown_s=0.0, max_triggers=1
        )
        result = coordinator.analyze(
            object(), cause="target_lost", frame_id=20, timestamp_s=2.0
        )
        self.assertIsNone(result)
        self.assertIn("TimeoutError", coordinator.events[0]["error"])

    def test_initial_vlm_failure_can_be_fatal(self) -> None:
        coordinator = VlmCoordinator(
            FailingVlm(), "find target", cooldown_s=0.0, max_triggers=1
        )
        with self.assertRaises(TimeoutError):
            coordinator.analyze(
                object(),
                cause="initial",
                frame_id=0,
                timestamp_s=0.0,
                force=True,
                raise_on_error=True,
            )


if __name__ == "__main__":
    unittest.main()
