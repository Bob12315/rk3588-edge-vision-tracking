import unittest

from edge_vision.adapters.ultralytics_yolo import result_to_observations


class FakeValues:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def int(self):
        return self

    def tolist(self):
        return self.values


class FakeBoxes:
    xyxy = FakeValues([[10.0, 20.0, 50.0, 80.0]])
    conf = FakeValues([0.9])
    cls = FakeValues([0.0])
    id = FakeValues([42])


class FakeResult:
    boxes = FakeBoxes()
    names = {0: "person"}


class UltralyticsAdapterTests(unittest.TestCase):
    def test_tracking_result_preserves_track_id(self) -> None:
        observations = result_to_observations(FakeResult(), width=100, height=100)
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].track_id, 42)
        self.assertEqual(observations[0].label, "person")


if __name__ == "__main__":
    unittest.main()
