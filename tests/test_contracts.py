import unittest

from edge_vision.contracts import BoundingBox


class BoundingBoxTests(unittest.TestCase):
    def test_center_of_normalized_box(self) -> None:
        self.assertEqual(BoundingBox(0.1, 0.2, 0.5, 0.8).center, (0.3, 0.5))

    def test_invalid_box_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BoundingBox(0.8, 0.2, 0.5, 0.8)


if __name__ == "__main__":
    unittest.main()
