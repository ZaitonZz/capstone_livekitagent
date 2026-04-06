import unittest

import numpy as np
import torch

from deepfakebench_effnet import (
    aggregate_fake_scores,
    crop_face_regions,
    extract_checkpoint_state_dict,
    logits_to_fake_probability,
    normalize_checkpoint_state_dict,
    preprocess_rgb_frame,
)


class DeepfakeBenchEfficientNetUtilsTest(unittest.TestCase):
    def test_extract_checkpoint_state_dict_prefers_state_dict_key(self) -> None:
        checkpoint = {
            "state_dict": {
                "backbone.last_layer.weight": torch.zeros((2, 1792)),
                "backbone.last_layer.bias": torch.zeros(2),
            }
        }

        extracted = extract_checkpoint_state_dict(checkpoint)

        self.assertIn("backbone.last_layer.weight", extracted)
        self.assertIn("backbone.last_layer.bias", extracted)

    def test_normalize_checkpoint_state_dict_strips_known_prefixes(self) -> None:
        raw_state_dict = {
            "module.model.backbone.last_layer.weight": torch.zeros((2, 1792)),
            "module.model.backbone.last_layer.bias": torch.zeros(2),
        }

        normalized = normalize_checkpoint_state_dict(raw_state_dict)

        self.assertIn("backbone.last_layer.weight", normalized)
        self.assertIn("backbone.last_layer.bias", normalized)

    def test_logits_to_fake_probability_uses_softmax_for_two_class_logits(self) -> None:
        logits = torch.tensor([[0.0, 1.0]])

        probability = logits_to_fake_probability(logits, fake_class_index=1)

        self.assertAlmostEqual(probability, 0.7311, places=4)

    def test_preprocess_rgb_frame_uses_expected_shape_and_range(self) -> None:
        rgb = torch.zeros((8, 8, 3), dtype=torch.uint8).numpy()

        tensor = preprocess_rgb_frame(rgb, input_size=4)

        self.assertEqual(tuple(tensor.shape), (1, 3, 4, 4))
        self.assertAlmostEqual(float(tensor.min().item()), -1.0, places=4)
        self.assertAlmostEqual(float(tensor.max().item()), -1.0, places=4)

    def test_crop_face_regions_returns_expanded_crop(self) -> None:
        rgb = np.zeros((100, 120, 3), dtype=np.uint8)
        boxes = [[30, 20, 70, 60, 0.9]]

        crops = crop_face_regions(rgb, boxes, margin_ratio=0.25, min_face_size=16)

        self.assertEqual(len(crops), 1)
        crop = crops[0]
        self.assertGreaterEqual(crop.shape[0], 40)
        self.assertGreaterEqual(crop.shape[1], 40)

    def test_crop_face_regions_filters_small_faces(self) -> None:
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes = [[10, 10, 20, 20, 0.99]]

        crops = crop_face_regions(rgb, boxes, margin_ratio=0.2, min_face_size=24)

        self.assertEqual(crops, [])

    def test_aggregate_fake_scores_respects_mode(self) -> None:
        scores = [0.2, 0.8, 0.6]

        self.assertAlmostEqual(aggregate_fake_scores(scores, mode="max"), 0.8, places=6)
        self.assertAlmostEqual(aggregate_fake_scores(scores, mode="mean"), 0.533333, places=5)
        self.assertAlmostEqual(aggregate_fake_scores(scores, mode="median"), 0.6, places=6)


if __name__ == "__main__":
    unittest.main()
