import unittest

import torch

from deepfakebench_effnet import (
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


if __name__ == "__main__":
    unittest.main()
