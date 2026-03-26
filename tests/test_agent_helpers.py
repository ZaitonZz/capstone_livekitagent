import unittest

from agent import build_saved_frame_filename, build_scan_result_payload, determine_deepfake_result


class AgentHelperFunctionsTest(unittest.TestCase):
    def test_build_saved_frame_filename_uses_sequential_timestamp_pattern(self) -> None:
        filename = build_saved_frame_filename(consultation_id=27, frame_number=12, timestamp_us=1234567890)

        self.assertEqual(filename, "consultation-27_frame-000012_1234567890.jpg")

    def test_build_saved_frame_filename_handles_missing_consultation(self) -> None:
        filename = build_saved_frame_filename(consultation_id=None, frame_number=3, timestamp_us=999)

        self.assertEqual(filename, "consultation-unknown_frame-000003_999.jpg")

    def test_determine_deepfake_result_classifies_fake(self) -> None:
        result, confidence = determine_deepfake_result(0.9)

        self.assertEqual(result, "fake")
        self.assertAlmostEqual(confidence, 0.9, places=4)

    def test_determine_deepfake_result_classifies_real(self) -> None:
        result, confidence = determine_deepfake_result(0.1)

        self.assertEqual(result, "real")
        self.assertAlmostEqual(confidence, 0.9, places=4)

    def test_build_scan_result_payload_uses_expected_keys(self) -> None:
        payload = build_scan_result_payload(
            consultation_id=51,
            deepfake_result={
                "result": "fake",
                "confidence_score": 0.81234,
                "model_version": "efficientnet_v2_s",
                "flagged": True,
            },
            frame_path="saved_frames/consultation-51_frame-000001_123.jpg",
            frame_number=1,
        )

        self.assertEqual(payload["consultation_id"], 51)
        self.assertEqual(payload["result"], "fake")
        self.assertEqual(payload["confidence_score"], 0.8123)
        self.assertEqual(payload["frame_path"], "saved_frames/consultation-51_frame-000001_123.jpg")
        self.assertEqual(payload["frame_number"], 1)
        self.assertEqual(payload["model_version"], "efficientnet_v2_s")
        self.assertEqual(payload["flagged"], True)
        self.assertIn("scanned_at", payload)


if __name__ == "__main__":
    unittest.main()
