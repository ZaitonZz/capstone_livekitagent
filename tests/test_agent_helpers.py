import unittest

import agent
from agent import build_saved_frame_filename, build_scan_result_payload, determine_deepfake_result, should_report_deepfake_for_role


class AgentHelperFunctionsTest(unittest.TestCase):
    def test_build_saved_frame_filename_uses_sequential_timestamp_pattern(self) -> None:
        filename = build_saved_frame_filename(
            consultation_id=27,
            frame_number=12,
            timestamp_us=1234567890,
            track_id="TR_abcd1234",
        )

        self.assertEqual(filename, "consultation-27_track-TR-abcd1234_frame-000012_1234567890.jpg")

    def test_build_saved_frame_filename_handles_missing_consultation(self) -> None:
        filename = build_saved_frame_filename(
            consultation_id=None,
            frame_number=3,
            timestamp_us=999,
            track_id="TR:with:separators",
        )

        self.assertEqual(filename, "consultation-unknown_track-TR-with-separators_frame-000003_999.jpg")

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

    def test_should_report_deepfake_for_role_when_mode_is_patient(self) -> None:
        original = agent.DEEPFAKE_REPORTING_ROLE
        agent.DEEPFAKE_REPORTING_ROLE = "patient"
        try:
            self.assertEqual(should_report_deepfake_for_role("patient"), True)
            self.assertEqual(should_report_deepfake_for_role("doctor"), False)
            self.assertEqual(should_report_deepfake_for_role(None), False)
        finally:
            agent.DEEPFAKE_REPORTING_ROLE = original

    def test_should_report_deepfake_for_role_when_mode_is_both(self) -> None:
        original = agent.DEEPFAKE_REPORTING_ROLE
        agent.DEEPFAKE_REPORTING_ROLE = "both"
        try:
            self.assertEqual(should_report_deepfake_for_role("patient"), True)
            self.assertEqual(should_report_deepfake_for_role("doctor"), True)
            self.assertEqual(should_report_deepfake_for_role(None), True)
        finally:
            agent.DEEPFAKE_REPORTING_ROLE = original


if __name__ == "__main__":
    unittest.main()
