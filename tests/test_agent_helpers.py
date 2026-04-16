import unittest
import os

from tests._agent_import_stubs import install_agent_dependency_stubs

install_agent_dependency_stubs()

import agent
from agent import (
    build_deepfake_overlay_lines,
    build_saved_frame_filename,
    build_scan_result_payload,
    compute_retry_delay_seconds,
    determine_deepfake_result,
    determine_deepfake_result_with_threshold,
    read_positive_int_env,
    resolve_deepfake_backend,
    resolve_fake_class_index_for_backend,
    resolve_claim_subject_for_scan,
    should_analyze_frame_timestamp,
    should_retry_http_status,
    should_report_deepfake_for_role,
)


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

    def test_determine_deepfake_result_with_threshold_respects_inconclusive_margin(self) -> None:
        result_equal, confidence_equal = determine_deepfake_result_with_threshold(
            fake_score=0.5,
            threshold=0.5,
            inconclusive_margin=0.05,
        )
        result_high, confidence_high = determine_deepfake_result_with_threshold(
            fake_score=0.56,
            threshold=0.5,
            inconclusive_margin=0.05,
        )
        result_low, confidence_low = determine_deepfake_result_with_threshold(
            fake_score=0.44,
            threshold=0.5,
            inconclusive_margin=0.05,
        )

        self.assertEqual(result_equal, "inconclusive")
        self.assertAlmostEqual(confidence_equal, 0.5, places=4)
        self.assertEqual(result_high, "fake")
        self.assertAlmostEqual(confidence_high, 0.56, places=4)
        self.assertEqual(result_low, "real")
        self.assertAlmostEqual(confidence_low, 0.56, places=4)

    def test_build_scan_result_payload_uses_expected_keys(self) -> None:
        payload = build_scan_result_payload(
            consultation_id=51,
            microcheck_id=7,
            user_id=103,
            verified_role="patient",
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
        self.assertEqual(payload["microcheck_id"], 7)
        self.assertEqual(payload["user_id"], 103)
        self.assertEqual(payload["verified_role"], "patient")
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

    def test_should_analyze_frame_timestamp_allows_first_frame(self) -> None:
        self.assertEqual(
            should_analyze_frame_timestamp(
                last_analyzed_timestamp_us=None,
                current_timestamp_us=1_000_000,
                interval_seconds=1.0,
            ),
            True,
        )

    def test_should_analyze_frame_timestamp_enforces_interval(self) -> None:
        self.assertEqual(
            should_analyze_frame_timestamp(
                last_analyzed_timestamp_us=1_000_000,
                current_timestamp_us=1_500_000,
                interval_seconds=1.0,
            ),
            False,
        )
        self.assertEqual(
            should_analyze_frame_timestamp(
                last_analyzed_timestamp_us=1_000_000,
                current_timestamp_us=2_000_000,
                interval_seconds=1.0,
            ),
            True,
        )

    def test_read_positive_int_env_returns_default_for_invalid_values(self) -> None:
        env_key = "UNIT_TEST_POSITIVE_INT"
        original_value = os.environ.get(env_key)

        try:
            os.environ[env_key] = "abc"
            self.assertEqual(read_positive_int_env(env_key, 7), 7)

            os.environ[env_key] = "0"
            self.assertEqual(read_positive_int_env(env_key, 7), 7)

            os.environ[env_key] = "-2"
            self.assertEqual(read_positive_int_env(env_key, 7), 7)

            os.environ[env_key] = "3"
            self.assertEqual(read_positive_int_env(env_key, 7), 3)
        finally:
            if original_value is None:
                os.environ.pop(env_key, None)
            else:
                os.environ[env_key] = original_value

    def test_build_deepfake_overlay_lines_includes_expected_metrics(self) -> None:
        lines = build_deepfake_overlay_lines(
            {
                "result": "fake",
                "confidence_score": 0.81234,
                "fake_score": 0.93456,
                "alternate_score": 0.06544,
                "fake_class_index": 1,
                "score_aggregation": "max",
                "faces_evaluated": 3,
            }
        )

        self.assertEqual(
            lines,
            [
                "deepfake: fake",
                "confidence: 0.8123",
                "scores f/a: 0.9346/0.0654",
                "class index: 1",
                "aggregation: max",
                "regions: 3",
            ],
        )

    def test_build_deepfake_overlay_lines_handles_empty_results(self) -> None:
        lines = build_deepfake_overlay_lines(None)

        self.assertEqual(
            lines,
            [
                "deepfake: unavailable",
                "confidence: n/a",
                "scores f/a: n/a/n/a",
                f"class index: {agent.DEEPFAKE_FAKE_CLASS_INDEX}",
                f"aggregation: {agent.DEEPFAKE_SCORE_AGGREGATION}",
                "regions: 0",
            ],
        )

    def test_resolve_deepfake_backend_prefers_supported_requested_backend(self) -> None:
        resolved_backend, used_fallback = resolve_deepfake_backend(
            requested_backend="deepfakebench_ucf",
            fallback_backend="deepfakebench_effnb4",
            strict_mode=False,
        )

        self.assertEqual(resolved_backend, "deepfakebench_ucf")
        self.assertEqual(used_fallback, False)

    def test_resolve_deepfake_backend_falls_back_when_requested_is_unsupported(self) -> None:
        resolved_backend, used_fallback = resolve_deepfake_backend(
            requested_backend="unknown-backend",
            fallback_backend="deepfakebench_effnb4",
            strict_mode=False,
        )

        self.assertEqual(resolved_backend, "deepfakebench_effnb4")
        self.assertEqual(used_fallback, True)

    def test_resolve_deepfake_backend_respects_strict_mode(self) -> None:
        resolved_backend, used_fallback = resolve_deepfake_backend(
            requested_backend="unknown-backend",
            fallback_backend="deepfakebench_effnb4",
            strict_mode=True,
        )

        self.assertEqual(resolved_backend, "unknown-backend")
        self.assertEqual(used_fallback, True)

    def test_resolve_fake_class_index_for_backend_applies_autocorrect_for_effnet(self) -> None:
        resolved_class_index = resolve_fake_class_index_for_backend(
            backend="deepfakebench_effnb4",
            default_index=0,
            backend_override=None,
            auto_correct=True,
        )

        self.assertEqual(resolved_class_index, 1)

    def test_resolve_fake_class_index_for_backend_uses_backend_override(self) -> None:
        resolved_class_index = resolve_fake_class_index_for_backend(
            backend="deepfakebench_ucf",
            default_index=1,
            backend_override=0,
            auto_correct=False,
        )

        self.assertEqual(resolved_class_index, 0)

    def test_should_retry_http_status_flags_transient_statuses(self) -> None:
        self.assertEqual(should_retry_http_status(408), True)
        self.assertEqual(should_retry_http_status(429), True)
        self.assertEqual(should_retry_http_status(500), True)
        self.assertEqual(should_retry_http_status(503), True)

    def test_should_retry_http_status_ignores_non_transient_statuses(self) -> None:
        self.assertEqual(should_retry_http_status(400), False)
        self.assertEqual(should_retry_http_status(401), False)
        self.assertEqual(should_retry_http_status(422), False)

    def test_compute_retry_delay_seconds_grows_and_caps(self) -> None:
        original_base = agent.REQUEST_RETRY_BASE_DELAY_SECONDS
        original_max = agent.REQUEST_RETRY_MAX_DELAY_SECONDS

        try:
            agent.REQUEST_RETRY_BASE_DELAY_SECONDS = 0.2
            agent.REQUEST_RETRY_MAX_DELAY_SECONDS = 0.5

            self.assertAlmostEqual(compute_retry_delay_seconds(0), 0.2, places=4)
            self.assertAlmostEqual(compute_retry_delay_seconds(1), 0.4, places=4)
            self.assertAlmostEqual(compute_retry_delay_seconds(3), 0.5, places=4)
        finally:
            agent.REQUEST_RETRY_BASE_DELAY_SECONDS = original_base
            agent.REQUEST_RETRY_MAX_DELAY_SECONDS = original_max

    def test_resolve_claim_subject_for_scan_prefers_inferred_user(self) -> None:
        gallery = type("GalleryState", (), {"patient_id": 11, "doctor_id": 22})()

        role, user_id = resolve_claim_subject_for_scan(
            inferred_role="patient",
            inferred_user_id=99,
            gallery=gallery,
        )

        self.assertEqual(role, "patient")
        self.assertEqual(user_id, 99)

    def test_resolve_claim_subject_for_scan_falls_back_to_gallery_user(self) -> None:
        gallery = type("GalleryState", (), {"patient_id": 11, "doctor_id": 22})()

        role, user_id = resolve_claim_subject_for_scan(
            inferred_role="doctor",
            inferred_user_id=None,
            gallery=gallery,
        )

        self.assertEqual(role, "doctor")
        self.assertEqual(user_id, 22)

    def test_resolve_claim_subject_for_scan_returns_none_without_role(self) -> None:
        gallery = type("GalleryState", (), {"patient_id": 11, "doctor_id": 22})()

        role, user_id = resolve_claim_subject_for_scan(
            inferred_role=None,
            inferred_user_id=11,
            gallery=gallery,
        )

        self.assertIsNone(role)
        self.assertIsNone(user_id)


if __name__ == "__main__":
    unittest.main()
