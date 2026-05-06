import asyncio
import logging
import os
import unittest
from unittest.mock import AsyncMock, Mock, patch

from tests._agent_import_stubs import install_agent_dependency_stubs

install_agent_dependency_stubs()

import agent
from agent import (
    ActiveConsultationRoom,
    PolledRoomWorkerState,
    build_face_match_payload,
    build_deepfake_overlay_lines,
    build_detection_data_channel_payload,
    build_pipeline_status_payload,
    build_saved_frame_filename,
    classify_camera_guidance,
    build_scan_result_payload,
    compute_retry_delay_seconds,
    describe_active_consultation_rooms,
    dedupe_active_consultation_rooms,
    determine_deepfake_result,
    determine_deepfake_result_with_threshold,
    extract_active_consultation_rooms,
    parse_active_consultation_room,
    parse_consultation_id_from_room_name,
    poll_active_consultation_rooms,
    read_positive_int_env,
    resolve_livekit_api_url,
    resolve_log_level,
    reconcile_polled_room_workers,
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

    def test_describe_active_consultation_rooms_uses_stable_summary(self) -> None:
        room = ActiveConsultationRoom(
            consultation_id=42,
            room_name="consultation-42-abcdef",
            room_sid="RM_123",
            ws_url="wss://example.livekit.cloud",
            pipeline_token="token-123",
        )

        self.assertEqual(
            describe_active_consultation_rooms([room]),
            "consultation-42-abcdef#42",
        )
        self.assertEqual(describe_active_consultation_rooms([]), "none")

    def test_parse_consultation_id_from_room_name(self) -> None:
        self.assertEqual(parse_consultation_id_from_room_name("consultation-42-abcdef"), 42)
        self.assertIsNone(parse_consultation_id_from_room_name("not-a-consultation"))

    def test_resolve_log_level_accepts_names_and_numbers(self) -> None:
        self.assertEqual(resolve_log_level("warning"), logging.WARNING)
        self.assertEqual(resolve_log_level("15"), 15)
        self.assertEqual(resolve_log_level("not-a-level", default=logging.ERROR), logging.ERROR)

    def test_resolve_livekit_api_url_from_websocket_url(self) -> None:
        with patch.object(agent, "LIVEKIT_API_URL", ""), patch.object(agent, "LIVEKIT_URL", ""):
            self.assertEqual(
                resolve_livekit_api_url("wss://example.livekit.cloud"),
                "https://example.livekit.cloud",
            )
            self.assertEqual(
                resolve_livekit_api_url("ws://localhost:7880"),
                "http://localhost:7880",
            )

    def test_resolve_livekit_api_url_prefers_livekit_url_env(self) -> None:
        with patch.object(agent, "LIVEKIT_API_URL", ""), patch.object(agent, "LIVEKIT_URL", "wss://env.livekit.cloud"):
            self.assertEqual(
                resolve_livekit_api_url("wss://wrong.example.test"),
                "https://env.livekit.cloud",
            )

    def test_resolve_livekit_api_url_prefers_explicit_api_url_env(self) -> None:
        with patch.object(agent, "LIVEKIT_API_URL", "https://api.livekit.test"), patch.object(agent, "LIVEKIT_URL", "wss://env.livekit.cloud"):
            self.assertEqual(
                resolve_livekit_api_url("wss://wrong.example.test"),
                "https://api.livekit.test",
            )

    def test_issue_livekit_room_list_token_requires_credentials(self) -> None:
        with patch.object(agent, "LIVEKIT_API_KEY", ""), patch.object(agent, "LIVEKIT_API_SECRET", ""):
            with self.assertRaises(RuntimeError):
                agent.issue_livekit_room_list_token(now_timestamp=1000)

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

    def test_build_face_match_payload_marks_explicit_match(self) -> None:
        payload = build_face_match_payload(
            consultation_id=51,
            microcheck_id=7,
            user_id=103,
            verified_role="patient",
            recognition_result={
                "reference_loaded": True,
                "matched": True,
                "best_similarity": 0.81234,
            },
        )

        self.assertEqual(payload["consultation_id"], 51)
        self.assertEqual(payload["microcheck_id"], 7)
        self.assertEqual(payload["user_id"], 103)
        self.assertEqual(payload["verified_role"], "patient")
        self.assertEqual(payload["matched"], True)
        self.assertEqual(payload["face_match_score"], 0.8123)
        self.assertEqual(payload["flagged"], False)

    def test_build_face_match_payload_marks_explicit_failure(self) -> None:
        payload = build_face_match_payload(
            consultation_id=19,
            microcheck_id=23,
            user_id=88,
            verified_role="doctor",
            recognition_result={
                "reference_loaded": True,
                "matched": False,
                "best_similarity": 0.21999,
            },
        )

        self.assertEqual(payload["matched"], False)
        self.assertEqual(payload["face_match_score"], 0.22)
        self.assertEqual(payload["flagged"], True)

    def test_build_face_match_payload_uses_non_failure_fallback_for_non_explicit_result(self) -> None:
        payload = build_face_match_payload(
            consultation_id=19,
            microcheck_id=23,
            user_id=88,
            verified_role="doctor",
            recognition_result={
                "reference_loaded": False,
                "matched": None,
                "best_similarity": None,
            },
        )

        self.assertEqual(payload["matched"], False)
        self.assertEqual(payload["face_match_score"], 0.0)
        self.assertEqual(payload["flagged"], False)

    def test_build_pipeline_status_payload_includes_runtime_fields(self) -> None:
        payload = build_pipeline_status_payload(
            consultation_id=9,
            room_name="consultation-9-runtime",
            status="running",
            guidance={"low_light": False, "too_far": True},
            last_scan_at="2026-05-06T01:02:03+00:00",
        )

        self.assertEqual(payload["status"], "running")
        self.assertEqual(payload["consultation_id"], 9)
        self.assertEqual(payload["room_name"], "consultation-9-runtime")
        self.assertIn("heartbeat_at", payload)
        self.assertEqual(payload["guidance"]["too_far"], True)
        self.assertEqual(payload["last_scan_at"], "2026-05-06T01:02:03+00:00")

    def test_build_detection_data_channel_payload_maps_running_state(self) -> None:
        payload = build_detection_data_channel_payload(
            {
                "status": "running",
                "guidance": {
                    "low_light": True,
                    "too_far": False,
                    "face_area_ratio": 0.1,
                    "brightness": 0.12,
                    "participant_identity": "user-3",
                    "role": "patient",
                },
            }
        )

        self.assertEqual(payload["state"], "running")
        self.assertEqual(payload["guidance"]["low_light"], True)
        self.assertNotIn("participant_identity", payload["guidance"])
        self.assertNotIn("role", payload["guidance"])
        self.assertNotIn("last_error", payload)
        self.assertEqual(payload["last_heartbeat_age_seconds"], 0)

    def test_classify_camera_guidance_flags_low_light(self) -> None:
        frame = agent.np.zeros((80, 80, 3), dtype=agent.np.uint8)
        guidance = classify_camera_guidance(frame, [[10, 10, 60, 60, 0.99]], "user-3", "patient")

        self.assertEqual(guidance["low_light"], True)
        self.assertEqual(guidance["too_far"], False)
        self.assertEqual(guidance["role"], "patient")

    def test_classify_camera_guidance_flags_too_far_face(self) -> None:
        frame = agent.np.full((100, 100, 3), 240, dtype=agent.np.uint8)
        guidance = classify_camera_guidance(frame, [[10, 10, 20, 20, 0.99]], "user-4", "doctor")

        self.assertEqual(guidance["low_light"], False)
        self.assertEqual(guidance["too_far"], True)
        self.assertLess(guidance["face_area_ratio"], agent.CAMERA_MIN_FACE_AREA_RATIO)

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

    def test_parse_active_consultation_room_accepts_valid_payload(self) -> None:
        parsed = parse_active_consultation_room(
            {
                "consultation_id": "12",
                "room_name": "consultation-12-abcdef",
                "room_sid": "RM_123",
                "ws_url": "wss://example.livekit.cloud",
                "pipeline_token": "token-123",
            }
        )

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.consultation_id, 12)
        self.assertEqual(parsed.room_name, "consultation-12-abcdef")
        self.assertEqual(parsed.room_sid, "RM_123")
        self.assertEqual(parsed.ws_url, "wss://example.livekit.cloud")
        self.assertEqual(parsed.pipeline_token, "token-123")

    def test_parse_active_consultation_room_rejects_missing_required_fields(self) -> None:
        parsed = parse_active_consultation_room(
            {
                "consultation_id": 12,
                "room_name": "",
                "ws_url": "wss://example.livekit.cloud",
                "pipeline_token": "token-123",
            }
        )

        self.assertIsNone(parsed)

    def test_extract_active_consultation_rooms_filters_invalid_payload_items(self) -> None:
        rooms = extract_active_consultation_rooms(
            [
                {
                    "consultation_id": 12,
                    "room_name": "consultation-12-abcdef",
                    "room_sid": "RM_123",
                    "ws_url": "wss://example.livekit.cloud",
                    "pipeline_token": "token-123",
                },
                {
                    "consultation_id": "invalid",
                    "room_name": "consultation-99-invalid",
                    "ws_url": "wss://example.livekit.cloud",
                    "pipeline_token": "token-456",
                },
            ]
        )

        self.assertEqual(len(rooms), 1)
        self.assertEqual(rooms[0].consultation_id, 12)

    def test_dedupe_active_consultation_rooms_by_room_name(self) -> None:
        room_a = ActiveConsultationRoom(
            consultation_id=12,
            room_name="consultation-12-abcdef",
            room_sid="RM_123",
            ws_url="wss://example.livekit.cloud",
            pipeline_token="token-123",
        )
        room_b = ActiveConsultationRoom(
            consultation_id=99,
            room_name="consultation-12-abcdef",
            room_sid="RM_999",
            ws_url="wss://example.livekit.cloud",
            pipeline_token="token-999",
        )

        deduped_rooms = dedupe_active_consultation_rooms([room_a, room_b])

        self.assertEqual(len(deduped_rooms), 1)
        self.assertEqual(deduped_rooms[0].consultation_id, 12)

    def test_reconcile_polled_room_workers_starts_missing_worker(self) -> None:
        discovered_room = ActiveConsultationRoom(
            consultation_id=12,
            room_name="consultation-12-abcdef",
            room_sid="RM_123",
            ws_url="wss://example.livekit.cloud",
            pipeline_token="token-123",
        )

        rooms_to_start, rooms_to_stop = reconcile_polled_room_workers(
            discovered_rooms=[discovered_room],
            worker_states={},
            now_monotonic=10.0,
            stale_room_seconds=15.0,
        )

        self.assertEqual([room.room_name for room in rooms_to_start], ["consultation-12-abcdef"])
        self.assertEqual(rooms_to_stop, [])

    def test_reconcile_polled_room_workers_updates_existing_worker(self) -> None:
        running_task = Mock()
        running_task.done.return_value = False

        existing_room = ActiveConsultationRoom(
            consultation_id=12,
            room_name="consultation-12-abcdef",
            room_sid="RM_123",
            ws_url="wss://old.example.livekit.cloud",
            pipeline_token="old-token",
        )
        updated_room = ActiveConsultationRoom(
            consultation_id=12,
            room_name="consultation-12-abcdef",
            room_sid="RM_123",
            ws_url="wss://new.example.livekit.cloud",
            pipeline_token="new-token",
        )

        worker_states = {
            "consultation-12-abcdef": PolledRoomWorkerState(
                room=existing_room,
                task=running_task,
                last_seen_at_monotonic=1.0,
            )
        }

        rooms_to_start, rooms_to_stop = reconcile_polled_room_workers(
            discovered_rooms=[updated_room],
            worker_states=worker_states,
            now_monotonic=20.0,
            stale_room_seconds=15.0,
        )

        self.assertEqual(rooms_to_start, [])
        self.assertEqual(rooms_to_stop, [])
        self.assertEqual(worker_states["consultation-12-abcdef"].last_seen_at_monotonic, 20.0)
        self.assertEqual(
            worker_states["consultation-12-abcdef"].room.ws_url,
            "wss://new.example.livekit.cloud",
        )

    def test_reconcile_polled_room_workers_stops_stale_workers(self) -> None:
        running_task = Mock()
        running_task.done.return_value = False

        stale_room = ActiveConsultationRoom(
            consultation_id=12,
            room_name="consultation-12-abcdef",
            room_sid="RM_123",
            ws_url="wss://example.livekit.cloud",
            pipeline_token="token-123",
        )
        worker_states = {
            "consultation-12-abcdef": PolledRoomWorkerState(
                room=stale_room,
                task=running_task,
                last_seen_at_monotonic=2.0,
            )
        }

        rooms_to_start, rooms_to_stop = reconcile_polled_room_workers(
            discovered_rooms=[],
            worker_states=worker_states,
            now_monotonic=20.0,
            stale_room_seconds=10.0,
        )

        self.assertEqual(rooms_to_start, [])
        self.assertEqual(rooms_to_stop, ["consultation-12-abcdef"])

    def test_reconcile_polled_room_workers_keeps_recent_missing_worker(self) -> None:
        running_task = Mock()
        running_task.done.return_value = False

        stale_room = ActiveConsultationRoom(
            consultation_id=12,
            room_name="consultation-12-abcdef",
            room_sid="RM_123",
            ws_url="wss://example.livekit.cloud",
            pipeline_token="token-123",
        )
        worker_states = {
            "consultation-12-abcdef": PolledRoomWorkerState(
                room=stale_room,
                task=running_task,
                last_seen_at_monotonic=16.0,
            )
        }

        rooms_to_start, rooms_to_stop = reconcile_polled_room_workers(
            discovered_rooms=[],
            worker_states=worker_states,
            now_monotonic=20.0,
            stale_room_seconds=10.0,
        )

        self.assertEqual(rooms_to_start, [])
        self.assertEqual(rooms_to_stop, [])

    def test_reconcile_polled_room_workers_limits_new_workers(self) -> None:
        discovered_rooms = [
            ActiveConsultationRoom(
                consultation_id=index,
                room_name=f"consultation-{index}",
                room_sid=f"RM_{index}",
                ws_url="wss://example.livekit.cloud",
                pipeline_token=f"token-{index}",
            )
            for index in range(1, 5)
        ]

        rooms_to_start, rooms_to_stop = reconcile_polled_room_workers(
            discovered_rooms=discovered_rooms,
            worker_states={},
            now_monotonic=20.0,
            stale_room_seconds=10.0,
            max_active_workers=2,
        )

        self.assertEqual([room.room_name for room in rooms_to_start], ["consultation-1", "consultation-2"])
        self.assertEqual(rooms_to_stop, [])

    def test_reconcile_polled_room_workers_stops_workers_over_limit(self) -> None:
        running_task = Mock()
        running_task.done.return_value = False

        discovered_rooms = [
            ActiveConsultationRoom(
                consultation_id=index,
                room_name=f"consultation-{index}",
                room_sid=f"RM_{index}",
                ws_url="wss://example.livekit.cloud",
                pipeline_token=f"token-{index}",
            )
            for index in range(1, 4)
        ]
        worker_states = {
            room.room_name: PolledRoomWorkerState(
                room=room,
                task=running_task,
                last_seen_at_monotonic=20.0,
            )
            for room in discovered_rooms
        }

        rooms_to_start, rooms_to_stop = reconcile_polled_room_workers(
            discovered_rooms=discovered_rooms,
            worker_states=worker_states,
            now_monotonic=25.0,
            stale_room_seconds=10.0,
            max_active_workers=2,
        )

        self.assertEqual(rooms_to_start, [])
        self.assertEqual(rooms_to_stop, ["consultation-3"])

    def test_filter_rooms_with_livekit_participants_keeps_occupied_rooms(self) -> None:
        rooms = [
            ActiveConsultationRoom(
                consultation_id=1,
                room_name="consultation-1",
                room_sid="RM_1",
                ws_url="wss://example.livekit.cloud",
                pipeline_token="token-1",
            ),
            ActiveConsultationRoom(
                consultation_id=2,
                room_name="consultation-2",
                room_sid="RM_2",
                ws_url="wss://example.livekit.cloud",
                pipeline_token="token-2",
            ),
        ]

        async def run_filter() -> list[ActiveConsultationRoom]:
            with patch.object(agent, "PIPELINE_SUPERVISOR_REQUIRE_PARTICIPANTS", True):
                with patch("agent.fetch_livekit_room_participant_counts", new_callable=AsyncMock) as fetch_counts:
                    fetch_counts.return_value = {
                        "consultation-1": 1,
                        "consultation-2": 0,
                    }
                    return await agent.filter_rooms_with_livekit_participants(Mock(), rooms)

        filtered_rooms = asyncio.run(run_filter())

        self.assertEqual([room.room_name for room in filtered_rooms], ["consultation-1"])

    def test_poll_active_consultation_rooms_returns_failure_on_retryable_error(self) -> None:
        async def run_poll() -> tuple[list[ActiveConsultationRoom], bool]:
            with patch("agent.fetch_active_consultation_rooms_page", new_callable=AsyncMock) as fetch_page:
                fetch_page.return_value = ([], False, True)
                return await poll_active_consultation_rooms(Mock(), per_page=1, max_pages=2)

        rooms, succeeded = asyncio.run(run_poll())

        self.assertEqual(rooms, [])
        self.assertEqual(succeeded, False)


if __name__ == "__main__":
    unittest.main()
