import asyncio
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import hmac
import json
import logging
import os
import threading
import time
from typing import Any
from urllib.parse import urljoin

# Apply torch runtime log controls before any module imports torch.
os.environ.setdefault("PYTORCH_DISABLE_NNPACK", "1")
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import aiohttp
import numpy as np
from deepfakebench_effnet import (
    DeepfakeBenchEfficientNetB4Adapter,
    DeepfakeBenchEfficientNetB4Config,
    aggregate_fake_scores,
    crop_face_regions,
)
from deepfakebench_ucf import DeepfakeBenchUCFAdapter, DeepfakeBenchUCFConfig
from dotenv import load_dotenv
from face_recognition import FaceGallery as BaseFaceGallery, cosine_similarity, normalize_embedding
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, JobProcess, WorkerOptions, cli

load_dotenv()

logger = logging.getLogger("video-pipeline-agent")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def read_positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        parsed_value = int(raw_value.strip())
    except (TypeError, ValueError):
        return default

    return parsed_value if parsed_value > 0 else default


def read_optional_int_env(name: str) -> int | None:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return None

    try:
        return int(raw_value.strip())
    except (TypeError, ValueError):
        return None


def read_optional_float_env(name: str) -> float | None:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return None

    try:
        return float(raw_value.strip())
    except (TypeError, ValueError):
        return None


def read_bool_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    return default


LARAVEL_BASE_URL = os.getenv("LARAVEL_BASE_URL", "http://localhost:8000").rstrip("/")
LARAVEL_ENDPOINT = os.getenv("LARAVEL_ENDPOINT", f"{LARAVEL_BASE_URL}/api/frame-results")
PIPELINE_INTERNAL_BASE_URL = os.getenv(
    "PIPELINE_INTERNAL_BASE_URL",
    f"{LARAVEL_BASE_URL}/internal/pipeline",
).rstrip("/")
PIPELINE_SHARED_SECRET = os.getenv("PIPELINE_SHARED_SECRET", "")
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.4"))
FRAME_ANALYSIS_INTERVAL_SECONDS = float(os.getenv("FRAME_ANALYSIS_INTERVAL_SECONDS", "1.0"))
VIDEO_STREAM_CAPACITY = read_positive_int_env("VIDEO_STREAM_CAPACITY", 1)
FRAME_ANALYSIS_MAX_WORKERS = read_positive_int_env("FRAME_ANALYSIS_MAX_WORKERS", 1)
FACE_MATCH_STREAK_TARGET = int(os.getenv("FACE_MATCH_STREAK_TARGET", "3"))
SAVED_FRAMES_DIR = os.getenv("SAVED_FRAMES_DIR", "saved_frames")
REPORT_MISSING_REFERENCE_AS_FLAG = read_bool_env("REPORT_MISSING_REFERENCE_AS_FLAG", True)
VERIFICATION_TARGET = os.getenv("VERIFICATION_TARGET", "both").strip().lower()
PARTICIPANT_AWARE_VERIFICATION = read_bool_env("PARTICIPANT_AWARE_VERIFICATION", True)
DEEPFAKE_MODEL_BACKEND = os.getenv("DEEPFAKE_MODEL_BACKEND", "deepfakebench_effnb4").strip().lower()
DEEPFAKE_FALLBACK_BACKEND = os.getenv("DEEPFAKE_FALLBACK_BACKEND", "deepfakebench_effnb4").strip().lower()
DEEPFAKE_BACKEND_STRICT = read_bool_env("DEEPFAKE_BACKEND_STRICT", False)
DEEPFAKE_AUTO_CORRECT_FAKE_CLASS_INDEX = read_bool_env("DEEPFAKE_AUTO_CORRECT_FAKE_CLASS_INDEX", False)
DEEPFAKE_WEIGHTS_DIR = os.getenv("DEEPFAKE_WEIGHTS_DIR", os.path.join(BASE_DIR, "models", "deepfakebench"))
DEEPFAKE_MODEL_PATH = os.getenv("DEEPFAKE_MODEL_PATH", os.path.join(DEEPFAKE_WEIGHTS_DIR, "effnb4_best.pth"))
DEEPFAKE_BACKBONE_PATH = os.getenv(
    "DEEPFAKE_BACKBONE_PATH",
    os.path.join(DEEPFAKE_WEIGHTS_DIR, "efficientnet-b4-6ed6700e.pth"),
)
DEEPFAKE_MODEL_URL = os.getenv(
    "DEEPFAKE_MODEL_URL",
    "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/effnb4_best.pth",
)
DEEPFAKE_BACKBONE_URL = os.getenv(
    "DEEPFAKE_BACKBONE_URL",
    "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth",
)
DEEPFAKE_AUTO_DOWNLOAD = read_bool_env("DEEPFAKE_AUTO_DOWNLOAD", True)
DEEPFAKE_PREFER_CPU = read_bool_env("DEEPFAKE_PREFER_CPU", True)
DEEPFAKE_MODEL_VERSION = os.getenv("DEEPFAKE_MODEL_VERSION", "deepfakebench_effnb4")
DEEPFAKE_INPUT_SIZE = int(os.getenv("DEEPFAKE_INPUT_SIZE", "256"))
DEEPFAKE_FAKE_THRESHOLD = float(os.getenv("DEEPFAKE_FAKE_THRESHOLD", "0.5"))
DEEPFAKE_INCONCLUSIVE_MARGIN = float(os.getenv("DEEPFAKE_INCONCLUSIVE_MARGIN", "0.05"))
# DeepfakeBench train_config label_dict maps fake->1 and real->0.
DEEPFAKE_FAKE_CLASS_INDEX = int(os.getenv("DEEPFAKE_FAKE_CLASS_INDEX", "1"))
DEEPFAKE_USE_FACE_CROPS = read_bool_env("DEEPFAKE_USE_FACE_CROPS", True)
DEEPFAKE_FULL_FRAME_FALLBACK = read_bool_env("DEEPFAKE_FULL_FRAME_FALLBACK", False)
DEEPFAKE_FACE_MARGIN_RATIO = float(os.getenv("DEEPFAKE_FACE_MARGIN_RATIO", "0.25"))
DEEPFAKE_MIN_FACE_SIZE = int(os.getenv("DEEPFAKE_MIN_FACE_SIZE", "48"))
DEEPFAKE_SCORE_AGGREGATION = os.getenv("DEEPFAKE_SCORE_AGGREGATION", "max").strip().lower()
DEEPFAKE_REPORTING_ROLE = os.getenv("DEEPFAKE_REPORTING_ROLE", "both").strip().lower()

DEEPFAKE_UCF_MODEL_PATH = os.getenv("DEEPFAKE_UCF_MODEL_PATH", os.path.join(DEEPFAKE_WEIGHTS_DIR, "ucf_best.pth"))
DEEPFAKE_UCF_MODEL_URL = os.getenv("DEEPFAKE_UCF_MODEL_URL", "")
DEEPFAKE_UCF_BACKBONE_NAME = os.getenv("DEEPFAKE_UCF_BACKBONE_NAME", "xception").strip().lower()
DEEPFAKE_UCF_BACKBONE_PATH = os.getenv("DEEPFAKE_UCF_BACKBONE_PATH", DEEPFAKE_BACKBONE_PATH)
DEEPFAKE_UCF_BACKBONE_URL = os.getenv("DEEPFAKE_UCF_BACKBONE_URL", DEEPFAKE_BACKBONE_URL)
DEEPFAKE_UCF_MODEL_VERSION = os.getenv("DEEPFAKE_UCF_MODEL_VERSION", "deepfakebench_ucf")
DEEPFAKE_UCF_INPUT_SIZE = int(os.getenv("DEEPFAKE_UCF_INPUT_SIZE", str(DEEPFAKE_INPUT_SIZE)))
DEEPFAKE_UCF_ENCODER_FEAT_DIM = int(os.getenv("DEEPFAKE_UCF_ENCODER_FEAT_DIM", "512"))
DEEPFAKE_UCF_FAKE_CLASS_INDEX = read_optional_int_env("DEEPFAKE_UCF_FAKE_CLASS_INDEX")
DEEPFAKE_UCF_FAKE_THRESHOLD = read_optional_float_env("DEEPFAKE_UCF_FAKE_THRESHOLD")

if DEEPFAKE_UCF_BACKBONE_NAME in {"effnb4", "efficientnet", "efficientnet_b4", "efficientnet-b4"}:
    DEEPFAKE_UCF_BACKBONE_NAME = "efficientnet-b4"
else:
    DEEPFAKE_UCF_BACKBONE_NAME = "xception"

SUPPORTED_DEEPFAKE_BACKENDS = {"deepfakebench_effnb4", "deepfakebench_ucf"}


def build_pipeline_signature_headers(body: str = "") -> dict[str, str]:
    digest = hmac.new(
        PIPELINE_SHARED_SECRET.encode("utf-8"),
        body.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return {
        "Content-Type": "application/json",
        "X-Pipeline-Signature": f"sha256={digest}",
    }


def build_internal_url(path: str) -> str:
    return f"{PIPELINE_INTERNAL_BASE_URL}/{path.lstrip('/')}"


def resolve_asset_url(url: str) -> str:
    if url.startswith("http://") or url.startswith("https://"):
        return url

    return urljoin(f"{LARAVEL_BASE_URL}/", url.lstrip("/"))


def resolve_deepfake_backend(requested_backend: str, fallback_backend: str, strict_mode: bool) -> tuple[str, bool]:
    normalized_requested = requested_backend.strip().lower()
    normalized_fallback = fallback_backend.strip().lower()

    if normalized_requested in SUPPORTED_DEEPFAKE_BACKENDS:
        return normalized_requested, False

    if strict_mode:
        return normalized_requested, True

    if normalized_fallback in SUPPORTED_DEEPFAKE_BACKENDS:
        return normalized_fallback, True

    return "deepfakebench_effnb4", True


def resolve_fake_class_index_for_backend(
    backend: str,
    default_index: int,
    backend_override: int | None,
    auto_correct: bool,
) -> int:
    if backend_override is not None:
        return backend_override

    if backend == "deepfakebench_effnb4" and auto_correct and default_index != 1:
        return 1

    return default_index


def determine_deepfake_result_with_threshold(
    fake_score: float,
    threshold: float,
    inconclusive_margin: float,
) -> tuple[str, float]:
    bounded_score = max(0.0, min(1.0, float(fake_score)))
    if abs(bounded_score - threshold) <= inconclusive_margin:
        return "inconclusive", max(bounded_score, 1.0 - bounded_score)

    if bounded_score >= threshold:
        return "fake", bounded_score

    return "real", 1.0 - bounded_score


def determine_deepfake_result(fake_score: float) -> tuple[str, float]:
    return determine_deepfake_result_with_threshold(
        fake_score=fake_score,
        threshold=DEEPFAKE_FAKE_THRESHOLD,
        inconclusive_margin=DEEPFAKE_INCONCLUSIVE_MARGIN,
    )


def normalize_track_id(track_id: str) -> str:
    safe = "".join(character if character.isalnum() else "-" for character in track_id.strip())
    return safe.strip("-") or "unknown-track"


def should_report_deepfake_for_role(inferred_role: str | None) -> bool:
    if DEEPFAKE_REPORTING_ROLE == "both":
        return True

    if DEEPFAKE_REPORTING_ROLE in {"patient", "doctor"}:
        return inferred_role == DEEPFAKE_REPORTING_ROLE

    return True


def should_analyze_frame_timestamp(
    last_analyzed_timestamp_us: int | None,
    current_timestamp_us: int,
    interval_seconds: float,
) -> bool:
    if last_analyzed_timestamp_us is None:
        return True

    interval_us = max(int(interval_seconds * 1_000_000), 0)
    return (current_timestamp_us - last_analyzed_timestamp_us) >= interval_us


def build_saved_frame_filename(consultation_id: int | None, frame_number: int, timestamp_us: int, track_id: str) -> str:
    consultation_fragment = str(consultation_id) if consultation_id is not None else "unknown"
    track_fragment = normalize_track_id(track_id)
    return f"consultation-{consultation_fragment}_track-{track_fragment}_frame-{frame_number:06d}_{timestamp_us}.jpg"


def build_scan_result_payload(
    consultation_id: int,
    microcheck_id: int,
    user_id: int,
    verified_role: str,
    deepfake_result: dict[str, Any],
    frame_path: str,
    frame_number: int,
) -> dict[str, Any]:
    result = str(deepfake_result.get("result", "inconclusive"))
    confidence_score = round(float(deepfake_result.get("confidence_score", 0.0)), 4)
    flagged = bool(deepfake_result.get("flagged", result == "fake"))

    return {
        "consultation_id": consultation_id,
        "microcheck_id": microcheck_id,
        "user_id": user_id,
        "verified_role": verified_role,
        "result": result,
        "confidence_score": confidence_score,
        "frame_path": frame_path,
        "frame_number": frame_number,
        "model_version": str(deepfake_result.get("model_version", DEEPFAKE_MODEL_VERSION)),
        "flagged": flagged,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
    }


def format_overlay_metric(value: Any, precision: int = 4) -> str:
    if value is None:
        return "n/a"

    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def build_deepfake_overlay_lines(deepfake_result: dict[str, Any] | None) -> list[str]:
    if not deepfake_result:
        return [
            "deepfake: unavailable",
            "confidence: n/a",
            "scores f/a: n/a/n/a",
            f"class index: {DEEPFAKE_FAKE_CLASS_INDEX}",
            f"aggregation: {DEEPFAKE_SCORE_AGGREGATION}",
            "regions: 0",
        ]

    class_index_value = deepfake_result.get("fake_class_index", DEEPFAKE_FAKE_CLASS_INDEX)
    aggregation_mode = deepfake_result.get("score_aggregation", DEEPFAKE_SCORE_AGGREGATION)

    try:
        regions_evaluated = int(deepfake_result.get("faces_evaluated", 0))
    except (TypeError, ValueError):
        regions_evaluated = 0

    return [
        f"deepfake: {deepfake_result.get('result', 'inconclusive')}",
        f"confidence: {format_overlay_metric(deepfake_result.get('confidence_score'))}",
        (
            "scores f/a: "
            f"{format_overlay_metric(deepfake_result.get('fake_score'))}/"
            f"{format_overlay_metric(deepfake_result.get('alternate_score'))}"
        ),
        f"class index: {class_index_value}",
        f"aggregation: {aggregation_mode}",
        f"regions: {regions_evaluated}",
    ]


def draw_text_overlay(frame: np.ndarray, lines: list[str]) -> None:
    import cv2

    if not lines:
        return

    start_x = 12
    start_y = 26
    line_height = 24
    text_thickness = 2
    text_color = (255, 255, 255)
    background_color = (16, 16, 16)

    for index, line in enumerate(lines):
        text_y = start_y + (index * line_height)
        (text_width, text_height), baseline = cv2.getTextSize(
            line,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_thickness,
        )
        top_left = (start_x - 6, max(text_y - text_height - 6, 0))
        bottom_right = (start_x + text_width + 6, min(text_y + baseline + 4, frame.shape[0] - 1))
        cv2.rectangle(frame, top_left, bottom_right, background_color, -1)
        cv2.putText(
            frame,
            line,
            (start_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            text_thickness,
            cv2.LINE_AA,
        )


class FaceGallery(BaseFaceGallery):
    def __init__(
        self,
        threshold: float = FACE_MATCH_THRESHOLD,
        confirmation_streak: int = FACE_MATCH_STREAK_TARGET,
    ):
        super().__init__(threshold=threshold, confirmation_streak=confirmation_streak)
        self.consultation_id: int | None = None

        # Patient attributes
        self.patient_id: int | None = None
        self.patient_name: str | None = None
        self.patient_photo_id: int | None = None
        self.patient_reference_embedding: np.ndarray | None = None
        self._patient_missing_reference_reported = False

        # Doctor attributes
        self.doctor_id: int | None = None
        self.doctor_name: str | None = None
        self.doctor_photo_id: int | None = None
        self.doctor_reference_embedding: np.ndarray | None = None
        self._doctor_missing_reference_reported = False

    async def _report_missing_patient_reference(self, session: aiohttp.ClientSession) -> None:
        if not REPORT_MISSING_REFERENCE_AS_FLAG:
            return

        if self._patient_missing_reference_reported:
            return

        if self.consultation_id is None:
            return

        if self.patient_id is None:
            return

        payload = {
            "consultation_id": self.consultation_id,
            "user_id": self.patient_id,
            "verified_role": "patient",
            "matched": False,
            "face_match_score": 0.0,
            "flagged": True,
        }

        stored = await post_internal_json(
            session,
            build_internal_url("face-match-results"),
            payload,
        )

        if stored:
            self._patient_missing_reference_reported = True
            logger.warning(
                "Reported missing patient reference for consultation %s",
                self.consultation_id,
            )

    async def _report_missing_doctor_reference(self, session: aiohttp.ClientSession) -> None:
        if not REPORT_MISSING_REFERENCE_AS_FLAG:
            return

        if self._doctor_missing_reference_reported:
            return

        if self.consultation_id is None:
            return

        if self.doctor_id is None:
            return

        payload = {
            "consultation_id": self.consultation_id,
            "user_id": self.doctor_id,
            "verified_role": "doctor",
            "matched": False,
            "face_match_score": 0.0,
            "flagged": True,
        }

        stored = await post_internal_json(
            session,
            build_internal_url("face-match-results"),
            payload,
        )

        if stored:
            self._doctor_missing_reference_reported = True
            logger.warning(
                "Reported missing doctor reference for consultation %s",
                self.consultation_id,
            )

    async def _load_patient_reference(
        self,
        room_name: str,
        session: aiohttp.ClientSession,
        pipeline_manager: "PipelineManager",
    ) -> bool:
        """Load patient face reference embedding from Laravel endpoint."""
        try:
            async with session.get(
                build_internal_url(f"consultation/{room_name}/patient-face"),
                headers=build_pipeline_signature_headers(""),
            ) as response:
                if response.status == 404:
                    logger.warning("No patient face data found for room %s", room_name)
                    return False

                if response.status != 200:
                    logger.warning(
                        "Failed to load patient face data for room %s: HTTP %s",
                        room_name,
                        response.status,
                    )
                    return False

                payload = await response.json()
        except Exception as error:
            logger.error("Failed to fetch patient face data for room %s: %s", room_name, error)
            return False

        self.patient_id = payload.get("patient_id") or payload.get("subject_user_id")
        self.patient_name = payload.get("patient_name") or payload.get("subject_name")
        self.patient_photo_id = payload.get("photo_id")
        self.patient_reference_embedding = normalize_embedding(payload.get("face_embedding"))

        logger.info(
            "Patient-face payload: patient_id=%s patient_name=%s photo_id=%s has_embedding=%s used_fallback_photo=%s",
            self.patient_id,
            self.patient_name,
            self.patient_photo_id,
            self.patient_reference_embedding is not None,
            payload.get("used_fallback_photo"),
        )

        if self.patient_reference_embedding is not None:
            logger.info("Loaded stored ArcFace embedding for patient %s", self.patient_id)
            return True

        photo_path = payload.get("photo_path")
        if photo_path is None or self.patient_photo_id is None:
            logger.warning(
                "Patient %s has no usable photo from Laravel (primary/fallback) for ArcFace enrollment",
                self.patient_id,
            )
            await self._report_missing_patient_reference(session)
            return False

        photo_url = f"{LARAVEL_BASE_URL}{photo_path}"
        computed_embedding = await pipeline_manager.compute_reference_embedding_from_url(session, photo_url)
        if computed_embedding is None:
            logger.warning("Unable to compute ArcFace embedding for patient photo %s", self.patient_photo_id)
            await self._report_missing_patient_reference(session)
            return False

        stored = await post_internal_json(
            session,
            build_internal_url(f"face-embeddings/{self.patient_photo_id}"),
            {"embedding": computed_embedding.astype(np.float32).tolist()},
        )
        if not stored:
            await self._report_missing_patient_reference(session)
            return False

        self.patient_reference_embedding = normalize_embedding(computed_embedding)
        logger.info("Computed and stored ArcFace embedding for patient photo %s", self.patient_photo_id)
        return self.patient_reference_embedding is not None

    async def _load_doctor_reference(
        self,
        room_name: str,
        session: aiohttp.ClientSession,
        pipeline_manager: "PipelineManager",
    ) -> bool:
        """Load doctor face reference embedding from Laravel endpoint."""
        try:
            async with session.get(
                build_internal_url(f"consultation/{room_name}/patient-face?role=doctor"),
                headers=build_pipeline_signature_headers(""),
            ) as response:
                if response.status == 404:
                    logger.info("No doctor face data found for room %s", room_name)
                    return False

                if response.status != 200:
                    logger.warning(
                        "Failed to load doctor face data for room %s: HTTP %s",
                        room_name,
                        response.status,
                    )
                    return False

                payload = await response.json()
        except Exception as error:
            logger.error("Failed to fetch doctor face data for room %s: %s", room_name, error)
            return False

        self.doctor_id = payload.get("doctor_id") or payload.get("subject_user_id")
        self.doctor_name = payload.get("doctor_name") or payload.get("subject_name")
        self.doctor_photo_id = payload.get("photo_id")
        self.doctor_reference_embedding = normalize_embedding(payload.get("face_embedding"))

        logger.info(
            "Doctor-face payload: doctor_id=%s doctor_name=%s photo_id=%s has_embedding=%s used_fallback_photo=%s",
            self.doctor_id,
            self.doctor_name,
            self.doctor_photo_id,
            self.doctor_reference_embedding is not None,
            payload.get("used_fallback_photo"),
        )

        if self.doctor_reference_embedding is not None:
            logger.info("Loaded stored ArcFace embedding for doctor %s", self.doctor_id)
            return True

        photo_path = payload.get("photo_path")
        if photo_path is None or self.doctor_photo_id is None:
            logger.warning(
                "Doctor %s has no usable photo from Laravel (primary/fallback) for ArcFace enrollment",
                self.doctor_id,
            )
            await self._report_missing_doctor_reference(session)
            return False

        photo_url = f"{LARAVEL_BASE_URL}{photo_path}"
        computed_embedding = await pipeline_manager.compute_reference_embedding_from_url(session, photo_url)
        if computed_embedding is None:
            logger.warning("Unable to compute ArcFace embedding for doctor photo %s", self.doctor_photo_id)
            await self._report_missing_doctor_reference(session)
            return False

        stored = await post_internal_json(
            session,
            build_internal_url(f"face-embeddings/doctor/{self.doctor_photo_id}"),
            {"embedding": computed_embedding.astype(np.float32).tolist()},
        )
        if not stored:
            await self._report_missing_doctor_reference(session)
            return False

        self.doctor_reference_embedding = normalize_embedding(computed_embedding)
        logger.info("Computed and stored ArcFace embedding for doctor photo %s", self.doctor_photo_id)
        return self.doctor_reference_embedding is not None

    async def load_for_room(
        self,
        room_name: str,
        session: aiohttp.ClientSession,
        pipeline_manager: "PipelineManager",
    ) -> bool:
        """Load both patient and doctor face references for the consultation room."""
        self.consultation_id = None

        try:
            # First, get the consultation ID from patient endpoint (required)
            async with session.get(
                build_internal_url(f"consultation/{room_name}/patient-face"),
                headers=build_pipeline_signature_headers(""),
            ) as response:
                if response.status == 200:
                    payload = await response.json()
                    self.consultation_id = payload.get("consultation_id")
                elif response.status != 404:
                    logger.warning("Failed to get consultation ID: HTTP %s", response.status)
        except Exception as error:
            logger.error("Failed to fetch consultation ID: %s", error)

        target = VERIFICATION_TARGET if VERIFICATION_TARGET in {"patient", "doctor", "both"} else "both"
        logger.info("Face verification target mode: %s", target)

        patient_loaded = False
        doctor_loaded = False

        if target in {"patient", "both"}:
            patient_loaded = await self._load_patient_reference(room_name, session, pipeline_manager)

        if target in {"doctor", "both"}:
            doctor_loaded = await self._load_doctor_reference(room_name, session, pipeline_manager)

        return patient_loaded or doctor_loaded

    def resolve_track_subject(self, participant: rtc.RemoteParticipant | None) -> tuple[str | None, int | None]:
        if participant is None:
            return None, None

        metadata_raw = getattr(participant, "metadata", None)
        role_from_metadata: str | None = None
        if isinstance(metadata_raw, str) and metadata_raw.strip() != "":
            try:
                metadata = json.loads(metadata_raw)
                metadata_role_candidate = str(metadata.get("role", "")).strip().lower()
                if metadata_role_candidate in {"doctor", "patient"}:
                    role_from_metadata = metadata_role_candidate
            except json.JSONDecodeError:
                logger.debug("Unable to decode participant metadata JSON for participant %s", getattr(participant, "identity", "unknown"))

        identity = str(getattr(participant, "identity", "")).strip().lower()
        user_id: int | None = None

        if identity.startswith("user-"):
            user_id_str = identity.removeprefix("user-")
            if user_id_str.isdigit():
                user_id = int(user_id_str)
        elif identity.isdigit():
            user_id = int(identity)

        if role_from_metadata in {"patient", "doctor"}:
            if user_id is None:
                user_id = self.patient_id if role_from_metadata == "patient" else self.doctor_id

            return role_from_metadata, user_id

        if user_id is not None:
            if self.patient_id is not None and user_id == self.patient_id:
                return "patient", user_id
            if self.doctor_id is not None and user_id == self.doctor_id:
                return "doctor", user_id

        return None, user_id


class PipelineManager:
    def __init__(self):
        import torch
        from insightface.app import FaceAnalysis

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using compute device: %s", self.device)

        logger.info("Loading InsightFace buffalo_l with ArcFace recognition...")
        self.face_app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
        ctx_id = 0 if torch.cuda.is_available() else -1
        self.face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        logger.info("InsightFace loaded successfully.")

        deepfake_device = torch.device("cpu")
        if not DEEPFAKE_PREFER_CPU and torch.cuda.is_available():
            deepfake_device = torch.device("cuda")

        requested_backend = DEEPFAKE_MODEL_BACKEND
        resolved_backend, used_fallback_backend = resolve_deepfake_backend(
            requested_backend=DEEPFAKE_MODEL_BACKEND,
            fallback_backend=DEEPFAKE_FALLBACK_BACKEND,
            strict_mode=DEEPFAKE_BACKEND_STRICT,
        )

        if used_fallback_backend and resolved_backend != requested_backend:
            logger.warning(
                "Deepfake backend '%s' is not supported. Falling back to '%s'.",
                requested_backend,
                resolved_backend,
            )

        (
            self.deepfake_adapter,
            self.deepfake_backend,
            self.deepfake_detector_name,
            self.deepfake_model_version,
            self.deepfake_fake_threshold,
            self.deepfake_fake_class_index,
        ) = self._build_deepfake_adapter_for_backend(resolved_backend, deepfake_device)

        if (
            not self.deepfake_adapter.is_loaded
            and not DEEPFAKE_BACKEND_STRICT
            and self.deepfake_backend != "deepfakebench_effnb4"
        ):
            logger.warning(
                "Deepfake backend '%s' failed to initialize. Trying fallback backend 'deepfakebench_effnb4'.",
                self.deepfake_backend,
            )
            (
                self.deepfake_adapter,
                self.deepfake_backend,
                self.deepfake_detector_name,
                self.deepfake_model_version,
                self.deepfake_fake_threshold,
                self.deepfake_fake_class_index,
            ) = self._build_deepfake_adapter_for_backend("deepfakebench_effnb4", deepfake_device)

        if self.deepfake_backend == "deepfakebench_effnb4" and DEEPFAKE_FAKE_CLASS_INDEX != 1:
            if DEEPFAKE_AUTO_CORRECT_FAKE_CLASS_INDEX:
                logger.warning(
                    "Auto-corrected DEEPFAKE_FAKE_CLASS_INDEX from %s to %s for backend %s.",
                    DEEPFAKE_FAKE_CLASS_INDEX,
                    self.deepfake_fake_class_index,
                    self.deepfake_backend,
                )
            else:
                logger.warning(
                    "DEEPFAKE_FAKE_CLASS_INDEX is %s for backend %s. DeepfakeBench label mapping is usually fake=1 and real=0.",
                    DEEPFAKE_FAKE_CLASS_INDEX,
                    self.deepfake_backend,
                )

        logger.info(
            "Deepfake backend active: requested=%s active=%s model=%s version=%s",
            DEEPFAKE_MODEL_BACKEND,
            self.deepfake_backend,
            self.deepfake_detector_name,
            self.deepfake_model_version,
        )
        logger.info(
            "Deepfake config: fake_class_index=%s threshold=%.3f face_crops=%s full_frame_fallback=%s aggregation=%s",
            self.deepfake_fake_class_index,
            self.deepfake_fake_threshold,
            DEEPFAKE_USE_FACE_CROPS,
            DEEPFAKE_FULL_FRAME_FALLBACK,
            DEEPFAKE_SCORE_AGGREGATION,
        )

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=FRAME_ANALYSIS_MAX_WORKERS)
        logger.info("Frame analysis executor max_workers=%s", FRAME_ANALYSIS_MAX_WORKERS)

    def _build_deepfake_adapter_for_backend(
        self,
        backend: str,
        deepfake_device: "torch.device",
    ) -> tuple[Any, str, str, str, float, int]:
        if backend == "deepfakebench_effnb4":
            fake_class_index = resolve_fake_class_index_for_backend(
                backend=backend,
                default_index=DEEPFAKE_FAKE_CLASS_INDEX,
                backend_override=None,
                auto_correct=DEEPFAKE_AUTO_CORRECT_FAKE_CLASS_INDEX,
            )
            effnet_config = DeepfakeBenchEfficientNetB4Config(
                detector_checkpoint_path=DEEPFAKE_MODEL_PATH,
                detector_checkpoint_url=DEEPFAKE_MODEL_URL,
                backbone_weights_path=DEEPFAKE_BACKBONE_PATH,
                backbone_weights_url=DEEPFAKE_BACKBONE_URL,
                model_version=DEEPFAKE_MODEL_VERSION,
                auto_download=DEEPFAKE_AUTO_DOWNLOAD,
                input_size=DEEPFAKE_INPUT_SIZE,
            )
            adapter = DeepfakeBenchEfficientNetB4Adapter(
                config=effnet_config,
                device=deepfake_device,
                logger=logger,
            )
            return (
                adapter,
                backend,
                "DeepfakeBench-EfficientNetB4",
                DEEPFAKE_MODEL_VERSION,
                DEEPFAKE_FAKE_THRESHOLD,
                fake_class_index,
            )

        if backend == "deepfakebench_ucf":
            fake_threshold = DEEPFAKE_UCF_FAKE_THRESHOLD if DEEPFAKE_UCF_FAKE_THRESHOLD is not None else DEEPFAKE_FAKE_THRESHOLD
            fake_class_index = resolve_fake_class_index_for_backend(
                backend=backend,
                default_index=DEEPFAKE_FAKE_CLASS_INDEX,
                backend_override=DEEPFAKE_UCF_FAKE_CLASS_INDEX,
                auto_correct=False,
            )
            ucf_backbone_weights_path = (
                DEEPFAKE_UCF_BACKBONE_PATH
                if DEEPFAKE_UCF_BACKBONE_NAME == "efficientnet-b4"
                else None
            )
            ucf_backbone_weights_url = (
                DEEPFAKE_UCF_BACKBONE_URL
                if DEEPFAKE_UCF_BACKBONE_NAME == "efficientnet-b4"
                else ""
            )
            ucf_config = DeepfakeBenchUCFConfig(
                detector_checkpoint_path=DEEPFAKE_UCF_MODEL_PATH,
                detector_checkpoint_url=DEEPFAKE_UCF_MODEL_URL,
                backbone_name=DEEPFAKE_UCF_BACKBONE_NAME,
                backbone_weights_path=ucf_backbone_weights_path,
                backbone_weights_url=ucf_backbone_weights_url,
                model_version=DEEPFAKE_UCF_MODEL_VERSION,
                auto_download=DEEPFAKE_AUTO_DOWNLOAD,
                input_size=DEEPFAKE_UCF_INPUT_SIZE,
                encoder_feat_dim=DEEPFAKE_UCF_ENCODER_FEAT_DIM,
                num_classes=2,
                dropout=0.0,
            )
            adapter = DeepfakeBenchUCFAdapter(
                config=ucf_config,
                device=deepfake_device,
                logger=logger,
            )
            return (
                adapter,
                backend,
                f"DeepfakeBench-UCF-{DEEPFAKE_UCF_BACKBONE_NAME}",
                DEEPFAKE_UCF_MODEL_VERSION,
                fake_threshold,
                fake_class_index,
            )

        if DEEPFAKE_BACKEND_STRICT:
            raise ValueError(
                f"Unsupported deepfake backend '{backend}' in strict mode. Supported backends: {sorted(SUPPORTED_DEEPFAKE_BACKENDS)}"
            )

        logger.warning(
            "Unsupported deepfake backend '%s'. Defaulting to deepfakebench_effnb4.",
            backend,
        )
        return self._build_deepfake_adapter_for_backend("deepfakebench_effnb4", deepfake_device)

    def _infer_deepfake_from_frame(
        self,
        rgb_data_array: np.ndarray,
        face_boxes: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        start_time = time.time()
        detector_model_name = self.deepfake_detector_name

        if not self.deepfake_adapter.is_loaded:
            return {
                "model": detector_model_name,
                "model_version": self.deepfake_model_version,
                "result": "inconclusive",
                "fake_score": None,
                "confidence_score": 0.0,
                "flagged": False,
                "faces_evaluated": 0,
                "score_aggregation": DEEPFAKE_SCORE_AGGREGATION,
                "inference_time_ms": int((time.time() - start_time) * 1000),
            }

        try:
            candidate_regions: list[np.ndarray] = []
            if DEEPFAKE_USE_FACE_CROPS:
                candidate_regions = crop_face_regions(
                    rgb_data_array,
                    face_boxes,
                    margin_ratio=DEEPFAKE_FACE_MARGIN_RATIO,
                    min_face_size=DEEPFAKE_MIN_FACE_SIZE,
                )

            if not candidate_regions and DEEPFAKE_FULL_FRAME_FALLBACK:
                candidate_regions = [rgb_data_array]

            if not candidate_regions:
                raise RuntimeError("No valid regions available for deepfake inference")

            selected_scores: list[float] = []
            alternate_scores: list[float] = []
            for region in candidate_regions:
                probabilities = self.deepfake_adapter.infer_class_probabilities(region)
                if probabilities is None:
                    continue

                selected_index = (
                    self.deepfake_fake_class_index
                    if 0 <= self.deepfake_fake_class_index < len(probabilities)
                    else len(probabilities) - 1
                )
                selected_scores.append(float(probabilities[selected_index]))

                if len(probabilities) == 2:
                    alternate_scores.append(float(probabilities[1 - selected_index]))

            if not selected_scores:
                raise RuntimeError("Deepfake adapter returned no scores")

            fake_score = aggregate_fake_scores(selected_scores, mode=DEEPFAKE_SCORE_AGGREGATION)
            alternate_score = (
                aggregate_fake_scores(alternate_scores, mode=DEEPFAKE_SCORE_AGGREGATION)
                if alternate_scores
                else None
            )

            result, confidence_score = determine_deepfake_result_with_threshold(
                fake_score=fake_score,
                threshold=self.deepfake_fake_threshold,
                inconclusive_margin=DEEPFAKE_INCONCLUSIVE_MARGIN,
            )
            logger.info(
                "Deepfake inference: result=%s fake_score=%.4f alternate_score=%s confidence=%.4f regions=%s aggregation=%s fake_class_index=%s",
                result,
                fake_score,
                "none" if alternate_score is None else f"{alternate_score:.4f}",
                confidence_score,
                len(selected_scores),
                DEEPFAKE_SCORE_AGGREGATION,
                self.deepfake_fake_class_index,
            )

            return {
                "model": detector_model_name,
                "model_version": self.deepfake_model_version,
                "result": result,
                "fake_score": round(fake_score, 4),
                "alternate_score": None if alternate_score is None else round(alternate_score, 4),
                "confidence_score": round(confidence_score, 4),
                "flagged": result == "fake",
                "faces_evaluated": len(selected_scores),
                "fake_class_index": self.deepfake_fake_class_index,
                "score_aggregation": DEEPFAKE_SCORE_AGGREGATION,
                "inference_time_ms": int((time.time() - start_time) * 1000),
            }
        except Exception as error:
            logger.exception("Deepfake inference failed: %s", error)
            return {
                "model": detector_model_name,
                "model_version": self.deepfake_model_version,
                "result": "inconclusive",
                "fake_score": None,
                "confidence_score": 0.0,
                "flagged": False,
                "faces_evaluated": 0,
                "score_aggregation": DEEPFAKE_SCORE_AGGREGATION,
                "inference_time_ms": int((time.time() - start_time) * 1000),
            }

    def _extract_embedding_from_bgr(self, bgr_image: np.ndarray) -> np.ndarray | None:
        faces = self.face_app.get(bgr_image)
        if not faces:
            return None

        largest_face = max(
            faces,
            key=lambda face: float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])),
        )

        return normalize_embedding(getattr(largest_face, "embedding", None))

    def _analyze_frame(
        self,
        rgb_data_array: np.ndarray,
        patient_embedding: np.ndarray | None,
        patient_id: int | None,
        patient_name: str | None,
        doctor_embedding: np.ndarray | None,
        doctor_id: int | None,
        doctor_name: str | None,
        threshold: float,
        target_role: str | None,
    ) -> dict[str, Any]:
        import cv2

        start_time = time.time()
        bgr_data = cv2.cvtColor(rgb_data_array, cv2.COLOR_RGB2BGR)
        faces = self.face_app.get(bgr_data)
        elapsed_ms = int((time.time() - start_time) * 1000)

        bounding_boxes: list[list[float]] = []
        best_patient_candidate: dict[str, Any] | None = None
        best_doctor_candidate: dict[str, Any] | None = None

        check_patient = target_role in (None, "patient", "both")
        check_doctor = target_role in (None, "doctor", "both")

        for face in faces:
            box = face.bbox.astype(int).tolist()
            confidence = float(face.det_score)
            bounding_boxes.append([box[0], box[1], box[2], box[3], confidence])

            current_embedding = normalize_embedding(getattr(face, "embedding", None))

            # Check against patient reference
            if check_patient and patient_embedding is not None:
                patient_similarity = cosine_similarity(current_embedding, patient_embedding)
                if patient_similarity is not None:
                    candidate = {
                        "matched": patient_similarity >= threshold,
                        "similarity": patient_similarity,
                        "bounding_box": [box[0], box[1], box[2], box[3]],
                    }
                    if best_patient_candidate is None or candidate["similarity"] > best_patient_candidate["similarity"]:
                        best_patient_candidate = candidate

            # Check against doctor reference
            if check_doctor and doctor_embedding is not None:
                doctor_similarity = cosine_similarity(current_embedding, doctor_embedding)
                if doctor_similarity is not None:
                    candidate = {
                        "matched": doctor_similarity >= threshold,
                        "similarity": doctor_similarity,
                        "bounding_box": [box[0], box[1], box[2], box[3]],
                    }
                    if best_doctor_candidate is None or candidate["similarity"] > best_doctor_candidate["similarity"]:
                        best_doctor_candidate = candidate

        deepfake_results = self._infer_deepfake_from_frame(rgb_data_array, face_boxes=bounding_boxes)

        return {
            "model_A": {
                "model": "SCRFD",
                "faces_detected": len(faces),
                "bounding_boxes": bounding_boxes,
                "inference_time_ms": elapsed_ms,
            },
            "patient": {
                "model": "ArcFace",
                "patient_id": patient_id,
                "patient_name": patient_name,
                "reference_loaded": check_patient and patient_embedding is not None,
                "faces_checked": len(faces),
                "matched": None if best_patient_candidate is None else best_patient_candidate["matched"],
                "best_similarity": None if best_patient_candidate is None else round(float(best_patient_candidate["similarity"]), 4),
                "best_box": None if best_patient_candidate is None else best_patient_candidate["bounding_box"],
            },
            "doctor": {
                "model": "ArcFace",
                "doctor_id": doctor_id,
                "doctor_name": doctor_name,
                "reference_loaded": check_doctor and doctor_embedding is not None,
                "faces_checked": len(faces),
                "matched": None if best_doctor_candidate is None else best_doctor_candidate["matched"],
                "best_similarity": None if best_doctor_candidate is None else round(float(best_doctor_candidate["similarity"]), 4),
                "best_box": None if best_doctor_candidate is None else best_doctor_candidate["bounding_box"],
            },
            "deepfake": deepfake_results,
        }

    async def run_inference(
        self,
        rgb_data_array: np.ndarray,
        gallery: FaceGallery,
        target_role: str | None = None,
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            self._analyze_frame,
            rgb_data_array,
            gallery.patient_reference_embedding,
            gallery.patient_id,
            gallery.patient_name,
            gallery.doctor_reference_embedding,
            gallery.doctor_id,
            gallery.doctor_name,
            gallery.threshold,
            target_role,
        )

    async def compute_reference_embedding_from_url(
        self,
        session: aiohttp.ClientSession,
        photo_url: str,
    ) -> np.ndarray | None:
        import cv2

        resolved_url = resolve_asset_url(photo_url)

        try:
            async with session.get(resolved_url) as response:
                if response.status != 200:
                    logger.warning("Failed to fetch patient photo %s: HTTP %s", resolved_url, response.status)
                    return None

                image_bytes = await response.read()
        except Exception as error:
            logger.error("Failed to download patient photo %s: %s", resolved_url, error)
            return None

        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if bgr_image is None:
            logger.warning("Unable to decode patient photo from %s", resolved_url)
            return None

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._extract_embedding_from_bgr, bgr_image)


pipeline: PipelineManager | None = None
_pipeline_init_lock = threading.Lock()


def get_or_create_pipeline() -> PipelineManager:
    global pipeline

    if pipeline is None:
        with _pipeline_init_lock:
            if pipeline is None:
                logger.info("Lazy-loading InsightFace models on first video track...")
                pipeline = PipelineManager()

    return pipeline


async def post_internal_json(session: aiohttp.ClientSession, url: str, payload: dict[str, Any]) -> bool:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

    try:
        async with session.post(url, data=body.encode("utf-8"), headers=build_pipeline_signature_headers(body)) as response:
            if response.status not in (200, 201):
                response_text = await response.text()
                logger.warning(
                    "Internal pipeline endpoint %s returned HTTP %s body=%s",
                    url,
                    response.status,
                    response_text[:500],
                )
                return False

            return True
    except Exception as error:
        logger.error("Failed to post signed payload to %s: %s", url, error)
        return False


async def claim_due_microcheck(
    session: aiohttp.ClientSession,
    consultation_id: int,
    user_id: int | None,
    verified_role: str | None,
) -> dict[str, Any] | None:
    payload: dict[str, Any] = {
        "consultation_id": consultation_id,
    }

    if user_id is not None and verified_role in {"patient", "doctor"}:
        payload["user_id"] = user_id
        payload["verified_role"] = verified_role

    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

    try:
        async with session.post(
            build_internal_url("microchecks/claim"),
            data=body.encode("utf-8"),
            headers=build_pipeline_signature_headers(body),
        ) as response:
            if response.status != 200:
                response_text = await response.text()
                logger.warning(
                    "Microcheck claim endpoint returned HTTP %s body=%s",
                    response.status,
                    response_text[:500],
                )
                return None

            return await response.json()
    except Exception as error:
        logger.error("Failed to claim microcheck for consultation %s: %s", consultation_id, error)
        return None


async def send_frame_results(session: aiohttp.ClientSession, payload: dict[str, Any]) -> None:
    try:
        async with session.post(LARAVEL_ENDPOINT, json=payload) as response:
            if response.status not in (200, 201):
                logger.warning("Laravel frame endpoint returned HTTP %s", response.status)
    except Exception as error:
        logger.error("Failed to post frame results to Laravel: %s", error)


async def send_face_match_result(session: aiohttp.ClientSession, payload: dict[str, Any]) -> None:
    await post_internal_json(session, build_internal_url("face-match-results"), payload)


async def send_scan_result(session: aiohttp.ClientSession, payload: dict[str, Any]) -> None:
    await post_internal_json(session, build_internal_url("scan-results"), payload)


async def video_track_handler(
    track: rtc.RemoteVideoTrack,
    participant: rtc.RemoteParticipant | None,
    ctx: JobContext,
) -> None:
    import cv2

    logger.info("Video track subscribed: %s", track.sid)

    active_pipeline = get_or_create_pipeline()
    video_stream = rtc.VideoStream(track, capacity=VIDEO_STREAM_CAPACITY)
    logger.info("Video stream configured with capacity=%s for track=%s", VIDEO_STREAM_CAPACITY, track.sid)
    last_analyzed_timestamp_us: int | None = None
    analyzed_frame_number = 0
    os.makedirs(SAVED_FRAMES_DIR, exist_ok=True)

    async with aiohttp.ClientSession() as http_session:
        gallery = FaceGallery()
        await gallery.load_for_room(ctx.room.name, http_session, active_pipeline)

        participant_identity = str(getattr(participant, "identity", "unknown"))
        inferred_role, inferred_user_id = gallery.resolve_track_subject(participant)
        configured_target = VERIFICATION_TARGET if VERIFICATION_TARGET in {"patient", "doctor", "both"} else "both"
        target_role: str | None

        if PARTICIPANT_AWARE_VERIFICATION and inferred_role in {"patient", "doctor"}:
            target_role = inferred_role
            logger.info(
                "Participant-aware verification: identity=%s inferred_role=%s inferred_user_id=%s",
                participant_identity,
                inferred_role,
                inferred_user_id,
            )
        else:
            target_role = configured_target
            logger.info(
                "Fallback verification target: identity=%s target=%s participant_aware=%s inferred_role=%s inferred_user_id=%s",
                participant_identity,
                configured_target,
                PARTICIPANT_AWARE_VERIFICATION,
                inferred_role,
                inferred_user_id,
            )

        should_report_deepfake = should_report_deepfake_for_role(inferred_role)
        logger.info(
            "Deepfake reporting gate: role_mode=%s inferred_role=%s enabled=%s track=%s",
            DEEPFAKE_REPORTING_ROLE,
            inferred_role,
            should_report_deepfake,
            track.sid,
        )

        async for event in video_stream:
            frame_timestamp_us = int(event.timestamp_us)
            if not should_analyze_frame_timestamp(
                last_analyzed_timestamp_us,
                frame_timestamp_us,
                FRAME_ANALYSIS_INTERVAL_SECONDS,
            ):
                continue

            last_analyzed_timestamp_us = frame_timestamp_us
            analyzed_frame_number += 1

            rtc_frame = event.frame
            argb_frame = rtc_frame.convert(rtc.VideoBufferType.ARGB)
            frame_data = np.frombuffer(argb_frame.data, dtype=np.uint8)
            rgb_data = frame_data.reshape((argb_frame.height, argb_frame.width, 4))[:, :, 1:]

            inference_results = await active_pipeline.run_inference(rgb_data, gallery, target_role=target_role)

            bgr_frame = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
            detection_results = inference_results.get("model_A", {})
            patient_results = inference_results.get("patient", {})
            doctor_results = inference_results.get("doctor", {})
            deepfake_results = inference_results.get("deepfake", {})

            # Draw detection bounding boxes
            for box in detection_results.get("bounding_boxes", []):
                x1, y1, x2, y2, confidence = int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(box[4])
                cv2.rectangle(bgr_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    bgr_frame,
                    f"face {confidence:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            # Draw patient recognition box (blue for patient)
            if patient_results.get("best_box") is not None:
                x1, y1, x2, y2 = patient_results["best_box"]
                similarity = patient_results.get("best_similarity")
                matched = patient_results.get("matched")
                label = f"patient-match {similarity}" if matched else f"patient-nomatch {similarity}"
                color = (255, 0, 0) if matched else (0, 0, 255)  # Blue if match, Red if no match
                cv2.rectangle(bgr_frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(
                    bgr_frame,
                    label,
                    (x1, max(y1 - 35, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            # Draw doctor recognition box (cyan for doctor)
            if doctor_results.get("best_box") is not None:
                x1, y1, x2, y2 = doctor_results["best_box"]
                similarity = doctor_results.get("best_similarity")
                matched = doctor_results.get("matched")
                label = f"doctor-match {similarity}" if matched else f"doctor-nomatch {similarity}"
                color = (255, 255, 0) if matched else (0, 165, 255)  # Cyan if match, Orange if no match
                cv2.rectangle(bgr_frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(
                    bgr_frame,
                    label,
                    (x1, min(y2 + 35, argb_frame.height - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

            draw_text_overlay(bgr_frame, build_deepfake_overlay_lines(deepfake_results))

            frame_filename = os.path.join(
                SAVED_FRAMES_DIR,
                build_saved_frame_filename(gallery.consultation_id, analyzed_frame_number, event.timestamp_us, track.sid),
            )
            cv2.imwrite(frame_filename, bgr_frame)
            logger.info("Saved analyzed frame to %s", frame_filename)

            frame_payload = {
                "track_id": track.sid,
                "timestamp": frame_timestamp_us,
                "width": argb_frame.width,
                "height": argb_frame.height,
                "image": None,
                "ml_results": inference_results,
            }
            asyncio.create_task(send_frame_results(http_session, frame_payload))

            # Send separate match reports for patient and doctor
            if target_role in (None, "patient", "both") and patient_results.get("reference_loaded"):
                patient_report = gallery.build_match_report(patient_results, role="patient")
                if patient_report is not None:
                    asyncio.create_task(send_face_match_result(http_session, patient_report))

            if target_role in (None, "doctor", "both") and doctor_results.get("reference_loaded"):
                doctor_report = gallery.build_match_report(doctor_results, role="doctor")
                if doctor_report is not None:
                    asyncio.create_task(send_face_match_result(http_session, doctor_report))

            if should_report_deepfake and gallery.consultation_id is not None and deepfake_results:
                claim_role = inferred_role if inferred_role in {"patient", "doctor"} else None
                claim_user_id = inferred_user_id if claim_role is not None and inferred_user_id is not None else None

                claim_response = await claim_due_microcheck(
                    http_session,
                    consultation_id=gallery.consultation_id,
                    user_id=claim_user_id,
                    verified_role=claim_role,
                )

                if claim_user_id is None or claim_role is None:
                    logger.debug(
                        "Skipping deepfake scan submission due to unresolved participant identity. consultation_id=%s track=%s",
                        gallery.consultation_id,
                        track.sid,
                    )
                    continue

                if claim_response is None or not claim_response.get("claimed"):
                    continue

                microcheck_payload = claim_response.get("microcheck") or {}
                microcheck_id = microcheck_payload.get("id")
                if not isinstance(microcheck_id, int):
                    logger.warning("Claimed microcheck payload missing numeric id for consultation %s", gallery.consultation_id)
                    continue

                deepfake_scan_payload = build_scan_result_payload(
                    consultation_id=gallery.consultation_id,
                    microcheck_id=microcheck_id,
                    user_id=claim_user_id,
                    verified_role=claim_role,
                    deepfake_result=deepfake_results,
                    frame_path=frame_filename,
                    frame_number=analyzed_frame_number,
                )
                asyncio.create_task(send_scan_result(http_session, deepfake_scan_payload))


def prewarm(process: JobProcess) -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("Prewarm complete. Models will be loaded lazily on first track.")


async def entrypoint(ctx: JobContext) -> None:
    logger.info("Connected to room: %s", ctx.room.name)

    await ctx.connect(auto_subscribe=AutoSubscribe.VIDEO_ONLY)

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            asyncio.create_task(video_track_handler(track, participant, ctx))

    await asyncio.Future()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            initialize_process_timeout=60.0,
        )
    )
