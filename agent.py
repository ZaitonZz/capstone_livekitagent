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

import aiohttp
import numpy as np
from deepfakebench_effnet import (
    DeepfakeBenchEfficientNetB4Adapter,
    DeepfakeBenchEfficientNetB4Config,
    aggregate_fake_scores,
    crop_face_regions,
)
from dotenv import load_dotenv
from face_recognition import FaceGallery as BaseFaceGallery, cosine_similarity, normalize_embedding
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, JobProcess, WorkerOptions, cli

load_dotenv()

logger = logging.getLogger("video-pipeline-agent")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LARAVEL_BASE_URL = os.getenv("LARAVEL_BASE_URL", "http://localhost:8000").rstrip("/")
LARAVEL_ENDPOINT = os.getenv("LARAVEL_ENDPOINT", f"{LARAVEL_BASE_URL}/api/frame-results")
PIPELINE_INTERNAL_BASE_URL = os.getenv(
    "PIPELINE_INTERNAL_BASE_URL",
    f"{LARAVEL_BASE_URL}/internal/pipeline",
).rstrip("/")
PIPELINE_SHARED_SECRET = os.getenv("PIPELINE_SHARED_SECRET", "")
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.4"))
FRAME_ANALYSIS_INTERVAL_SECONDS = float(os.getenv("FRAME_ANALYSIS_INTERVAL_SECONDS", "5.0"))
FACE_MATCH_STREAK_TARGET = int(os.getenv("FACE_MATCH_STREAK_TARGET", "3"))
SAVED_FRAMES_DIR = os.getenv("SAVED_FRAMES_DIR", "saved_frames")
REPORT_MISSING_REFERENCE_AS_FLAG = os.getenv("REPORT_MISSING_REFERENCE_AS_FLAG", "true").lower() == "true"
VERIFICATION_TARGET = os.getenv("VERIFICATION_TARGET", "both").strip().lower()
PARTICIPANT_AWARE_VERIFICATION = os.getenv("PARTICIPANT_AWARE_VERIFICATION", "true").strip().lower() == "true"
DEEPFAKE_MODEL_BACKEND = os.getenv("DEEPFAKE_MODEL_BACKEND", "deepfakebench_effnb4").strip().lower()
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
DEEPFAKE_AUTO_DOWNLOAD = os.getenv("DEEPFAKE_AUTO_DOWNLOAD", "true").strip().lower() == "true"
DEEPFAKE_PREFER_CPU = os.getenv("DEEPFAKE_PREFER_CPU", "true").strip().lower() == "true"
DEEPFAKE_MODEL_VERSION = os.getenv("DEEPFAKE_MODEL_VERSION", "deepfakebench_effnb4")
DEEPFAKE_INPUT_SIZE = int(os.getenv("DEEPFAKE_INPUT_SIZE", "256"))
DEEPFAKE_FAKE_THRESHOLD = float(os.getenv("DEEPFAKE_FAKE_THRESHOLD", "0.5"))
DEEPFAKE_INCONCLUSIVE_MARGIN = float(os.getenv("DEEPFAKE_INCONCLUSIVE_MARGIN", "0.05"))
# Some released EfficientNet checkpoints are observed to emit fake-probability on class index 0.
# Keep this env-overridable for fast on-site tuning.
DEEPFAKE_FAKE_CLASS_INDEX = int(os.getenv("DEEPFAKE_FAKE_CLASS_INDEX", "0"))
DEEPFAKE_USE_FACE_CROPS = os.getenv("DEEPFAKE_USE_FACE_CROPS", "true").strip().lower() == "true"
DEEPFAKE_FULL_FRAME_FALLBACK = os.getenv("DEEPFAKE_FULL_FRAME_FALLBACK", "false").strip().lower() == "true"
DEEPFAKE_FACE_MARGIN_RATIO = float(os.getenv("DEEPFAKE_FACE_MARGIN_RATIO", "0.25"))
DEEPFAKE_MIN_FACE_SIZE = int(os.getenv("DEEPFAKE_MIN_FACE_SIZE", "48"))
DEEPFAKE_SCORE_AGGREGATION = os.getenv("DEEPFAKE_SCORE_AGGREGATION", "max").strip().lower()
DEEPFAKE_REPORTING_ROLE = os.getenv("DEEPFAKE_REPORTING_ROLE", "patient").strip().lower()


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


def determine_deepfake_result(fake_score: float) -> tuple[str, float]:
    bounded_score = max(0.0, min(1.0, float(fake_score)))
    if abs(bounded_score - DEEPFAKE_FAKE_THRESHOLD) <= DEEPFAKE_INCONCLUSIVE_MARGIN:
        return "inconclusive", max(bounded_score, 1.0 - bounded_score)

    if bounded_score >= DEEPFAKE_FAKE_THRESHOLD:
        return "fake", bounded_score

    return "real", 1.0 - bounded_score


def normalize_track_id(track_id: str) -> str:
    safe = "".join(character if character.isalnum() else "-" for character in track_id.strip())
    return safe.strip("-") or "unknown-track"


def should_report_deepfake_for_role(inferred_role: str | None) -> bool:
    if DEEPFAKE_REPORTING_ROLE == "both":
        return True

    if DEEPFAKE_REPORTING_ROLE in {"patient", "doctor"}:
        return inferred_role == DEEPFAKE_REPORTING_ROLE

    return True


def build_saved_frame_filename(consultation_id: int | None, frame_number: int, timestamp_us: int, track_id: str) -> str:
    consultation_fragment = str(consultation_id) if consultation_id is not None else "unknown"
    track_fragment = normalize_track_id(track_id)
    return f"consultation-{consultation_fragment}_track-{track_fragment}_frame-{frame_number:06d}_{timestamp_us}.jpg"


def build_scan_result_payload(
    consultation_id: int,
    deepfake_result: dict[str, Any],
    frame_path: str,
    frame_number: int,
) -> dict[str, Any]:
    result = str(deepfake_result.get("result", "inconclusive"))
    confidence_score = round(float(deepfake_result.get("confidence_score", 0.0)), 4)
    flagged = bool(deepfake_result.get("flagged", result == "fake"))

    return {
        "consultation_id": consultation_id,
        "result": result,
        "confidence_score": confidence_score,
        "frame_path": frame_path,
        "frame_number": frame_number,
        "model_version": str(deepfake_result.get("model_version", DEEPFAKE_MODEL_VERSION)),
        "flagged": flagged,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
    }


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

    def resolve_track_subject_role(self, participant: rtc.RemoteParticipant | None) -> str | None:
        if participant is None:
            return None

        metadata_raw = getattr(participant, "metadata", None)
        if isinstance(metadata_raw, str) and metadata_raw.strip() != "":
            try:
                metadata = json.loads(metadata_raw)
                role_from_metadata = str(metadata.get("role", "")).strip().lower()
                if role_from_metadata in {"doctor", "patient"}:
                    return role_from_metadata
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

        if user_id is not None:
            if self.patient_id is not None and user_id == self.patient_id:
                return "patient"
            if self.doctor_id is not None and user_id == self.doctor_id:
                return "doctor"

        return None


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

        if DEEPFAKE_MODEL_BACKEND != "deepfakebench_effnb4":
            logger.warning(
                "Unsupported deepfake backend '%s', falling back to deepfakebench_effnb4",
                DEEPFAKE_MODEL_BACKEND,
            )

        deepfake_device = torch.device("cpu")
        if not DEEPFAKE_PREFER_CPU and torch.cuda.is_available():
            deepfake_device = torch.device("cuda")

        deepfake_config = DeepfakeBenchEfficientNetB4Config(
            detector_checkpoint_path=DEEPFAKE_MODEL_PATH,
            detector_checkpoint_url=DEEPFAKE_MODEL_URL,
            backbone_weights_path=DEEPFAKE_BACKBONE_PATH,
            backbone_weights_url=DEEPFAKE_BACKBONE_URL,
            model_version=DEEPFAKE_MODEL_VERSION,
            auto_download=DEEPFAKE_AUTO_DOWNLOAD,
            input_size=DEEPFAKE_INPUT_SIZE,
        )
        self.deepfake_adapter = DeepfakeBenchEfficientNetB4Adapter(
            config=deepfake_config,
            device=deepfake_device,
            logger=logger,
        )
        logger.info(
            "Deepfake config: fake_class_index=%s threshold=%.3f face_crops=%s full_frame_fallback=%s aggregation=%s",
            DEEPFAKE_FAKE_CLASS_INDEX,
            DEEPFAKE_FAKE_THRESHOLD,
            DEEPFAKE_USE_FACE_CROPS,
            DEEPFAKE_FULL_FRAME_FALLBACK,
            DEEPFAKE_SCORE_AGGREGATION,
        )

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def _infer_deepfake_from_frame(
        self,
        rgb_data_array: np.ndarray,
        face_boxes: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        start_time = time.time()
        detector_model_name = "DeepfakeBench-EfficientNetB4"

        if not self.deepfake_adapter.is_loaded:
            return {
                "model": detector_model_name,
                "model_version": DEEPFAKE_MODEL_VERSION,
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

            fake_scores: list[float] = []
            for region in candidate_regions:
                score = self.deepfake_adapter.infer_fake_score(
                    region,
                    fake_class_index=DEEPFAKE_FAKE_CLASS_INDEX,
                )
                if score is not None:
                    fake_scores.append(score)

            if not fake_scores:
                raise RuntimeError("Deepfake adapter returned no scores")

            fake_score = aggregate_fake_scores(fake_scores, mode=DEEPFAKE_SCORE_AGGREGATION)

            result, confidence_score = determine_deepfake_result(fake_score)
            logger.info(
                "Deepfake inference: result=%s fake_score=%.4f confidence=%.4f regions=%s aggregation=%s",
                result,
                fake_score,
                confidence_score,
                len(fake_scores),
                DEEPFAKE_SCORE_AGGREGATION,
            )

            return {
                "model": detector_model_name,
                "model_version": DEEPFAKE_MODEL_VERSION,
                "result": result,
                "fake_score": round(fake_score, 4),
                "confidence_score": round(confidence_score, 4),
                "flagged": result == "fake",
                "faces_evaluated": len(fake_scores),
                "score_aggregation": DEEPFAKE_SCORE_AGGREGATION,
                "inference_time_ms": int((time.time() - start_time) * 1000),
            }
        except Exception as error:
            logger.exception("Deepfake inference failed: %s", error)
            return {
                "model": detector_model_name,
                "model_version": DEEPFAKE_MODEL_VERSION,
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
    video_stream = rtc.VideoStream(track)
    last_post_time = 0.0
    analyzed_frame_number = 0
    os.makedirs(SAVED_FRAMES_DIR, exist_ok=True)

    async with aiohttp.ClientSession() as http_session:
        gallery = FaceGallery()
        await gallery.load_for_room(ctx.room.name, http_session, active_pipeline)

        participant_identity = str(getattr(participant, "identity", "unknown"))
        inferred_role = gallery.resolve_track_subject_role(participant)
        configured_target = VERIFICATION_TARGET if VERIFICATION_TARGET in {"patient", "doctor", "both"} else "both"
        target_role: str | None

        if PARTICIPANT_AWARE_VERIFICATION and inferred_role in {"patient", "doctor"}:
            target_role = inferred_role
            logger.info(
                "Participant-aware verification: identity=%s inferred_role=%s",
                participant_identity,
                inferred_role,
            )
        else:
            target_role = configured_target
            logger.info(
                "Fallback verification target: identity=%s target=%s participant_aware=%s inferred_role=%s",
                participant_identity,
                configured_target,
                PARTICIPANT_AWARE_VERIFICATION,
                inferred_role,
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
            current_time = time.time()
            if current_time - last_post_time < FRAME_ANALYSIS_INTERVAL_SECONDS:
                continue

            last_post_time = current_time
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

            frame_filename = os.path.join(
                SAVED_FRAMES_DIR,
                build_saved_frame_filename(gallery.consultation_id, analyzed_frame_number, event.timestamp_us, track.sid),
            )
            cv2.imwrite(frame_filename, bgr_frame)
            logger.info("Saved analyzed frame to %s", frame_filename)

            frame_payload = {
                "track_id": track.sid,
                "timestamp": event.timestamp_us,
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
                deepfake_scan_payload = build_scan_result_payload(
                    consultation_id=gallery.consultation_id,
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
