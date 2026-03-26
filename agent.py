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
from dotenv import load_dotenv
from face_recognition import FaceGallery as BaseFaceGallery, cosine_similarity, normalize_embedding
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, JobProcess, WorkerOptions, cli

load_dotenv()

logger = logging.getLogger("video-pipeline-agent")

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
DEEPFAKE_MODEL_PATH = os.getenv(
    "DEEPFAKE_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "df_detector_efficientnet_v2_s_best.pth"),
)
DEEPFAKE_MODEL_VERSION = os.getenv("DEEPFAKE_MODEL_VERSION", "efficientnet_v2_s")
DEEPFAKE_INPUT_SIZE = int(os.getenv("DEEPFAKE_INPUT_SIZE", "384"))
DEEPFAKE_FAKE_THRESHOLD = float(os.getenv("DEEPFAKE_FAKE_THRESHOLD", "0.5"))
DEEPFAKE_INCONCLUSIVE_MARGIN = float(os.getenv("DEEPFAKE_INCONCLUSIVE_MARGIN", "0.05"))
DEEPFAKE_FAKE_CLASS_INDEX = int(os.getenv("DEEPFAKE_FAKE_CLASS_INDEX", "1"))
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
    def _looks_like_state_dict(self, value: Any) -> bool:
        if hasattr(value, "state_dict") and callable(value.state_dict):
            try:
                return self._looks_like_state_dict(value.state_dict())
            except Exception:
                return False

        if not isinstance(value, dict) or len(value) == 0:
            return False

        sample_keys = list(value.keys())[:20]
        if not all(isinstance(key, str) for key in sample_keys):
            return False

        sample_values = list(value.values())[:20]
        return any(hasattr(sample, "shape") for sample in sample_values)

    def _extract_checkpoint_state_dict(self, checkpoint: Any) -> dict[str, Any]:
        pending: list[Any] = [checkpoint]
        visited: set[int] = set()

        while pending:
            current = pending.pop(0)
            current_id = id(current)
            if current_id in visited:
                continue
            visited.add(current_id)

            if hasattr(current, "state_dict") and callable(current.state_dict):
                try:
                    module_state = current.state_dict()
                    if self._looks_like_state_dict(module_state):
                        return module_state

                    if isinstance(module_state, dict):
                        pending.extend(module_state.values())
                except Exception:
                    pass

            if isinstance(current, dict):
                if self._looks_like_state_dict(current):
                    return current

                for candidate_key in ["state_dict", "model_state_dict", "model", "net", "network", "weights", "backbone"]:
                    if candidate_key in current:
                        pending.append(current[candidate_key])

        return {}

    def _normalize_checkpoint_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        prefixes = ["module.", "model.", "_orig_mod.", "net.", "network.", "backbone."]

        for raw_key, value in state_dict.items():
            key = raw_key
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if key.startswith(prefix):
                        key = key[len(prefix):]
                        changed = True

            if key == "classifier.weight":
                key = "classifier.1.weight"
            elif key == "classifier.bias":
                key = "classifier.1.bias"

            normalized[key] = value

        return normalized

    def __init__(self):
        import torch
        import torchvision.models as tv_models
        from insightface.app import FaceAnalysis

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using compute device: %s", self.device)

        logger.info("Loading InsightFace buffalo_l with ArcFace recognition...")
        self.face_app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
        ctx_id = 0 if torch.cuda.is_available() else -1
        self.face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        logger.info("InsightFace loaded successfully.")

        self.deepfake_model = tv_models.efficientnet_v2_s(weights=None)
        classifier_input_features = self.deepfake_model.classifier[1].in_features
        self.deepfake_model.classifier[1] = torch.nn.Linear(classifier_input_features, 1)
        self.deepfake_output_classes = 1

        self.deepfake_model_loaded = False
        if os.path.exists(DEEPFAKE_MODEL_PATH):
            try:
                checkpoint = torch.load(DEEPFAKE_MODEL_PATH, map_location=self.device)

                state_dict = self._extract_checkpoint_state_dict(checkpoint)
                state_dict = self._normalize_checkpoint_state_dict(state_dict)

                if not state_dict:
                    logger.error("Deepfake checkpoint does not contain a usable state_dict: %s", DEEPFAKE_MODEL_PATH)
                    self.deepfake_model_loaded = False
                    self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
                    return

                classifier_weight = state_dict.get("classifier.1.weight")
                if classifier_weight is not None and getattr(classifier_weight, "ndim", 0) == 2:
                    checkpoint_output_classes = int(classifier_weight.shape[0])
                    if checkpoint_output_classes > 0 and checkpoint_output_classes != self.deepfake_output_classes:
                        self.deepfake_model.classifier[1] = torch.nn.Linear(classifier_input_features, checkpoint_output_classes)
                        self.deepfake_output_classes = checkpoint_output_classes

                missing_keys, unexpected_keys = self.deepfake_model.load_state_dict(state_dict, strict=False)
                self.deepfake_model.to(self.device)
                self.deepfake_model.eval()
                if missing_keys:
                    logger.warning("Deepfake model missing keys while loading: %s", missing_keys)
                if unexpected_keys:
                    logger.warning("Deepfake model unexpected keys while loading: %s", unexpected_keys)

                model_state_keys = set(self.deepfake_model.state_dict().keys())
                matched_keys = model_state_keys.intersection(state_dict.keys())
                critical_keys_present = any(
                    key.startswith("features.") or key.startswith("classifier.")
                    for key in matched_keys
                )

                self.deepfake_model_loaded = critical_keys_present and len(matched_keys) >= 10
                if self.deepfake_model_loaded:
                    logger.info(
                        "Deepfake model loaded from %s with %s output class(es), matched_keys=%s/%s",
                        DEEPFAKE_MODEL_PATH,
                        self.deepfake_output_classes,
                        len(matched_keys),
                        len(model_state_keys),
                    )
                else:
                    logger.error(
                        "Deepfake model load incomplete; matched_keys=%s/%s. Model will report inconclusive until key mapping is fixed",
                        len(matched_keys),
                        len(model_state_keys),
                    )
            except Exception as error:
                logger.exception("Failed to load deepfake model from %s: %s", DEEPFAKE_MODEL_PATH, error)
        else:
            logger.warning("Deepfake model path does not exist: %s", DEEPFAKE_MODEL_PATH)

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def _infer_deepfake_from_frame(self, rgb_data_array: np.ndarray) -> dict[str, Any]:
        import cv2
        import torch

        start_time = time.time()

        if not self.deepfake_model_loaded:
            return {
                "model": "EfficientNetV2-S",
                "model_version": DEEPFAKE_MODEL_VERSION,
                "result": "inconclusive",
                "fake_score": None,
                "confidence_score": 0.0,
                "flagged": False,
                "inference_time_ms": int((time.time() - start_time) * 1000),
            }

        try:
            resized_rgb = cv2.resize(rgb_data_array, (DEEPFAKE_INPUT_SIZE, DEEPFAKE_INPUT_SIZE), interpolation=cv2.INTER_AREA)
            normalized = resized_rgb.astype(np.float32) / 255.0

            tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(1, 3, 1, 1)
            tensor = (tensor - mean) / std
            tensor = tensor.to(self.device)

            with torch.no_grad():
                logits = self.deepfake_model(tensor)

            fake_score: float
            flattened_logits = logits.flatten()

            if self.deepfake_output_classes == 1 or flattened_logits.numel() == 1:
                fake_score = float(torch.sigmoid(flattened_logits[0]).item())
            else:
                probabilities = torch.softmax(flattened_logits, dim=0)
                target_index = DEEPFAKE_FAKE_CLASS_INDEX if DEEPFAKE_FAKE_CLASS_INDEX < probabilities.numel() else probabilities.numel() - 1
                fake_score = float(probabilities[target_index].item())

            result, confidence_score = determine_deepfake_result(fake_score)

            return {
                "model": "EfficientNetV2-S",
                "model_version": DEEPFAKE_MODEL_VERSION,
                "result": result,
                "fake_score": round(fake_score, 4),
                "confidence_score": round(confidence_score, 4),
                "flagged": result == "fake",
                "inference_time_ms": int((time.time() - start_time) * 1000),
            }
        except Exception as error:
            logger.exception("Deepfake inference failed: %s", error)
            return {
                "model": "EfficientNetV2-S",
                "model_version": DEEPFAKE_MODEL_VERSION,
                "result": "inconclusive",
                "fake_score": None,
                "confidence_score": 0.0,
                "flagged": False,
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

        deepfake_results = self._infer_deepfake_from_frame(rgb_data_array)

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
