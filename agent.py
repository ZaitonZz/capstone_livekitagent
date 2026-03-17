import asyncio
import concurrent.futures
import hashlib
import hmac
import json
import logging
import os
import time
from typing import Any
from urllib.parse import urljoin

import aiohttp
import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from face_recognition import FaceGallery as BaseFaceGallery, cosine_similarity, normalize_embedding
from insightface.app import FaceAnalysis
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


class FaceGallery(BaseFaceGallery):
    def __init__(
        self,
        threshold: float = FACE_MATCH_THRESHOLD,
        confirmation_streak: int = FACE_MATCH_STREAK_TARGET,
    ):
        super().__init__(threshold=threshold, confirmation_streak=confirmation_streak)

    async def load_for_room(
        self,
        room_name: str,
        session: aiohttp.ClientSession,
        pipeline_manager: "PipelineManager",
    ) -> bool:
        try:
            async with session.get(
                build_internal_url(f"consultation/{room_name}/patient-face"),
                headers=build_pipeline_signature_headers(""),
            ) as response:
                if response.status == 404:
                    logger.warning("No consultation face data found for room %s", room_name)
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

        self.consultation_id = payload.get("consultation_id")
        self.patient_id = payload.get("patient_id")
        self.patient_name = payload.get("patient_name")
        self.photo_id = payload.get("photo_id")
        self.reference_embedding = normalize_embedding(payload.get("face_embedding"))

        if self.reference_embedding is not None:
            logger.info("Loaded stored ArcFace embedding for patient %s", self.patient_id)
            return True

        photo_url = payload.get("photo_url")
        if photo_url is None or self.photo_id is None:
            logger.warning("Patient %s has no primary photo available for ArcFace enrollment", self.patient_id)
            return False

        computed_embedding = await pipeline_manager.compute_reference_embedding_from_url(session, photo_url)
        if computed_embedding is None:
            logger.warning("Unable to compute ArcFace embedding for patient photo %s", self.photo_id)
            return False

        stored = await post_internal_json(
            session,
            build_internal_url(f"face-embeddings/{self.photo_id}"),
            {"embedding": computed_embedding.astype(np.float32).tolist()},
        )
        if not stored:
            return False

        self.reference_embedding = normalize_embedding(computed_embedding)
        logger.info("Computed and stored ArcFace embedding for patient photo %s", self.photo_id)
        return self.reference_embedding is not None


class PipelineManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using compute device: %s", self.device)

        logger.info("Loading InsightFace buffalo_l with ArcFace recognition...")
        self.face_app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
        ctx_id = 0 if torch.cuda.is_available() else -1
        self.face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        logger.info("InsightFace loaded successfully.")

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

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
        reference_embedding: np.ndarray | None,
        patient_id: int | None,
        patient_name: str | None,
        threshold: float,
    ) -> dict[str, Any]:
        start_time = time.time()
        bgr_data = cv2.cvtColor(rgb_data_array, cv2.COLOR_RGB2BGR)
        faces = self.face_app.get(bgr_data)
        elapsed_ms = int((time.time() - start_time) * 1000)

        bounding_boxes: list[list[float]] = []
        best_candidate: dict[str, Any] | None = None

        for face in faces:
            box = face.bbox.astype(int).tolist()
            confidence = float(face.det_score)
            bounding_boxes.append([box[0], box[1], box[2], box[3], confidence])

            current_embedding = normalize_embedding(getattr(face, "embedding", None))
            similarity = cosine_similarity(current_embedding, reference_embedding)
            if similarity is None:
                continue

            candidate = {
                "matched": similarity >= threshold,
                "similarity": similarity,
                "bounding_box": [box[0], box[1], box[2], box[3]],
            }

            if best_candidate is None or candidate["similarity"] > best_candidate["similarity"]:
                best_candidate = candidate

        return {
            "model_A": {
                "model": "SCRFD",
                "faces_detected": len(faces),
                "bounding_boxes": bounding_boxes,
                "inference_time_ms": elapsed_ms,
            },
            "model_B": {
                "model": "ArcFace",
                "patient_id": patient_id,
                "patient_name": patient_name,
                "reference_loaded": reference_embedding is not None,
                "faces_checked": len(faces),
                "matched": None if best_candidate is None else best_candidate["matched"],
                "best_similarity": None if best_candidate is None else round(float(best_candidate["similarity"]), 4),
                "best_box": None if best_candidate is None else best_candidate["bounding_box"],
            },
        }

    async def run_inference(self, rgb_data_array: np.ndarray, gallery: FaceGallery) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            self._analyze_frame,
            rgb_data_array,
            gallery.reference_embedding,
            gallery.patient_id,
            gallery.patient_name,
            gallery.threshold,
        )

    async def compute_reference_embedding_from_url(
        self,
        session: aiohttp.ClientSession,
        photo_url: str,
    ) -> np.ndarray | None:
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


async def post_internal_json(session: aiohttp.ClientSession, url: str, payload: dict[str, Any]) -> bool:
    body = json.dumps(payload)

    try:
        async with session.post(url, data=body, headers=build_pipeline_signature_headers(body)) as response:
            if response.status not in (200, 201):
                logger.warning("Internal pipeline endpoint %s returned HTTP %s", url, response.status)
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


async def video_track_handler(track: rtc.RemoteVideoTrack, ctx: JobContext) -> None:
    logger.info("Video track subscribed: %s", track.sid)

    video_stream = rtc.VideoStream(track)
    last_post_time = 0.0
    os.makedirs(SAVED_FRAMES_DIR, exist_ok=True)

    async with aiohttp.ClientSession() as http_session:
        gallery = FaceGallery()
        await gallery.load_for_room(ctx.room.name, http_session, pipeline)

        async for event in video_stream:
            current_time = time.time()
            if current_time - last_post_time < FRAME_ANALYSIS_INTERVAL_SECONDS:
                continue

            last_post_time = current_time

            rtc_frame = event.frame
            argb_frame = rtc_frame.convert(rtc.VideoBufferType.ARGB)
            frame_data = np.frombuffer(argb_frame.data, dtype=np.uint8)
            rgb_data = frame_data.reshape((argb_frame.height, argb_frame.width, 4))[:, :, 1:]

            inference_results = await pipeline.run_inference(rgb_data, gallery)

            bgr_frame = cv2.cvtColor(rgb_data, cv2.COLOR_RGB2BGR)
            detection_results = inference_results.get("model_A", {})
            recognition_results = inference_results.get("model_B", {})

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

            if recognition_results.get("best_box") is not None:
                x1, y1, x2, y2 = recognition_results["best_box"]
                similarity = recognition_results.get("best_similarity")
                matched = recognition_results.get("matched")
                label = "match" if matched else "mismatch"
                color = (0, 255, 0) if matched else (0, 0, 255)
                cv2.rectangle(bgr_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    bgr_frame,
                    f"{label} {similarity}",
                    (x1, min(y2 + 25, argb_frame.height - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

            frame_filename = os.path.join(SAVED_FRAMES_DIR, "latest_frame.jpg")
            cv2.imwrite(frame_filename, bgr_frame)
            logger.info("Saved latest frame to %s", frame_filename)

            frame_payload = {
                "track_id": track.sid,
                "timestamp": event.timestamp_us,
                "width": argb_frame.width,
                "height": argb_frame.height,
                "image": None,
                "ml_results": inference_results,
            }
            asyncio.create_task(send_frame_results(http_session, frame_payload))

            report_payload = gallery.build_match_report(recognition_results)
            if report_payload is not None:
                asyncio.create_task(send_face_match_result(http_session, report_payload))


def prewarm(process: JobProcess) -> None:
    global pipeline
    logging.basicConfig(level=logging.INFO)
    logger.info("Initializing agent and loading InsightFace models...")
    pipeline = PipelineManager()


async def entrypoint(ctx: JobContext) -> None:
    logger.info("Connected to room: %s", ctx.room.name)

    global pipeline
    if pipeline is None:
        pipeline = PipelineManager()

    await ctx.connect(auto_subscribe=AutoSubscribe.VIDEO_ONLY)

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            asyncio.create_task(video_track_handler(track, ctx))

    await asyncio.Future()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
