import asyncio
import logging
import os
import concurrent.futures
import json
import time
import base64
import io
import numpy as np
import aiohttp
from dotenv import load_dotenv
import cv2
import insightface
from insightface.app import FaceAnalysis

from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
)
from livekit import rtc

# For your PyTorch Models
import torch
from PIL import Image

load_dotenv()
logger = logging.getLogger("video-pipeline-agent")

LARAVEL_ENDPOINT = os.getenv("LARAVEL_ENDPOINT", "http://localhost:8000/api/frame-results")

class PipelineManager:
    """
    Manages loading the local .pth models and running them concurrently
    so as not to block the LiveKit async video ingestion loop.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using compute device: {self.device}")
        
        # Initialize actual InsightFace (SCRFD) model for detection
        # Note: 'buffalo_l' is a standard high-accuracy pack covering detection/recognition.
        # It automatically downloads models to ~/.insightface/models/ on first run.
        logger.info("Loading SCRFD InsightFace Model...")
        self.face_app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection'])
        ctx_id = 0 if torch.cuda.is_available() else -1
        self.face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        logger.info("SCRFD Loaded successfully!")
        
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def _inference_model_a(self, tensor_frame, rgb_data_array=None):
        """Synchronous CPU/GPU bound inference for Model A (Actual SCRFD)"""
        if rgb_data_array is None:
            return {"error": "No image data"}

        start_time = time.time()
        
        # insightface expects BGR format (OpenCV standard)
        bgr_data = cv2.cvtColor(rgb_data_array, cv2.COLOR_RGB2BGR)
        
        # Run SCRFD
        faces = self.face_app.get(bgr_data)
        
        elapsed_ms = int((time.time() - start_time) * 1000)

        bounding_boxes = []
        for face in faces:
            # bbox is naturally [x1, y1, x2, y2]
            box = face.bbox.astype(int).tolist()
            conf = float(face.det_score)
            bounding_boxes.append([box[0], box[1], box[2], box[3], conf])

        return {
            "model": "SCRFD",
            "faces_detected": len(faces),
            "bounding_boxes": bounding_boxes,
            "inference_time_ms": elapsed_ms
        }

    def _inference_model_b(self, tensor_frame):
        """Synchronous CPU/GPU bound inference for Model B"""
        # with torch.no_grad():
        #     output = self.model_b(tensor_frame)
        # return output.tolist()
        return {"simulated_result": "B-ok"}

    async def run_parallel_inference(self, rgb_data_array):
        loop = asyncio.get_running_loop()
        
        # 1. Transform numpy array to Torch Tensor
        # E.g.: [H, W, 3] -> [3, H, W] for PyTorch
        # tensor_frame = torch.from_numpy(rgb_data_array).permute(2, 0, 1).float()
        # tensor_frame = tensor_frame.unsqueeze(0).to(self.device) # Add batch dimension
        tensor_frame = None  # Using None for placeholder

        # 2. Run inference in parallel using thread pool
        # PyTorch generally releases the Global Interpreter Lock (GIL) 
        # so threading works well for concurrent ML tasks here.
        task_a = loop.run_in_executor(self.executor, self._inference_model_a, tensor_frame, rgb_data_array)
        task_b = loop.run_in_executor(self.executor, self._inference_model_b, tensor_frame)
        
        results = await asyncio.gather(task_a, task_b)
        
        return {
            "model_A": results[0],
            "model_B": results[1]
        }

# Global manager to hold models in memory
pipeline = None

async def send_to_laravel(session: aiohttp.ClientSession, payload: dict):
    """Fire-and-forget HTTP POST to Laravel."""
    try:
        async with session.post(LARAVEL_ENDPOINT, json=payload) as response:
            if response.status not in (200, 201):
                logger.warning(f"Laravel returned HTTP {response.status}")
    except Exception as e:
        logger.error(f"Failed to post to Laravel: {e}")

async def video_track_handler(track: rtc.RemoteVideoTrack, ctx: JobContext):
    """Processes incoming frames from a video track."""
    logger.info(f"Video track subscribed: {track.sid}")
    
    video_stream = rtc.VideoStream(track)
    last_post_time = 0
    
    # Maintain a session for repeated HTTP requests to the backend
    async with aiohttp.ClientSession() as http_session:
        async for event in video_stream:
            current_time = time.time()
            # Throttle to 1 update per second to avoid overloading Laravel backend
            if current_time - last_post_time < 1.0:
                continue
            last_post_time = current_time

            # `event.frame` contains the raw rtc.VideoFrame (often I420)
            rtc_frame = event.frame
            
            # Convert frame format to ARGB to easily get standard RGB array
            argb_frame = rtc_frame.convert(rtc.VideoBufferType.ARGB)
            frame_data = np.frombuffer(argb_frame.data, dtype=np.uint8)
            
            # Reshape based on height and width, and 4 channels (A, R, G, B)
            # and slice off Alpha to just get RGB
            rgb_data = frame_data.reshape((argb_frame.height, argb_frame.width, 4))[:, :, 1:]
            
            # Encode image as base64 JPEG for verification UI
            img = Image.fromarray(rgb_data, 'RGB')
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=60)
            img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            image_data_uri = f"data:image/jpeg;base64,{img_b64}"
            
            # Run the parallel pipeline over the frame
            inference_results = await pipeline.run_parallel_inference(rgb_data)
            
            # Prepare payload for Laravel
            payload = {
                "track_id": track.sid,
                "timestamp": event.timestamp_us,
                "width": argb_frame.width,
                "height": argb_frame.height,
                "image": image_data_uri,
                "ml_results": inference_results
            }
            
            # Send results without blocking the next frame ingestion
            asyncio.create_task(send_to_laravel(http_session, payload))


def prewarm(process: JobProcess):
    """Called when the worker starts before handling any sessions."""
    global pipeline
    logging.basicConfig(level=logging.INFO)
    logger.info("Initializing Agent and loading PyTorch Models...")
    pipeline = PipelineManager()

async def entrypoint(ctx: JobContext):
    """Entrypoint function called whenever a new session/room connects to this agent."""
    logger.info(f"Connected to room: {ctx.room.name}")
    
    # Tell the room we only care about video tracks
    await ctx.connect(auto_subscribe=AutoSubscribe.VIDEO_ONLY)

    # Listen for new video tracks being published to the room
    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            # Start a background task for processing the new video stream
            asyncio.create_task(video_track_handler(track, ctx))

    # Keep the session context alive
    await asyncio.Future()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
