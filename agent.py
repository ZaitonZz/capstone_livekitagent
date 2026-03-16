import asyncio
import logging
import os
import concurrent.futures
import json
import numpy as np
import aiohttp
from dotenv import load_dotenv

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
        
        # TODO: Define and load your specific .pth models here
        # E.g.,
        # self.model_a = MyCustomModelA().to(self.device)
        # self.model_a.load_state_dict(torch.load("models/model_A.pth", map_location=self.device))
        # self.model_a.eval()
        #
        # self.model_b = MyCustomModelB().to(self.device)
        # self.model_b.load_state_dict(torch.load("models/model_B.pth", map_location=self.device))
        # self.model_b.eval()
        
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def _inference_model_a(self, tensor_frame):
        """Synchronous CPU/GPU bound inference for Model A (Simulated SCRFD)"""
        # Simulated SCRFD face detection results: [x1, y1, x2, y2, confidence]
        import random
        # Fake a face detection in the middle of the frame
        x1, y1 = random.randint(100, 150), random.randint(100, 150)
        x2, y2 = x1 + random.randint(100, 200), y1 + random.randint(100, 200)
        confidence = round(random.uniform(0.85, 0.99), 4)

        return {
            "model": "SCRFD",
            "faces_detected": 1,
            "bounding_boxes": [[x1, y1, x2, y2, confidence]],
            "simulated_time_ms": random.randint(10, 45)
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
        task_a = loop.run_in_executor(self.executor, self._inference_model_a, tensor_frame)
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
    
    # Maintain a session for repeated HTTP requests to the backend
    async with aiohttp.ClientSession() as http_session:
        async for event in video_stream:
            # `event.frame` contains the raw rtc.VideoFrame (often I420)
            rtc_frame = event.frame
            
            # Convert frame format to ARGB to easily get standard RGB array
            argb_frame = rtc_frame.convert(rtc.VideoBufferType.ARGB)
            frame_data = np.frombuffer(argb_frame.data, dtype=np.uint8)
            
            # Reshape based on height and width, and 4 channels (A, R, G, B)
            # and slice off Alpha to just get RGB
            rgb_data = frame_data.reshape((argb_frame.height, argb_frame.width, 4))[:, :, 1:]
            
            # Run the parallel pipeline over the frame
            inference_results = await pipeline.run_parallel_inference(rgb_data)
            
            # Prepare payload for Laravel
            payload = {
                "track_id": track.sid,
                "timestamp": event.timestamp,
                "width": argb_frame.width,
                "height": argb_frame.height,
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
