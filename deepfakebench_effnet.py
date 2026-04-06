from __future__ import annotations

import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DeepfakeBenchEfficientNetB4Config:
    detector_checkpoint_path: str
    detector_checkpoint_url: str
    backbone_weights_path: str
    backbone_weights_url: str
    model_version: str = "deepfakebench_effnb4"
    auto_download: bool = True
    input_size: int = 256


class DeepfakeBenchEfficientNetB4Backbone(nn.Module):
    def __init__(
        self,
        pretrained_backbone_path: str | None,
        num_classes: int = 2,
        inc: int = 3,
        dropout: float | bool = False,
        mode: str = "Original",
    ) -> None:
        super().__init__()

        try:
            from efficientnet_pytorch import EfficientNet
        except ImportError as error:
            raise RuntimeError(
                "efficientnet_pytorch is required for DeepfakeBench EfficientNet-B4 backend"
            ) from error

        if pretrained_backbone_path and os.path.exists(pretrained_backbone_path):
            self.efficientnet = EfficientNet.from_pretrained(
                "efficientnet-b4",
                weights_path=pretrained_backbone_path,
            )
        else:
            self.efficientnet = EfficientNet.from_name("efficientnet-b4")

        # DeepfakeBench replaces the stem and classifier in its backbone definition.
        self.efficientnet._conv_stem = nn.Conv2d(inc, 48, kernel_size=3, stride=2, bias=False)
        self.efficientnet._fc = nn.Identity()

        self.dropout = float(dropout) if dropout else 0.0
        self.dropout_layer = nn.Dropout(p=self.dropout) if self.dropout > 0.0 else None
        self.last_layer = nn.Linear(1792, num_classes)
        self.mode = mode

        if self.mode == "adjust_channel":
            self.adjust_channel = nn.Sequential(
                nn.Conv2d(1792, 512, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )
        else:
            self.adjust_channel = None

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.efficientnet.extract_features(x)
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        return x

    def classifier(self, x: torch.Tensor) -> torch.Tensor:
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        if self.dropout_layer is not None:
            x = self.dropout_layer(x)
        return self.last_layer(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class DeepfakeBenchEfficientNetB4Detector(nn.Module):
    def __init__(self, pretrained_backbone_path: str | None) -> None:
        super().__init__()
        self.backbone = DeepfakeBenchEfficientNetB4Backbone(
            pretrained_backbone_path=pretrained_backbone_path,
            num_classes=2,
            inc=3,
            dropout=False,
            mode="Original",
        )

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        features = self.backbone.features(image_tensor)
        logits = self.backbone.classifier(features)
        return logits


def extract_checkpoint_state_dict(checkpoint: Any) -> dict[str, Any]:
    def looks_like_state_dict(value: Any) -> bool:
        if not isinstance(value, dict) or not value:
            return False

        sample_keys = list(value.keys())[:20]
        if not all(isinstance(sample_key, str) for sample_key in sample_keys):
            return False

        sample_values = list(value.values())[:20]
        return any(hasattr(sample_value, "shape") for sample_value in sample_values)

    pending: list[Any] = [checkpoint]
    visited: set[int] = set()

    while pending:
        current = pending.pop(0)
        current_id = id(current)
        if current_id in visited:
            continue
        visited.add(current_id)

        if looks_like_state_dict(current):
            return current

        if isinstance(current, dict):
            for key in ("state_dict", "model_state_dict", "model", "net", "network", "weights"):
                if key in current:
                    pending.append(current[key])

    return {}


def normalize_checkpoint_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    prefixes = ("module.", "_orig_mod.", "model.", "net.", "network.")

    for raw_key, value in state_dict.items():
        key = raw_key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        normalized[key] = value

    return normalized


def preprocess_rgb_frame(
    rgb_frame: np.ndarray,
    input_size: int,
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> torch.Tensor:
    import cv2

    resized_rgb = cv2.resize(rgb_frame, (input_size, input_size), interpolation=cv2.INTER_AREA)
    normalized_rgb = resized_rgb.astype(np.float32) / 255.0

    tensor = torch.from_numpy(normalized_rgb).permute(2, 0, 1).unsqueeze(0)
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype).view(1, 3, 1, 1)

    return (tensor - mean_tensor) / std_tensor


def crop_face_regions(
    rgb_frame: np.ndarray,
    face_boxes: list[list[float]] | None,
    margin_ratio: float = 0.25,
    min_face_size: int = 64,
) -> list[np.ndarray]:
    if rgb_frame.ndim != 3:
        return []

    if not face_boxes:
        return []

    frame_height, frame_width = rgb_frame.shape[:2]
    crops: list[np.ndarray] = []

    for raw_box in face_boxes:
        if len(raw_box) < 4:
            continue

        x1, y1, x2, y2 = [int(float(value)) for value in raw_box[:4]]
        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(0, min(x2, frame_width))
        y2 = max(0, min(y2, frame_height))

        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            continue

        if min(width, height) < min_face_size:
            continue

        pad_x = int(width * margin_ratio)
        pad_y = int(height * margin_ratio)

        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(frame_width, x2 + pad_x)
        crop_y2 = min(frame_height, y2 + pad_y)

        crop = rgb_frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            continue

        crops.append(crop)

    return crops


def aggregate_fake_scores(scores: list[float], mode: str = "max") -> float:
    if not scores:
        raise ValueError("No scores to aggregate")

    normalized_mode = mode.strip().lower()
    if normalized_mode == "mean":
        return float(np.mean(scores))
    if normalized_mode == "median":
        return float(np.median(scores))
    if normalized_mode == "min":
        return float(np.min(scores))

    return float(np.max(scores))


def logits_to_fake_probability(logits: torch.Tensor, fake_class_index: int = 1) -> float:
    flattened_logits = logits.flatten()

    if flattened_logits.numel() == 1:
        return float(torch.sigmoid(flattened_logits[0]).item())

    probabilities = torch.softmax(flattened_logits, dim=0)
    index = fake_class_index if 0 <= fake_class_index < probabilities.numel() else probabilities.numel() - 1
    return float(probabilities[index].item())


def download_file(url: str, destination_path: str, timeout_seconds: int = 90) -> None:
    destination_dir = os.path.dirname(destination_path)
    if destination_dir:
        os.makedirs(destination_dir, exist_ok=True)
    temporary_path = f"{destination_path}.tmp"

    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            with open(temporary_path, "wb") as file_handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    file_handle.write(chunk)
    except urllib.error.URLError as error:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise RuntimeError(f"Failed to download {url}: {error}") from error

    os.replace(temporary_path, destination_path)


class DeepfakeBenchEfficientNetB4Adapter:
    def __init__(
        self,
        config: DeepfakeBenchEfficientNetB4Config,
        device: torch.device,
        logger: logging.Logger,
    ) -> None:
        self.config = config
        self.device = device
        self.logger = logger
        self.model: DeepfakeBenchEfficientNetB4Detector | None = None

        try:
            self._prepare_weights()
            self.model = self._load_model()
            self.logger.info(
                "DeepfakeBench EfficientNet-B4 loaded successfully on %s (input=%s)",
                self.device,
                self.config.input_size,
            )
        except Exception as error:
            self.model = None
            self.logger.error("DeepfakeBench EfficientNet-B4 initialization failed: %s", error)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def infer_fake_score(self, rgb_frame: np.ndarray, fake_class_index: int = 1) -> float | None:
        if self.model is None:
            return None

        tensor = preprocess_rgb_frame(rgb_frame, self.config.input_size)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)

        return logits_to_fake_probability(logits, fake_class_index=fake_class_index)

    def _prepare_weights(self) -> None:
        required_files = (
            (
                self.config.backbone_weights_path,
                self.config.backbone_weights_url,
                "EfficientNet-B4 backbone weights",
            ),
            (
                self.config.detector_checkpoint_path,
                self.config.detector_checkpoint_url,
                "DeepfakeBench EfficientNet-B4 detector checkpoint",
            ),
        )

        for file_path, file_url, label in required_files:
            if os.path.exists(file_path):
                continue

            if not self.config.auto_download:
                raise FileNotFoundError(f"{label} not found at {file_path}")

            self.logger.info("Downloading %s from %s", label, file_url)
            download_file(file_url, file_path)
            self.logger.info("Saved %s to %s", label, file_path)

    def _load_model(self) -> DeepfakeBenchEfficientNetB4Detector:
        detector = DeepfakeBenchEfficientNetB4Detector(
            pretrained_backbone_path=self.config.backbone_weights_path,
        )

        checkpoint = torch.load(
            self.config.detector_checkpoint_path,
            map_location=self.device,
        )

        state_dict = extract_checkpoint_state_dict(checkpoint)
        if not state_dict:
            raise RuntimeError("EfficientNet-B4 checkpoint did not contain a usable state_dict")

        state_dict = normalize_checkpoint_state_dict(state_dict)
        missing, unexpected = detector.load_state_dict(state_dict, strict=False)

        model_state_keys = set(detector.state_dict().keys())
        matched_keys = model_state_keys.intersection(state_dict.keys())

        if len(matched_keys) < 10:
            prefixed_state_dict = {f"backbone.{key}": value for key, value in state_dict.items()}
            missing, unexpected = detector.load_state_dict(prefixed_state_dict, strict=False)
            matched_keys = model_state_keys.intersection(prefixed_state_dict.keys())

        if len(matched_keys) < 10:
            raise RuntimeError(
                "EfficientNet-B4 checkpoint loading matched too few keys "
                f"({len(matched_keys)}/{len(model_state_keys)})"
            )

        if missing:
            self.logger.warning("EfficientNet-B4 checkpoint missing keys: %s", missing)
        if unexpected:
            self.logger.warning("EfficientNet-B4 checkpoint unexpected keys: %s", unexpected)

        detector.to(self.device)
        detector.eval()
        return detector
