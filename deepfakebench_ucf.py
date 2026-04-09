from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from deepfakebench_effnet import (
    DeepfakeBenchEfficientNetB4Backbone,
    download_file,
    extract_checkpoint_state_dict,
    normalize_checkpoint_state_dict,
    preprocess_rgb_frame,
)


@dataclass(frozen=True)
class DeepfakeBenchUCFConfig:
    detector_checkpoint_path: str
    detector_checkpoint_url: str = ""
    backbone_weights_path: str | None = None
    backbone_weights_url: str = ""
    model_version: str = "deepfakebench_ucf_effnb4"
    auto_download: bool = False
    input_size: int = 256
    encoder_feat_dim: int = 512
    num_classes: int = 2
    dropout: float = 0.0


class Conv2d1x1(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, out_features: int) -> None:
        super().__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_features, hidden_dim, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_features, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d(x)


class Head(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, out_features: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        pooled = self.pool(x).view(batch_size, -1)
        logits = self.mlp(pooled)
        return self.dropout(logits)


class DeepfakeBenchUCFInferenceEfficientNet(nn.Module):
    def __init__(self, config: DeepfakeBenchUCFConfig) -> None:
        super().__init__()

        half_fingerprint_dim = config.encoder_feat_dim // 2

        self.encoder_f = DeepfakeBenchEfficientNetB4Backbone(
            pretrained_backbone_path=config.backbone_weights_path,
            num_classes=config.num_classes,
            inc=3,
            dropout=config.dropout,
            mode="adjust_channel",
        )
        self.encoder_c = DeepfakeBenchEfficientNetB4Backbone(
            pretrained_backbone_path=config.backbone_weights_path,
            num_classes=config.num_classes,
            inc=3,
            dropout=config.dropout,
            mode="adjust_channel",
        )

        self.block_spe = Conv2d1x1(
            in_features=config.encoder_feat_dim,
            hidden_dim=half_fingerprint_dim,
            out_features=half_fingerprint_dim,
        )
        self.block_sha = Conv2d1x1(
            in_features=config.encoder_feat_dim,
            hidden_dim=half_fingerprint_dim,
            out_features=half_fingerprint_dim,
        )
        self.head_sha = Head(
            in_features=half_fingerprint_dim,
            hidden_dim=config.encoder_feat_dim,
            out_features=config.num_classes,
        )

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        forgery_features = self.encoder_f.features(image_tensor)
        _ = self.encoder_c.features(image_tensor)

        f_spe = self.block_spe(forgery_features)
        f_share = self.block_sha(forgery_features)
        _ = f_spe

        return self.head_sha(f_share)


class DeepfakeBenchUCFAdapter:
    def __init__(
        self,
        config: DeepfakeBenchUCFConfig,
        device: torch.device,
        logger: logging.Logger,
    ) -> None:
        self.config = config
        self.device = device
        self.logger = logger
        self.model: DeepfakeBenchUCFInferenceEfficientNet | None = None

        try:
            self._prepare_weights()
            self.model = self._load_model()
            self.logger.info(
                "DeepfakeBench UCF loaded successfully on %s (input=%s)",
                self.device,
                self.config.input_size,
            )
        except Exception as error:
            self.model = None
            self.logger.error("DeepfakeBench UCF initialization failed: %s", error)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def infer_class_probabilities(self, rgb_frame: np.ndarray) -> list[float] | None:
        if self.model is None:
            return None

        tensor = preprocess_rgb_frame(rgb_frame, self.config.input_size).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)

        probabilities = torch.softmax(logits.flatten(), dim=0)
        return [float(probability.item()) for probability in probabilities]

    def infer_fake_score(self, rgb_frame: np.ndarray, fake_class_index: int = 1) -> float | None:
        class_probabilities = self.infer_class_probabilities(rgb_frame)
        if class_probabilities is None:
            return None

        index = fake_class_index if 0 <= fake_class_index < len(class_probabilities) else len(class_probabilities) - 1
        return class_probabilities[index]

    def _prepare_weights(self) -> None:
        required_files: list[tuple[str, str, str]] = [
            (
                self.config.detector_checkpoint_path,
                self.config.detector_checkpoint_url,
                "DeepfakeBench UCF detector checkpoint",
            )
        ]

        if self.config.backbone_weights_path:
            required_files.insert(
                0,
                (
                    self.config.backbone_weights_path,
                    self.config.backbone_weights_url,
                    "UCF EfficientNet-B4 backbone weights",
                ),
            )

        for file_path, file_url, label in required_files:
            if os.path.exists(file_path):
                continue

            if not self.config.auto_download:
                raise FileNotFoundError(f"{label} not found at {file_path}")

            if file_url.strip() == "":
                raise FileNotFoundError(f"{label} not found at {file_path} and no download URL is configured")

            self.logger.info("Downloading %s from %s", label, file_url)
            download_file(file_url, file_path)
            self.logger.info("Saved %s to %s", label, file_path)

    def _load_model(self) -> DeepfakeBenchUCFInferenceEfficientNet:
        detector = DeepfakeBenchUCFInferenceEfficientNet(self.config)

        checkpoint = torch.load(
            self.config.detector_checkpoint_path,
            map_location=self.device,
        )

        state_dict = extract_checkpoint_state_dict(checkpoint)
        if not state_dict:
            raise RuntimeError("UCF checkpoint did not contain a usable state_dict")

        state_dict = normalize_checkpoint_state_dict(state_dict)
        missing, unexpected = detector.load_state_dict(state_dict, strict=False)

        model_state_keys = set(detector.state_dict().keys())
        matched_keys = model_state_keys.intersection(state_dict.keys())

        if len(matched_keys) < 10:
            prefixed_state_dict = {f"model.{key}": value for key, value in state_dict.items()}
            missing, unexpected = detector.load_state_dict(prefixed_state_dict, strict=False)
            matched_keys = model_state_keys.intersection(prefixed_state_dict.keys())

        if len(matched_keys) < 10:
            raise RuntimeError(
                "UCF checkpoint loading matched too few keys "
                f"({len(matched_keys)}/{len(model_state_keys)})"
            )

        if missing:
            self.logger.warning("UCF checkpoint missing keys: %s", missing)
        if unexpected:
            self.logger.warning("UCF checkpoint unexpected keys: %s", unexpected)

        detector.to(self.device)
        detector.eval()
        return detector
