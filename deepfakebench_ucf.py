from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    backbone_name: str = "xception"
    backbone_weights_path: str | None = None
    backbone_weights_url: str = ""
    model_version: str = "deepfakebench_ucf"
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


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        return self.pointwise(x)


class XceptionBlock(nn.Module):
    def __init__(
        self,
        in_filters: int,
        out_filters: int,
        reps: int,
        strides: int = 1,
        start_with_relu: bool = True,
        grow_first: bool = True,
    ) -> None:
        super().__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep: list[nn.Module] = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for _ in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))

        self.rep = nn.Sequential(*rep)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        return x + skip


class DeepfakeBenchXceptionBackbone(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        inc: int = 3,
        dropout: float | bool = False,
        mode: str = "adjust_channel",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.mode = mode

        self.conv1 = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = XceptionBlock(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = XceptionBlock(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = XceptionBlock(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = XceptionBlock(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block12 = XceptionBlock(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        final_channel = 2048
        if self.mode == "adjust_channel_iid":
            final_channel = 512
            self.mode = "adjust_channel"

        dropout_probability = float(dropout) if dropout else 0.0
        self.last_linear: nn.Module
        if dropout_probability > 0.0:
            self.last_linear = nn.Sequential(
                nn.Dropout(p=dropout_probability),
                nn.Linear(final_channel, self.num_classes),
            )
        else:
            self.last_linear = nn.Linear(final_channel, self.num_classes)

        self.adjust_channel = nn.Sequential(
            nn.Conv2d(2048, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
        )

    def fea_part1(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

    def fea_part2(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def fea_part3(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        return x

    def fea_part4(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        return x

    def fea_part5(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def features(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = self.fea_part1(input_tensor)
        x = self.fea_part2(x)
        x = self.fea_part3(x)
        x = self.fea_part4(x)
        x = self.fea_part5(x)

        if self.mode == "adjust_channel":
            x = self.adjust_channel(x)
        return x

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        x = features if self.mode == "adjust_channel" else self.relu(features)
        if len(x.shape) == 4:
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
        return self.last_linear(x)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(input_tensor))


class DeepfakeBenchUCFInferenceModel(nn.Module):
    def __init__(self, config: DeepfakeBenchUCFConfig) -> None:
        super().__init__()

        half_fingerprint_dim = config.encoder_feat_dim // 2

        self.encoder_f = self._build_backbone(config)
        self.encoder_c = self._build_backbone(config)

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

    @staticmethod
    def _build_backbone(config: DeepfakeBenchUCFConfig) -> nn.Module:
        if config.backbone_name == "xception":
            return DeepfakeBenchXceptionBackbone(
                num_classes=config.num_classes,
                inc=3,
                dropout=config.dropout,
                mode="adjust_channel",
            )

        if config.backbone_name == "efficientnet-b4":
            return DeepfakeBenchEfficientNetB4Backbone(
                pretrained_backbone_path=config.backbone_weights_path,
                num_classes=config.num_classes,
                inc=3,
                dropout=config.dropout,
                mode="adjust_channel",
            )

        raise ValueError(
            f"Unsupported UCF backbone '{config.backbone_name}'. Supported backbones: ['xception', 'efficientnet-b4']"
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
        self.model: DeepfakeBenchUCFInferenceModel | None = None

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

        if self.config.backbone_name == "efficientnet-b4" and self.config.backbone_weights_path:
            required_files.insert(
                0,
                (
                    self.config.backbone_weights_path,
                    self.config.backbone_weights_url,
                    "UCF efficientnet-b4 backbone weights",
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

    @staticmethod
    def _load_state_dict_with_shape_filter(
        model: nn.Module,
        state_dict: dict[str, Any],
    ) -> tuple[list[str], list[str], set[str], list[str]]:
        model_state = model.state_dict()
        filtered_state: dict[str, Any] = {}
        skipped_mismatched: list[str] = []

        for key, value in state_dict.items():
            target_value = model_state.get(key)
            if target_value is None:
                continue

            if hasattr(value, "shape") and tuple(value.shape) != tuple(target_value.shape):
                skipped_mismatched.append(key)
                continue

            filtered_state[key] = value

        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        return list(missing), list(unexpected), set(filtered_state.keys()), skipped_mismatched

    def _load_model(self) -> DeepfakeBenchUCFInferenceModel:
        detector = DeepfakeBenchUCFInferenceModel(self.config)

        checkpoint = torch.load(
            self.config.detector_checkpoint_path,
            map_location=self.device,
        )

        state_dict = extract_checkpoint_state_dict(checkpoint)
        if not state_dict:
            raise RuntimeError("UCF checkpoint did not contain a usable state_dict")

        state_dict = normalize_checkpoint_state_dict(state_dict)
        missing, unexpected, matched_keys, skipped = self._load_state_dict_with_shape_filter(detector, state_dict)

        model_state_keys = set(detector.state_dict().keys())

        if len(matched_keys) < 10:
            prefixed_state_dict = {f"model.{key}": value for key, value in state_dict.items()}
            missing, unexpected, matched_keys, skipped_prefixed = self._load_state_dict_with_shape_filter(
                detector,
                prefixed_state_dict,
            )
            skipped.extend(skipped_prefixed)

        if len(matched_keys) < 10:
            raise RuntimeError(
                "UCF checkpoint loading matched too few keys "
                f"({len(matched_keys)}/{len(model_state_keys)}) for backbone '{self.config.backbone_name}'"
            )

        if skipped:
            self.logger.warning(
                "UCF checkpoint skipped %s mismatched keys for backbone %s (examples: %s)",
                len(skipped),
                self.config.backbone_name,
                ", ".join(skipped[:3]),
            )
        if missing:
            self.logger.warning("UCF checkpoint missing keys (%s): %s", len(missing), missing[:20])
        if unexpected:
            self.logger.warning("UCF checkpoint unexpected keys (%s): %s", len(unexpected), unexpected[:20])

        detector.to(self.device)
        detector.eval()
        return detector
