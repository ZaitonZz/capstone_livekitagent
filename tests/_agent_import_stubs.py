import sys
import types
from typing import Any


def _build_effnet_stub_module() -> types.ModuleType:
    module = types.ModuleType("deepfakebench_effnet")

    class DeepfakeBenchEfficientNetB4Config:
        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    class DeepfakeBenchEfficientNetB4Adapter:
        def __init__(self, config: DeepfakeBenchEfficientNetB4Config) -> None:
            self.config = config
            self.model_name = "stub-deepfakebench-effnet"
            self.model_version = str(getattr(config, "model_version", "stub"))
            self.fake_threshold = float(getattr(config, "fake_threshold", 0.5))
            self.fake_class_index = int(getattr(config, "fake_class_index", 1))
            self.score_aggregation = str(getattr(config, "score_aggregation", "max"))

        def infer_class_probabilities(
            self,
            frame: Any,
            face_boxes: Any,
            detector_name: str | None = None,
        ) -> list[list[float]]:
            return []

    def aggregate_fake_scores(scores: list[float], mode: str = "max") -> float | None:
        if not scores:
            return None

        if mode == "min":
            return min(scores)

        return max(scores)

    def crop_face_regions(frame: Any, face_boxes: Any, margin_ratio: float = 0.25, min_face_size: int = 48) -> list[Any]:
        return []

    module.DeepfakeBenchEfficientNetB4Config = DeepfakeBenchEfficientNetB4Config
    module.DeepfakeBenchEfficientNetB4Adapter = DeepfakeBenchEfficientNetB4Adapter
    module.aggregate_fake_scores = aggregate_fake_scores
    module.crop_face_regions = crop_face_regions
    return module


def _build_ucf_stub_module() -> types.ModuleType:
    module = types.ModuleType("deepfakebench_ucf")

    class DeepfakeBenchUCFConfig:
        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    class DeepfakeBenchUCFAdapter:
        def __init__(self, config: DeepfakeBenchUCFConfig) -> None:
            self.config = config
            self.model_name = "stub-deepfakebench-ucf"
            self.model_version = str(getattr(config, "model_version", "stub"))
            self.fake_threshold = float(getattr(config, "fake_threshold", 0.5))
            self.fake_class_index = int(getattr(config, "fake_class_index", 1))
            self.score_aggregation = str(getattr(config, "score_aggregation", "max"))

        def infer_class_probabilities(
            self,
            frame: Any,
            face_boxes: Any,
            detector_name: str | None = None,
        ) -> list[list[float]]:
            return []

    module.DeepfakeBenchUCFConfig = DeepfakeBenchUCFConfig
    module.DeepfakeBenchUCFAdapter = DeepfakeBenchUCFAdapter
    return module


def install_agent_dependency_stubs() -> None:
    # Tests that focus on helper logic do not require torch-backed deepfake adapters.
    # Injecting stubs keeps agent imports lightweight in CI and local envs.
    sys.modules["deepfakebench_effnet"] = _build_effnet_stub_module()
    sys.modules["deepfakebench_ucf"] = _build_ucf_stub_module()