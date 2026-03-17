from typing import Any

import numpy as np


def normalize_embedding(embedding: Any) -> np.ndarray | None:
    if embedding is None:
        return None

    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.size == 0:
        return None

    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return None

    return vector / norm


def cosine_similarity(left: Any, right: Any) -> float | None:
    left_embedding = normalize_embedding(left)
    right_embedding = normalize_embedding(right)

    if left_embedding is None or right_embedding is None:
        return None

    return float(np.clip(np.dot(left_embedding, right_embedding), -1.0, 1.0))


class FaceGallery:
    def __init__(self, threshold: float, confirmation_streak: int):
        self.threshold = threshold
        self.confirmation_streak = confirmation_streak
        self.consultation_id: int | None = None
        self.patient_id: int | None = None
        self.patient_name: str | None = None
        self.photo_id: int | None = None
        self.reference_embedding: np.ndarray | None = None
        self._last_decision: bool | None = None
        self._decision_streak = 0
        self._reported_decision: bool | None = None

    def build_match_report(self, recognition_result: dict[str, Any]) -> dict[str, Any] | None:
        if self.consultation_id is None:
            return None

        matched = recognition_result.get("matched")
        similarity = recognition_result.get("best_similarity")
        if matched is None or similarity is None:
            return None

        decision = bool(matched)
        if self._last_decision == decision:
            self._decision_streak += 1
        else:
            self._last_decision = decision
            self._decision_streak = 1

        if self._decision_streak < self.confirmation_streak:
            return None

        if self._reported_decision == decision:
            return None

        self._reported_decision = decision

        return {
            "consultation_id": self.consultation_id,
            "matched": decision,
            "face_match_score": round(float(similarity), 4),
            "flagged": not decision,
        }