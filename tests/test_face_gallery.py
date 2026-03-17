import unittest

import numpy as np

from face_recognition import FaceGallery, cosine_similarity, normalize_embedding


class FaceGalleryTest(unittest.TestCase):
    def test_normalize_embedding_returns_unit_vector(self) -> None:
        normalized = normalize_embedding([3.0, 4.0])

        self.assertIsNotNone(normalized)
        self.assertAlmostEqual(float(np.linalg.norm(normalized)), 1.0, places=6)

    def test_cosine_similarity_returns_expected_value(self) -> None:
        similarity = cosine_similarity([1.0, 0.0], [1.0, 0.0])

        self.assertEqual(similarity, 1.0)

    def test_cosine_similarity_returns_none_for_invalid_vectors(self) -> None:
        similarity = cosine_similarity([0.0, 0.0], [1.0, 0.0])

        self.assertIsNone(similarity)

    def test_face_gallery_requires_confirmation_streak(self) -> None:
        gallery = FaceGallery(threshold=0.4, confirmation_streak=2)
        gallery.consultation_id = 15

        first_report = gallery.build_match_report({"matched": True, "best_similarity": 0.82})
        second_report = gallery.build_match_report({"matched": True, "best_similarity": 0.84})

        self.assertIsNone(first_report)
        self.assertEqual(
            second_report,
            {
                "consultation_id": 15,
                "matched": True,
                "face_match_score": 0.84,
                "flagged": False,
            },
        )

    def test_face_gallery_only_reports_each_decision_once(self) -> None:
        gallery = FaceGallery(threshold=0.4, confirmation_streak=1)
        gallery.consultation_id = 22

        first_report = gallery.build_match_report({"matched": False, "best_similarity": 0.21})
        second_report = gallery.build_match_report({"matched": False, "best_similarity": 0.18})
        third_report = gallery.build_match_report({"matched": True, "best_similarity": 0.77})

        self.assertIsNotNone(first_report)
        self.assertIsNone(second_report)
        self.assertEqual(third_report["matched"], True)
        self.assertEqual(third_report["flagged"], False)


if __name__ == "__main__":
    unittest.main()