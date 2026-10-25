import unittest

from modules.evaluation import (
    evaluate_retrieval,
    summarize_screening_results,
)


class TestEvaluation(unittest.TestCase):

    def test_retrieval_metrics(self):
        queries = [
            {
                "relevant_ids": ["doc1", "doc2"],
                "retrieved_ids": ["doc1", "doc3", "doc2"],
            }
        ]

        result = evaluate_retrieval(queries, k=3)

        self.assertEqual(result["status"], "EVALUATED")
        self.assertEqual(result["precision_at_k"], 0.6667)
        self.assertEqual(result["recall_at_k"], 1.0)
        self.assertEqual(result["mean_reciprocal_rank"], 1.0)

    def test_empty_retrieval_data(self):
        result = evaluate_retrieval([])

        self.assertEqual(result["status"], "INSUFFICIENT_DATA")
        self.assertIsNone(result["precision_at_k"])

    def test_screening_summary(self):
        results = [
            {"status": "POTENTIALLY_SUPPORTED"},
            {"status": "NEEDS_VERIFICATION"},
            {"status": "POTENTIALLY_SUPPORTED"},
        ]

        summary = summarize_screening_results(results)

        self.assertEqual(summary["total_claims"], 3)
        self.assertEqual(
            summary["status_counts"]["POTENTIALLY_SUPPORTED"], 2
        )
        self.assertEqual(
            summary["status_counts"]["NEEDS_VERIFICATION"], 1
        )

    def test_empty_screening_results(self):
        result = summarize_screening_results([])

        self.assertEqual(result["status"], "INSUFFICIENT_DATA")
        self.assertEqual(result["total_claims"], 0)


if __name__ == "__main__":
    unittest.main()
