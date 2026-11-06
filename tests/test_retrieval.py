import unittest

from modules.retrieval import (
    build_retrieval_index,
    retrieve_relevant_chunks,
)


class TestRetrieval(unittest.TestCase):

    def test_retrieval_returns_results(self):
        text = (
            "Heart failure can cause shortness of breath. "
            "Diabetes affects blood glucose regulation."
        )

        index = build_retrieval_index(
            text,
            source_name="medical_test.txt",
        )

        results = retrieve_relevant_chunks(
            index,
            "heart failure shortness of breath",
        )

        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertIn("Heart failure", results[0]["text"])
        self.assertGreater(results[0]["score"], 0)

    def test_empty_query(self):
        text = "Heart failure information."

        index = build_retrieval_index(text)

        results = retrieve_relevant_chunks(
            index,
            "",
        )

        self.assertEqual(results, [])

    def test_invalid_index(self):
        results = retrieve_relevant_chunks(
            None,
            "heart failure",
        )

        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()