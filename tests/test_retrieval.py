import unittest

from modules.retrieval import retrieve_relevant_chunks


class TestRetrieval(unittest.TestCase):

    def test_retrieval_returns_results(self):
        documents = [
            {
                "text": "Heart failure can cause shortness of breath.",
                "source": "heart_failure.txt",
            },
            {
                "text": "Diabetes affects blood glucose regulation.",
                "source": "diabetes.txt",
            },
        ]

        results = retrieve_relevant_chunks(
            "heart failure shortness of breath",
            documents,
        )

        self.assertIsInstance(results, list)

    def test_empty_query(self):
        documents = [
            {
                "text": "Heart failure information.",
                "source": "heart_failure.txt",
            }
        ]

        results = retrieve_relevant_chunks("", documents)

        self.assertIsInstance(results, list)


if __name__ == "__main__":
    unittest.main()
