import sys
from pathlib import Path

import requests
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claim_verification_cases import TEST_CASES


OLLAMA_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"

LABELS = ["RELEVANT", "PARTIAL", "UNRELATED"]


def get_embeddings(texts):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": EMBED_MODEL,
            "input": texts,
        },
        timeout=120,
    )
    response.raise_for_status()

    embeddings = response.json()["embeddings"]

    if len(embeddings) != len(texts):
        raise ValueError("Invalid embedding response.")

    return np.asarray(embeddings, dtype=float)


def cosine_score(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def predict_category(score, method):
    """
    Experimental thresholds for this small prototype dataset.
    These are not clinically validated.
    """

    if method == "semantic":
        if score >= 0.80:
            return "RELEVANT"
        elif score >= 0.65:
            return "PARTIAL"
        return "UNRELATED"

    if score >= 0.25:
        return "RELEVANT"
    elif score >= 0.10:
        return "PARTIAL"

    return "UNRELATED"


def evaluate():
    print("\n========== TRUSTMED AI EVALUATION ==========\n")

    texts = []

    for case in TEST_CASES:
        texts.extend([case["claim"], case["evidence"]])

    embeddings = get_embeddings(texts)

    actual = []
    lexical_predictions = []
    semantic_predictions = []

    for i, case in enumerate(TEST_CASES):
        claim = case["claim"]
        evidence = case["evidence"]
        expected = case["expected"]

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )

        vectors = vectorizer.fit_transform([claim, evidence])

        lexical_score = float(
            (vectors[0] @ vectors[1].T).toarray()[0][0]
            / (
                np.sqrt(vectors[0].multiply(vectors[0]).sum())
                * np.sqrt(vectors[1].multiply(vectors[1]).sum())
            )
        ) if vectors.nnz else 0.0

        semantic_score = cosine_score(
            embeddings[2 * i],
            embeddings[2 * i + 1],
        )

        lexical_prediction = predict_category(
            lexical_score, "lexical"
        )

        semantic_prediction = predict_category(
            semantic_score, "semantic"
        )

        actual.append(expected)
        lexical_predictions.append(lexical_prediction)
        semantic_predictions.append(semantic_prediction)

        print(f"Case {i + 1}")
        print(f"Claim: {claim}")
        print(f"Expected: {expected}")
        print(f"Lexical: {lexical_score:.4f} -> {lexical_prediction}")
        print(f"Semantic: {semantic_score:.4f} -> {semantic_prediction}")
        print("-" * 55)

    for method, predictions in [
        ("LEXICAL", lexical_predictions),
        ("SEMANTIC", semantic_predictions),
    ]:
        print(f"\n========== {method} EVALUATION ==========")

        print(
            f"Accuracy: {accuracy_score(actual, predictions):.4f}"
        )

        print("\nClassification Report:")
        print(
            classification_report(
                actual,
                predictions,
                labels=LABELS,
                zero_division=0,
            )
        )

        print("Confusion Matrix:")
        print(confusion_matrix(actual, predictions, labels=LABELS))

        print("\nMisclassified cases:")

        found = False

        for index, (expected, predicted) in enumerate(
            zip(actual, predictions)
        ):
            if expected != predicted:
                found = True
                print(
                    f"Case {index + 1}: "
                    f"Expected={expected}, Predicted={predicted}"
                )

        if not found:
            print("None")

    print("\n===========================================")
    print("IMPORTANT:")
    print("Only four illustrative test cases were evaluated.")
    print("Thresholds are experimental, not clinically validated.")
    print("Similarity does not establish truth or entailment.")
    print("===========================================")


if __name__ == "__main__":
    evaluate()
