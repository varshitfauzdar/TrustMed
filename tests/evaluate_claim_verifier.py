import sys
from pathlib import Path

import requests
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent))
from claim_verification_cases import TEST_CASES


OLLAMA_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"


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
    return np.asarray(response.json()["embeddings"], dtype=float)


def cosine_score(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def evaluate():
    print("\n=== TrustMed AI: Retrieval Similarity Comparison ===\n")

    texts = []
    for case in TEST_CASES:
        texts.extend([case["claim"], case["evidence"]])

    embeddings = get_embeddings(texts)

    results = []

    for i, case in enumerate(TEST_CASES):
        claim = case["claim"]
        evidence = case["evidence"]

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )

        tfidf_vectors = vectorizer.fit_transform([claim, evidence])

        tfidf_score = float(
            cosine_similarity(
                tfidf_vectors[0:1],
                tfidf_vectors[1:2],
            )[0][0]
        )

        semantic_score = cosine_score(
            embeddings[2 * i],
            embeddings[2 * i + 1],
        )

        results.append((case["expected"], tfidf_score, semantic_score))

        print(f"Case {i + 1}: {case['expected']}")
        print(f"Claim: {claim}")
        print(f"Evidence: {evidence}")
        print(f"TF-IDF score: {tfidf_score:.4f}")
        print(f"Semantic score: {semantic_score:.4f}")
        print("-" * 60)

    print("\nSummary")
    print(f"Cases evaluated: {len(results)}")
    print(
        "Mean TF-IDF similarity: "
        f"{np.mean([row[1] for row in results]):.4f}"
    )
    print(
        "Mean semantic similarity: "
        f"{np.mean([row[2] for row in results]):.4f}"
    )
    print(
        "\nThese scores measure similarity, not truth, "
        "medical correctness, or entailment."
    )


if __name__ == "__main__":
    evaluate()
