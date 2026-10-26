import requests
import numpy as np

OLLAMA_URL = "http://localhost:11434/api/embed"
DEFAULT_EMBED_MODEL = "nomic-embed-text"


def get_embeddings(texts, model=DEFAULT_EMBED_MODEL, timeout=120):
    """Generate embeddings using a locally running Ollama model."""

    if isinstance(texts, str):
        texts = [texts]

    if not texts:
        return np.empty((0, 0), dtype=float)

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "input": texts,
        },
        timeout=timeout,
    )

    response.raise_for_status()

    data = response.json()
    embeddings = data.get("embeddings")

    if not embeddings or len(embeddings) != len(texts):
        raise ValueError("Ollama returned an invalid embeddings response.")

    vectors = np.asarray(embeddings, dtype=float)

    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Embedding vectors are invalid.")

    return vectors


def cosine_similarity_score(vector_a, vector_b):
    """Calculate cosine similarity between two embedding vectors."""

    vector_a = np.asarray(vector_a, dtype=float)
    vector_b = np.asarray(vector_b, dtype=float)

    if vector_a.shape != vector_b.shape:
        raise ValueError("Embedding vectors must have the same dimensions.")

    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)

    if denominator == 0:
        return 0.0

    return float(np.dot(vector_a, vector_b) / denominator)


def compare_texts(text_a, text_b, model=DEFAULT_EMBED_MODEL):
    """Return a semantic similarity score for two text strings."""

    vectors = get_embeddings([text_a, text_b], model=model)

    return cosine_similarity_score(vectors[0], vectors[1])
