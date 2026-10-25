import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MEDLINEPLUS_JSON_PATH = (
    PROJECT_ROOT
    / "data"
    / "knowledge_base"
    / "medlineplus"
    / "medical_knowledge.json"
)


def split_into_chunks(text, chunk_size=180, overlap=40):
    if not text or not text.strip():
        return []

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size.")

    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]

        if chunk_words:
            chunks.append({
                "text": " ".join(chunk_words),
                "start_word": start,
                "end_word": start + len(chunk_words),
                "metadata": {},
            })

        if start + chunk_size >= len(words):
            break

    return chunks


def build_retrieval_index(text, source_name="uploaded_document.pdf"):
    chunks = split_into_chunks(text)

    for chunk in chunks:
        chunk["source"] = source_name

    return _build_index(chunks, source_name)


def build_medlineplus_index(json_path=MEDLINEPLUS_JSON_PATH):
    """Build a TF-IDF index from normalized MedlinePlus topic records."""
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(
            f"Medical knowledge file not found: {json_path}"
        )

    with json_path.open("r", encoding="utf-8") as file:
        topics = json.load(file)

    chunks = []

    for topic in topics:
        title = str(topic.get("title", "")).strip()
        summary = str(topic.get("summary", "")).strip()

        if not title or not summary:
            continue

        text = f"Medical topic: {title}\nSummary: {summary}"

        chunks.append({
            "text": text,
            "source": topic.get("source", "MedlinePlus"),
            "metadata": {
                "title": title,
                "source_url": topic.get("source_url", ""),
                "topic_id": topic.get("id", ""),
                "language": topic.get("language", "English"),
            },
        })

    return _build_index(chunks, "MedlinePlus")


def _build_index(chunks, source_name):
    if not chunks:
        return None

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
    )

    try:
        matrix = vectorizer.fit_transform(
            [chunk["text"] for chunk in chunks]
        )
    except ValueError:
        return None

    return {
        "chunks": chunks,
        "vectorizer": vectorizer,
        "matrix": matrix,
        "source_name": source_name,
    }


def retrieve_relevant_chunks(index, question, top_k=4):
    if (
        not index
        or not isinstance(question, str)
        or not question.strip()
        or top_k < 1
    ):
        return []

    query_vector = index["vectorizer"].transform([question])

    scores = cosine_similarity(
        query_vector,
        index["matrix"],
    ).ravel()

    ranked_indices = scores.argsort()[::-1]
    results = []

    for idx in ranked_indices:
        score = float(scores[idx])

        if score <= 0:
            continue

        chunk = index["chunks"][idx]

        results.append({
            "text": chunk["text"],
            "score": score,
            "source": chunk.get("source", index["source_name"]),
            "chunk_number": int(idx) + 1,
            **chunk.get("metadata", {}),
        })

        if len(results) >= top_k:
            break

    return results
