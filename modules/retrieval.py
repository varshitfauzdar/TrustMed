from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def split_into_chunks(text, chunk_size=180, overlap=40):
    words = text.split()

    if not words:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)

    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]

        if chunk_words:
            chunks.append({
                "text": " ".join(chunk_words),
                "start_word": start,
                "end_word": start + len(chunk_words)
            })

        if start + chunk_size >= len(words):
            break

    return chunks


def build_retrieval_index(text, source_name="uploaded_document.pdf"):
    chunks = split_into_chunks(text)

    if not chunks:
        return None

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
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
        "source_name": source_name
    }


def retrieve_relevant_chunks(index, question, top_k=4):
    if not index or not question.strip():
        return []

    query_vector = index["vectorizer"].transform([question])

    scores = cosine_similarity(
        query_vector,
        index["matrix"]
    ).ravel()

    ranked_indices = scores.argsort()[::-1]
    results = []

    for idx in ranked_indices[:top_k]:
        score = float(scores[idx])

        if score <= 0:
            continue

        chunk = index["chunks"][idx]

        results.append({
            "text": chunk["text"],
            "score": score,
            "source": index["source_name"],
            "chunk_number": int(idx) + 1
        })

    return results
