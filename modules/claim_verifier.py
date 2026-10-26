import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from modules.semantic_matcher import (
    get_embeddings,
    cosine_similarity_score,
)


def split_claims(answer: str) -> list[str]:
    """Split an answer into sentence-level claims."""
    if not isinstance(answer, str) or not answer.strip():
        return []

    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", answer.strip())
        if sentence.strip()
    ]


def lexical_similarity(claim: str, evidence: str) -> float:
    """Calculate TF-IDF cosine similarity between two texts."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
    )

    try:
        vectors = vectorizer.fit_transform([claim, evidence])
        return float(
            cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        )
    except ValueError:
        return 0.0


def verify_claims(answer: str, evidence: list[dict]) -> dict:
    """
    Compare each answer sentence against individual evidence passages.

    Similarity scores measure textual similarity, not medical correctness.
    """

    claims = split_claims(answer)

    if not claims:
        return {
            "status": "NO_CLAIMS",
            "method": "lexical_and_semantic_similarity",
            "claims": [],
            "message": "No answer sentences were available to screen.",
        }

    passages = [
        item for item in (evidence or [])
        if isinstance(item, dict)
        and isinstance(item.get("text"), str)
        and item["text"].strip()
    ]

    if not passages:
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "method": "lexical_and_semantic_similarity",
            "claims": [
                {
                    "claim": claim,
                    "lexical_score": None,
                    "semantic_score": None,
                    "status": "INSUFFICIENT_EVIDENCE",
                    "matching_sources": [],
                }
                for claim in claims
            ],
            "message": "No evidence passages are available.",
            "limitations": (
                "No evidence was available for comparison. "
                "Medical correctness has not been established."
            ),
        }

    results = []

    # Generate claim embeddings once, then evidence embeddings once.
    all_texts = claims + [passage["text"] for passage in passages]

    try:
        embeddings = get_embeddings(all_texts)
        semantic_available = True
        semantic_error = None
    except Exception as exc:
        embeddings = None
        semantic_available = False
        semantic_error = str(exc)

    claim_count = len(claims)

    for claim_index, claim in enumerate(claims):
        best_lexical_score = 0.0
        best_semantic_score = None
        matching_sources = []
        best_passage = ""

        claim_vector = (
            embeddings[claim_index]
            if semantic_available
            else None
        )

        for passage_index, passage in enumerate(passages):
            evidence_text = passage["text"]

            lexical_score = lexical_similarity(
                claim,
                evidence_text,
            )

            semantic_score = None

            if semantic_available:
                evidence_vector = embeddings[
                    claim_count + passage_index
                ]

                semantic_score = cosine_similarity_score(
                    claim_vector,
                    evidence_vector,
                )

            # Rank passages by semantic score when available.
            # Otherwise, fall back to lexical score.
            if semantic_available:
                is_better = (
                    best_semantic_score is None
                    or semantic_score > best_semantic_score
                )
            else:
                is_better = lexical_score > best_lexical_score

            if is_better:
                best_lexical_score = lexical_score
                best_semantic_score = semantic_score
                best_passage = evidence_text
                matching_sources = [
                    passage.get(
                        "source",
                        passage.get("source_url", "Unknown"),
                    )
                ]

            elif (
                semantic_available
                and semantic_score == best_semantic_score
            ):
                source = passage.get(
                    "source",
                    passage.get("source_url", "Unknown"),
                )

                if source not in matching_sources:
                    matching_sources.append(source)

        if semantic_available:
            # These are provisional screening thresholds, not validated
            # medical decision thresholds.
            if best_semantic_score >= 0.80:
                status = "POTENTIAL_MATCH_REVIEW_REQUIRED"
            elif best_semantic_score >= 0.65:
                status = "POSSIBLE_MATCH_REVIEW_REQUIRED"
            else:
                status = "LOW_SIMILARITY_REVIEW_REQUIRED"
        else:
            status = (
                "POTENTIAL_LEXICAL_MATCH_REVIEW_REQUIRED"
                if best_lexical_score >= 0.25
                else "LOW_LEXICAL_OVERLAP_REVIEW_REQUIRED"
            )

        results.append({
            "claim": claim,
            "lexical_score": round(best_lexical_score, 4),
            "semantic_score": (
                round(best_semantic_score, 4)
                if best_semantic_score is not None
                else None
            ),
            "status": status,
            "matching_sources": matching_sources,
            "best_evidence_excerpt": best_passage[:500],
        })

    return {
        "status": "REVIEW_REQUIRED",
        "method": (
            "lexical_and_semantic_similarity"
            if semantic_available
            else "lexical_similarity_fallback"
        ),
        "claims": results,
        "message": (
            "Similarity screening completed. All claims still require "
            "appropriate evidence review; similarity does not prove truth."
        ),
        "semantic_available": semantic_available,
        "semantic_error": semantic_error,
        "limitations": (
            "Lexical and semantic similarity cannot reliably establish "
            "medical correctness, entailment, or contradiction. The "
            "screening thresholds are provisional and have not been "
            "clinically validated."
        ),
    }
