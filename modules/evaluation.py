from collections import Counter
from typing import Any


def evaluate_retrieval(
    queries: list[dict[str, Any]],
    k: int = 3,
) -> dict[str, Any]:
    """
    Evaluate retrieval using manually labelled relevant document IDs.

    Expected input:
    [
        {
            "relevant_ids": ["doc1", "doc2"],
            "retrieved_ids": ["doc1", "doc3", "doc2"]
        }
    ]

    Each ID must identify a document or evidence chunk consistently.
    """

    if k < 1:
        raise ValueError("k must be at least 1.")

    if not queries:
        return {
            "status": "INSUFFICIENT_DATA",
            "query_count": 0,
            "precision_at_k": None,
            "recall_at_k": None,
            "mean_reciprocal_rank": None,
            "message": "No evaluation queries were provided.",
        }

    precision_scores = []
    recall_scores = []
    reciprocal_ranks = []
    skipped_queries = 0

    for query in queries:
        relevant_ids = set(query.get("relevant_ids", []))
        retrieved_ids = list(query.get("retrieved_ids", []))[:k]

        if not relevant_ids:
            skipped_queries += 1
            continue

        retrieved_unique = list(dict.fromkeys(retrieved_ids))
        relevant_retrieved = set(retrieved_unique) & relevant_ids

        precision = len(relevant_retrieved) / k
        recall = len(relevant_retrieved) / len(relevant_ids)

        precision_scores.append(precision)
        recall_scores.append(recall)

        reciprocal_rank = 0.0

        for rank, document_id in enumerate(retrieved_unique, start=1):
            if document_id in relevant_ids:
                reciprocal_rank = 1.0 / rank
                break

        reciprocal_ranks.append(reciprocal_rank)

    if not precision_scores:
        return {
            "status": "INSUFFICIENT_DATA",
            "query_count": len(queries),
            "evaluated_queries": 0,
            "skipped_queries": skipped_queries,
            "precision_at_k": None,
            "recall_at_k": None,
            "mean_reciprocal_rank": None,
            "message": (
                "No queries had labelled relevant IDs. "
                "Provide ground-truth relevance labels."
            ),
        }

    return {
        "status": "EVALUATED",
        "query_count": len(queries),
        "evaluated_queries": len(precision_scores),
        "skipped_queries": skipped_queries,
        "k": k,
        "precision_at_k": round(
            sum(precision_scores) / len(precision_scores), 4
        ),
        "recall_at_k": round(
            sum(recall_scores) / len(recall_scores), 4
        ),
        "mean_reciprocal_rank": round(
            sum(reciprocal_ranks) / len(reciprocal_ranks), 4
        ),
        "message": (
            "Metrics measure retrieval against the supplied relevance labels; "
            "they do not measure medical correctness or clinical safety."
        ),
    }


def summarize_screening_results(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Summarize claim-screening outcomes.

    This reports observed labels only; it does not calculate
    factual accuracy without independently verified ground truth.
    """

    if not results:
        return {
            "status": "INSUFFICIENT_DATA",
            "total_claims": 0,
            "status_counts": {},
            "message": "No screening results were provided.",
        }

    status_counts = Counter(
        str(item.get("status", "UNKNOWN"))
        for item in results
    )

    return {
        "status": "SUMMARIZED",
        "total_claims": len(results),
        "status_counts": dict(status_counts),
        "message": (
            "These counts describe heuristic screening outputs, "
            "not verified factual accuracy."
        ),
    }
