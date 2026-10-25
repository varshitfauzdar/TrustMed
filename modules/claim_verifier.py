import re


def split_claims(answer: str) -> list[str]:
    """Split an answer into sentence-level units for preliminary screening."""
    if not answer or not answer.strip():
        return []

    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", answer.strip())
        if sentence.strip()
    ]


def verify_claims(answer: str, evidence: list[dict]) -> dict:
    """
    Screen answer sentences against retrieved passages using lexical overlap.

    Important:
    This is a heuristic screening method, not a medical fact-checker.
    Lexical overlap does not establish whether a claim is true.
    """

    passages = [
        str(item.get("text", ""))
        for item in (evidence or [])
        if isinstance(item, dict) and item.get("text")
    ]

    if not passages:
        return {
            "status": "INSUFFICIENT_EVIDENCE",
            "method": "lexical_overlap",
            "claims": [],
            "message": (
                "No evidence passages available. "
                "Manual review is required."
            ),
        }

    evidence_text = " ".join(passages).lower()

    evidence_terms = set(
        re.findall(r"\b[a-zA-Z]{3,}\b", evidence_text)
    )

    results = []

    for sentence in split_claims(answer):
        terms = set(
            re.findall(r"\b[a-zA-Z]{3,}\b", sentence.lower())
        )

        if len(terms) < 4:
            results.append({
                "claim": sentence,
                "overlap": None,
                "status": "NOT_ASSESSED_SHORT_TEXT",
            })
            continue

        overlap = len(terms & evidence_terms) / len(terms)

        results.append({
            "claim": sentence,
            "overlap": round(overlap, 3),
            "status": (
                "POTENTIALLY_SUPPORTED"
                if overlap >= 0.25
                else "NEEDS_VERIFICATION"
            ),
        })

    needs_review = any(
        item["status"] == "NEEDS_VERIFICATION"
        for item in results
    )

    return {
        "status": (
            "REVIEW_REQUIRED"
            if needs_review
            else "SCREENING_COMPLETE"
        ),
        "method": "lexical_overlap",
        "claims": results,
        "message": (
            "Some statements need manual evidence verification."
            if needs_review
            else (
                "No low-overlap statement was flagged. "
                "Correctness is not established."
            )
        ),
    }
