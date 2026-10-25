def assess_uncertainty(
    answer: str,
    evidence: list[dict] | None = None,
    min_evidence: int = 1,
) -> dict:
    """
    Assess evidence availability and uncertainty indicators.

    This is a preliminary rule-based assessment, not a calibrated
    probability of correctness or a clinical confidence score.
    """

    answer = (answer or "").strip()
    evidence = evidence or []

    usable_evidence = [
        item
        for item in evidence
        if isinstance(item, dict)
        and str(item.get("text", "")).strip()
    ]

    reasons = []

    if not answer:
        reasons.append("The generated answer is empty.")

    if len(usable_evidence) < min_evidence:
        reasons.append(
            "Insufficient retrieved evidence for the configured threshold."
        )

    uncertainty_phrases = [
        "uncertain",
        "unclear",
        "not enough information",
        "insufficient evidence",
        "cannot determine",
        "could not determine",
        "may indicate",
        "might indicate",
        "requires further evaluation",
    ]

    answer_lower = answer.lower()

    detected_phrases = [
        phrase
        for phrase in uncertainty_phrases
        if phrase in answer_lower
    ]

    if detected_phrases:
        reasons.append(
            "The answer contains uncertainty-related language."
        )

    if not answer or len(usable_evidence) < min_evidence:
        status = "ABSTAIN_RECOMMENDED"
        requires_review = True
    elif detected_phrases:
        status = "UNCERTAINTY_DETECTED"
        requires_review = True
    else:
        status = "EVIDENCE_AVAILABLE"
        requires_review = True

    return {
        "status": status,
        "evidence_count": len(usable_evidence),
        "uncertainty_indicators": detected_phrases,
        "reasons": reasons,
        "requires_human_review": requires_review,
        "confidence_score": None,
        "message": (
            "This assessment does not calculate clinical confidence. "
            "A qualified healthcare professional must review the result."
        ),
    }
