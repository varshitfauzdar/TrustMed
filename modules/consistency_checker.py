import re


NEGATION_TERMS = {
    "no",
    "not",
    "never",
    "neither",
    "without",
    "absent",
    "negative",
    "unlikely",
}

POSITIVE_TERMS = {
    "present",
    "positive",
    "detected",
    "confirmed",
    "abnormal",
    "elevated",
    "increased",
    "high",
    "shows",
    "demonstrates",
}

UNCERTAINTY_TERMS = {
    "possible",
    "possibly",
    "may",
    "might",
    "could",
    "suspected",
    "uncertain",
    "suggests",
}


def _tokenize(text: str) -> list[str]:
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


def _has_negation(tokens: list[str]) -> bool:
    """Check whether a statement contains a negation term."""
    return any(token in NEGATION_TERMS for token in tokens)


def _has_positive_signal(tokens: list[str]) -> bool:
    """Check for words that may indicate a positive or abnormal finding."""
    return any(token in POSITIVE_TERMS for token in tokens)


def check_consistency(
    clinical_text: str,
    image_report: str = "",
) -> dict:
    """
    Perform preliminary text-level consistency screening.

    This function does not analyze medical images.
    It compares the supplied text descriptions using simple
    lexical heuristics and cannot establish clinical consistency.
    """

    if not clinical_text or not clinical_text.strip():
        return {
            "status": "INSUFFICIENT_INPUT",
            "method": "lexical_heuristic",
            "flags": [],
            "message": "Clinical text is missing.",
            "requires_human_review": True,
        }

    if not image_report or not image_report.strip():
        return {
            "status": "INSUFFICIENT_INPUT",
            "method": "lexical_heuristic",
            "flags": [],
            "message": (
                "No image report or image-derived description was provided. "
                "Cross-modal consistency cannot be assessed."
            ),
            "requires_human_review": True,
        }

    clinical_tokens = _tokenize(clinical_text)
    report_tokens = _tokenize(image_report)

    clinical_negation = _has_negation(clinical_tokens)
    report_negation = _has_negation(report_tokens)

    clinical_positive = _has_positive_signal(clinical_tokens)
    report_positive = _has_positive_signal(report_tokens)

    flags = []

    if (
        clinical_negation
        and report_positive
        and not report_negation
    ):
        flags.append({
            "type": "POSSIBLE_TEXTUAL_CONFLICT",
            "message": (
                "The clinical text contains negation while the supplied "
                "image description contains positive or abnormality-related "
                "language. Review the original evidence."
            ),
        })

    if (
        report_negation
        and clinical_positive
        and not clinical_negation
    ):
        flags.append({
            "type": "POSSIBLE_TEXTUAL_CONFLICT",
            "message": (
                "The image description contains negation while the clinical "
                "text contains positive or abnormality-related language. "
                "Review the original evidence."
            ),
        })

    if (
        any(token in UNCERTAINTY_TERMS for token in clinical_tokens)
        or any(token in UNCERTAINTY_TERMS for token in report_tokens)
    ):
        flags.append({
            "type": "UNCERTAINTY_LANGUAGE",
            "message": (
                "One or both descriptions contain uncertainty-related "
                "language. Preserve that uncertainty during interpretation."
            ),
        })

    return {
        "status": (
            "REVIEW_REQUIRED"
            if flags
            else "NO_HEURISTIC_CONFLICT_DETECTED"
        ),
        "method": "lexical_heuristic",
        "flags": flags,
        "message": (
            "Potential textual issues were flagged for human review."
            if flags
            else (
                "No conflict was detected by these simple text rules. "
                "This does not establish medical consistency."
            )
        ),
        "requires_human_review": True,
    }
