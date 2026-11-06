import re

PATTERNS = {
    "negation": [
        r"\bnot\b", r"\bnever\b", r"\bno\b",
        r"\bcannot\b", r"\bdoesn't\b", r"\bisn't\b",
        r"\bwithout\b", r"\bneither\b", r"\bnor\b",
    ],

    "opposition": [
        r"\bprevent(?:s|ed|ing)?\b",
        r"\bblock(?:s|ed|ing)?\b",
        r"\bstop(?:s|ped|ping)?\b",
        r"\binhibit(?:s|ed|ing)?\b",
        r"\bprohibit(?:s|ed|ing)?\b",
    ],

    "universal": [
        r"\balways\b", r"\bevery\b", r"\ball\b",
        r"\bnever\b", r"\bnone\b", r"\beveryone\b",
        r"\beverything\b",
    ],

    "numerical": [
        r"\bexactly\b", r"\bat least\b", r"\bat most\b",
        r"\b\d+(?:\.\d+)?\s*(?:kg|mg|g|times|days|hours|years|%)\b",
        r"\b\d+\b",
        r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b",
    ],

    "possibility": [
        r"\bmay\b", r"\bmight\b", r"\bcould\b",
        r"\bcan\b", r"\bpossibly\b",
    ],

    "certainty": [
        r"\bmust\b", r"\bdefinitely\b",
        r"\bguaranteed\b", r"\bcertainly\b",
    ],
}


def analyze_claim(claim):
    """Detect linguistic features requiring careful claim review."""

    if not isinstance(claim, str) or not claim.strip():
        return {
            "claim": claim,
            "features": {},
            "requires_careful_review": True,
        }

    normalized = claim.lower()
    features = {}

    for feature, patterns in PATTERNS.items():
        matches = []

        for pattern in patterns:
            matches.extend(
                match.group(0)
                for match in re.finditer(pattern, normalized)
            )

        features[feature] = sorted(set(matches))

    requires_review = any(features.values())

    return {
        "claim": claim,
        "features": features,
        "requires_careful_review": requires_review,
    }
