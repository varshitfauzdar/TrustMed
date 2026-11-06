import requests
import json

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:4b"


def assess_evidence(claim, evidence):
    prompt = f"""
You are an experimental evidence relationship classifier.

Compare the CLAIM with the EVIDENCE.

Classify the relationship using exactly one label:

ENTAILED:
The evidence directly supports the claim.

CONTRADICTED:
The evidence directly conflicts with the claim.

NOT_ESTABLISHED:
The evidence does not provide enough information to establish
or contradict the claim.

Do not use outside knowledge.
Do not assume that similar wording means factual support.

CLAIM:
{claim}

EVIDENCE:
{evidence}

Return only valid JSON:
{{
    "label": "ENTAILED, CONTRADICTED, or NOT_ESTABLISHED",
    "explanation": "Brief explanation based only on the evidence."
}}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "Follow the classification instructions exactly."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "think": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "num_predict": 200
            }
        },
        timeout=180,
    )

    response.raise_for_status()

    content = response.json()["message"]["content"]
    result = json.loads(content)

    allowed_labels = {
        "ENTAILED",
        "CONTRADICTED",
        "NOT_ESTABLISHED",
    }

    if result.get("label") not in allowed_labels:
        raise ValueError("Unexpected NLI classification label.")

    return result


if __name__ == "__main__":
    test_cases = [
        {
            "claim": "Diabetes causes high blood glucose.",
            "evidence": "Diabetes is a condition in which blood glucose levels are too high.",
        },
        {
            "claim": "Diabetes is caused by a fictional blue virus.",
            "evidence": "Diabetes involves problems with insulin and blood glucose regulation.",
        },
        {
            "claim": "Diabetes always causes weight loss.",
            "evidence": "Some people with diabetes may experience unexplained weight loss.",
        },
    ]

    for index, case in enumerate(test_cases, 1):
        print(f"\nTEST CASE {index}")

        try:
            result = assess_evidence(
                case["claim"],
                case["evidence"],
            )

            print(json.dumps(result, indent=2))

        except Exception as error:
            print("ERROR:", error)
