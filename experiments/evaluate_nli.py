import json
import time
from pathlib import Path

import requests
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from nli_evaluation_cases import TEST_CASES

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:4b"

LABELS = ["ENTAILED", "CONTRADICTED", "NOT_ESTABLISHED"]


def assess_evidence(claim, evidence):
    prompt = f"""
Compare the CLAIM with the EVIDENCE.

Choose exactly one relationship:

ENTAILED: Evidence directly supports the claim.
CONTRADICTED: Evidence directly conflicts with the claim.
NOT_ESTABLISHED: Evidence does not establish or directly contradict the claim.

Use only the supplied evidence.
Do not use outside knowledge.

CLAIM:
{claim}

EVIDENCE:
{evidence}

Return JSON with exactly two fields:
label and explanation.
Keep explanation under 35 words.
"""

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Classify evidence relationships. Return concise valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "think": False,
        "format": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "enum": LABELS
                },
                "explanation": {
                    "type": "string"
                }
            },
            "required": ["label", "explanation"]
        },
        "options": {
            "temperature": 0,
            "num_predict": 120,
        },
    }

    last_error = None

    for attempt in range(2):
        try:
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=180,
            )

            response.raise_for_status()

            content = response.json()["message"]["content"]
            result = json.loads(content)

            if result.get("label") not in LABELS:
                raise ValueError("Invalid classification label.")

            if not isinstance(result.get("explanation"), str):
                raise ValueError("Invalid explanation format.")

            return result

        except (
            requests.RequestException,
            ValueError,
            KeyError,
            TypeError,
        ) as exc:
            last_error = exc

            if attempt == 0:
                print("Retrying NLI request...", flush=True)

    raise RuntimeError(
        f"NLI classification failed after retry: {last_error}"
    )


def evaluate():
    print("\n========== TRUSTMED AI: NLI EVALUATION ==========\n")

    results = []

    for index, case in enumerate(TEST_CASES, 1):
        print(
            f"[{index}/{len(TEST_CASES)}] "
            f"Evaluating case {case['id']}...",
            flush=True,
        )

        try:
            prediction = assess_evidence(
                case["claim"],
                case["evidence"],
            )

            result = {
                "id": case["id"],
                "category": case["category"],
                "claim": case["claim"],
                "evidence": case["evidence"],
                "expected": case["expected"],
                "predicted": prediction["label"],
                "correct": prediction["label"] == case["expected"],
                "explanation": prediction["explanation"],
                "error": None,
            }

            print(
                f"Expected: {result['expected']} | "
                f"Predicted: {result['predicted']} | "
                f"{'PASS' if result['correct'] else 'FAIL'}",
                flush=True,
            )

        except Exception as exc:
            result = {
                "id": case["id"],
                "category": case["category"],
                "claim": case["claim"],
                "evidence": case["evidence"],
                "expected": case["expected"],
                "predicted": None,
                "correct": False,
                "explanation": "",
                "error": str(exc),
            }

            print(f"ERROR: {exc}", flush=True)

        results.append(result)

        # Avoid sending requests too rapidly.
        time.sleep(0.5)

    successful = [
        result for result in results
        if result["predicted"] in LABELS
    ]

    actual = [result["expected"] for result in successful]
    predicted = [result["predicted"] for result in successful]

    print("\n========== EVALUATION SUMMARY ==========\n")

    print(f"Total cases: {len(results)}")
    print(f"Successful predictions: {len(successful)}")
    print(f"Failed requests: {len(results) - len(successful)}")

    if successful:
        print(f"Accuracy: {accuracy_score(actual, predicted):.4f}")

        print("\nClassification Report:")
        print(
            classification_report(
                actual,
                predicted,
                labels=LABELS,
                zero_division=0,
            )
        )

        print("Confusion Matrix:")
        print(confusion_matrix(actual, predicted, labels=LABELS))

        print("\nMisclassified Cases:")

        mistakes = [
            item for item in successful
            if not item["correct"]
        ]

        if mistakes:
            for item in mistakes:
                print(
                    f"Case {item['id']}: "
                    f"Expected={item['expected']}, "
                    f"Predicted={item['predicted']}"
                )
                print(f"Reason: {item['explanation']}")
        else:
            print("No misclassified cases.")

    else:
        print("No successful predictions. Evaluation could not be completed.")

    output = {
        "model": MODEL,
        "total_cases": len(results),
        "successful_predictions": len(successful),
        "accuracy": (
            accuracy_score(actual, predicted)
            if successful else None
        ),
        "results": results,
        "limitations": (
            "Small controlled experimental dataset. "
            "Not clinically validated. "
            "Model classifications are not medical truth."
        ),
    }

    output_path = Path("experiments/nli_evaluation_results.json")
    output_path.write_text(
        json.dumps(output, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nResults saved to: {output_path}")
    print("\n========================================")


if __name__ == "__main__":
    evaluate()
