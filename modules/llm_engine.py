import requests
import re

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 300


def ollama_status():
    """Check Ollama connectivity and available models."""
    try:
        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=10,
        )
        response.raise_for_status()

        data = response.json()
        names = [
            item.get("name", "")
            for item in data.get("models", [])
            if item.get("name")
        ]

        if names:
            return True, "Ollama is reachable. Models: " + ", ".join(names)

        return True, "Ollama is reachable, but no models were found."

    except requests.RequestException as exc:
        return False, f"Cannot connect to Ollama: {exc}"

    except (ValueError, TypeError) as exc:
        return False, f"Could not parse Ollama response: {exc}"


def clean_model_response(content):
    """
    Remove explicit Qwen thinking blocks and reject obvious internal planning.
    This is a heuristic, not a guarantee of answer quality or safety.
    """
    if not isinstance(content, str):
        return ""

    content = content.strip()

    if "</think>" in content:
        content = content.rsplit("</think>", 1)[-1].strip()
    elif "<think>" in content:
        return ""

    if not content:
        return ""

    # Detect obvious internal-planning text rather than displaying it as an answer.
    planning_patterns = [
        r"(?im)^\s*I need to provide\b",
        r"(?im)^\s*Let me analyze\b",
        r"(?im)^\s*I should structure my response\b",
        r"(?im)^\s*I must be careful to\b",
        r"(?im)^\s*I need to structure\b",
    ]

    if any(re.search(pattern, content) for pattern in planning_patterns):
        return ""

    # Remove accidental surrounding Markdown code fences.
    content = re.sub(r"^```(?:markdown|md|text)?\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\s*```$", "", content)

    return content.strip()


def chat_with_ollama(
    model,
    messages,
    timeout=DEFAULT_TIMEOUT,
    temperature=0.2,
):
    """Generate a response using a local Ollama model."""

    if not isinstance(model, str) or not model.strip():
        return {
            "ok": False,
            "content": "A valid Ollama model name is required.",
        }

    if not isinstance(messages, list) or not messages:
        return {
            "ok": False,
            "content": "At least one chat message is required.",
        }

    valid_messages = all(
        isinstance(message, dict)
        and isinstance(message.get("role"), str)
        and isinstance(message.get("content"), str)
        for message in messages
    )

    if not valid_messages:
        return {
            "ok": False,
            "content": "Invalid chat message format.",
        }

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model.strip(),
                "messages": messages,
                "stream": False,
                "think": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": 512,
                },
            },
            timeout=timeout,
        )

        response.raise_for_status()
        data = response.json()

        message = data.get("message", {})
        content = clean_model_response(message.get("content", ""))

        if not content:
            return {
                "ok": False,
                "content": (
                    "The model did not return a usable final answer. "
                    "Please retry with a shorter question."
                ),
            }

        return {
            "ok": True,
            "content": content,
        }

    except requests.exceptions.Timeout:
        return {
            "ok": False,
            "content": (
                f"The model exceeded the {timeout}-second timeout. "
                "Try a shorter question or reduce the conversation context."
            ),
        }

    except requests.exceptions.ConnectionError:
        return {
            "ok": False,
            "content": (
                "Could not connect to Ollama. "
                "Make sure Ollama is running."
            ),
        }

    except requests.RequestException as exc:
        return {
            "ok": False,
            "content": f"Ollama request failed: {exc}",
        }

    except (ValueError, TypeError) as exc:
        return {
            "ok": False,
            "content": f"Could not parse Ollama response: {exc}",
        }
