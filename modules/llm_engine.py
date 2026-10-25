import requests

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 180


def ollama_status():
    """Check Ollama connectivity and list locally available models."""
    try:
        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=5,
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
        return False, (
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            "Start the Ollama application and retry. "
            f"Details: {exc}"
        )

    except (ValueError, TypeError) as exc:
        return False, f"Could not parse Ollama's response: {exc}"


def chat_with_ollama(
    model,
    messages,
    timeout=DEFAULT_TIMEOUT,
    temperature=0.2,
):
    """
    Generate a response using a locally installed Ollama model.

    Returns:
        {"ok": True, "content": "..."} on success.
        {"ok": False, "content": "..."} on failure.
    """
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

    if not all(
        isinstance(message, dict)
        and isinstance(message.get("role"), str)
        and isinstance(message.get("content"), str)
        for message in messages
    ):
        return {
            "ok": False,
            "content": (
                "Each message must contain string values for "
                "'role' and 'content'."
            ),
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
                },
            },
            timeout=timeout,
        )

        response.raise_for_status()
        data = response.json()

        content = data.get("message", {}).get("content", "")

        if not isinstance(content, str) or not content.strip():
            return {
                "ok": False,
                "content": (
                    "The model returned an empty response. "
                    "Check the model and retry."
                ),
            }

        return {
            "ok": True,
            "content": content.strip(),
        }

    except requests.exceptions.ConnectionError:
        return {
            "ok": False,
            "content": (
                "Could not connect to Ollama. "
                "Open the Ollama application and retry."
            ),
        }

    except requests.exceptions.Timeout:
        return {
            "ok": False,
            "content": (
                "The model timed out. Try a shorter question "
                "or increase the timeout."
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
            "content": f"Could not parse Ollama's response: {exc}",
        }
