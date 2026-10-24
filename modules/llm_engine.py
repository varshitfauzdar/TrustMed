import requests

OLLAMA_BASE_URL = "http://localhost:11434"


def ollama_status():
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        names = [item.get("name", "") for item in data.get("models", [])]
        if names:
            return True, "Ollama is reachable. Models: " + ", ".join(names)
        return True, "Ollama is reachable, but no models were found."
    except requests.RequestException as exc:
        return False, (
            "Cannot connect to Ollama at http://localhost:11434. "
            "Start the Ollama app, then retry. Details: " + str(exc)
        )


def chat_with_ollama(model, messages):
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "think": False,
            },
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("message", {}).get("content", "").strip()
        if not content:
            return {"ok": False, "content": "The model returned an empty response. Try again."}
        return {"ok": True, "content": content}
    except requests.exceptions.ConnectionError:
        return {
            "ok": False,
            "content": "Could not connect to Ollama. Open the Ollama application and retry.",
        }
    except requests.exceptions.Timeout:
        return {
            "ok": False,
            "content": "The model timed out. Try a shorter question or retry.",
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
