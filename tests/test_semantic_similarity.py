import requests
import numpy as np

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL_NAME = "nomic-embed-text"

claim = "Common symptoms include frequent urination."
evidence = "People with diabetes may urinate more often."

response = requests.post(
    OLLAMA_URL,
    json={
        "model": MODEL_NAME,
        "input": [claim, evidence],
    },
    timeout=120,
)

response.raise_for_status()
data = response.json()

embeddings = np.array(data["embeddings"], dtype=float)

claim_vector = embeddings[0]
evidence_vector = embeddings[1]

similarity = np.dot(claim_vector, evidence_vector) / (
    np.linalg.norm(claim_vector) * np.linalg.norm(evidence_vector)
)

print("Claim:", claim)
print("Evidence:", evidence)
print(f"Cosine similarity: {similarity:.4f}")
print("Embedding dimensions:", len(claim_vector))
