# TrustMed AI — Local Prototype

A starter Streamlit interface that calls a local Ollama model. It includes basic
chat, a PDF text extraction prototype, and conservative safety messaging.

## Prerequisites
- Python 3.12 virtual environment at `.venv`
- Ollama installed and running
- Model downloaded: `ollama pull qwen3:4b`

## Install dependencies
From the project folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If Windows blocks a native package, stop and inspect the Code Integrity event
rather than disabling Windows security or downloading random DLLs.

## Run
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://localhost:8501`.

## Important limitations
- This is an educational prototype, not a medical device.
- The system prompt does not guarantee safe or correct outputs.
- Uploaded PDF text is passed to the local model as context, truncated to 12,000
  characters. This is not yet a proper RAG pipeline and does not provide citations.
- Scanned PDFs need OCR; this prototype does not implement OCR.
- Avoid identifiable patient data.
