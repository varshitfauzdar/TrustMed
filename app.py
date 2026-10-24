import streamlit as st
from modules.llm_engine import chat_with_ollama, ollama_status
from modules.safety_guard import get_system_prompt
from modules.document_parser import extract_pdf_text

APP_NAME = "TrustMed AI"
MODEL_NAME = "qwen3:4b"

st.set_page_config(page_title=APP_NAME, page_icon="🩺", layout="wide")

st.title("🩺 TrustMed AI")
st.caption("Local AI assistant for general healthcare information")

st.warning(
    "Experimental prototype—not a medical device and not a substitute for a clinician. "
    "It may produce incorrect information. Do not use it for emergencies or to make "
    "diagnosis or treatment decisions."
)

with st.sidebar:
    st.header("System")
    st.write(f"**Model:** `{MODEL_NAME}`")
    if st.button("Check Ollama connection"):
        ok, message = ollama_status()
        (st.success if ok else st.error)(message)

    st.divider()
    st.header("Medical document (prototype)")
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_pdf:
        try:
            pdf_text = extract_pdf_text(uploaded_pdf.getvalue())
            st.session_state["document_text"] = pdf_text
            st.success(f"Extracted {len(pdf_text):,} characters.")
            st.download_button(
                "Download extracted text",
                data=pdf_text,
                file_name="extracted_document.txt",
                mime="text/plain",
            )
        except Exception as exc:
            st.error(f"Could not read this PDF: {exc}")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_text" not in st.session_state:
    st.session_state.document_text = ""

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a general healthcare question...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    system_prompt = get_system_prompt()
    document_text = st.session_state.document_text
    if document_text:
        # Prototype document context: truncated to avoid oversized requests.
        system_prompt += (
            "\n\nThe user has uploaded a document. Use only the excerpt below "
            "when discussing its contents. State clearly when the excerpt does "
            "not contain the answer. Do not treat it as verified clinical guidance.\n\n"
            "DOCUMENT EXCERPT:\n" + document_text[:12000]
        )

    conversation = [{"role": "system", "content": system_prompt}]
    conversation.extend(st.session_state.messages)

    with st.chat_message("assistant"):
        with st.spinner("Waiting for local model..."):
            answer = chat_with_ollama(MODEL_NAME, conversation)
            if answer["ok"]:
                st.markdown(answer["content"])
            else:
                st.error(answer["content"])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer["content"]}
    )

st.caption(
    "Privacy note: prompts and uploaded document text are sent to your local Ollama "
    "service on this computer. This prototype does not call a cloud LLM API. "
    "Avoid uploading identifiable patient information."
)
