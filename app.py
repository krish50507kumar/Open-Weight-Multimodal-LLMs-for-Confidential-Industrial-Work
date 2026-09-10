"""
SIH 26117 — Sovereign On-Premise Agentic AI Workbench
Streamlit Web UI
"""

import os
import time
import json
import base64
from datetime import datetime
from pathlib import Path

import streamlit as st

# ── page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="AI Workbench – SIH 26117",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports (so the app at least renders if deps are missing) ────────────
try:
    import ollama as _ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from src.rag import answer_question, get_available_models
    from src.embed import embed_chunks, model as embed_model
    from src.extract import extract_text_from_pdf
    from src.chunker import chunk_text
    from src.vectorstore import add_chunks, collection
    RAG_AVAILABLE = True
except Exception as e:
    RAG_AVAILABLE = False
    _RAG_ERROR = str(e)

PDF_FOLDER = Path("data/raw_pdfs")
PDF_FOLDER.mkdir(parents=True, exist_ok=True)

AUDIT_FILE = Path("data/audit_log.jsonl")
AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

def log_audit(event: str, detail: dict):
    record = {
        "timestamp": datetime.now().isoformat(),
        "event": event,
        **detail,
    }
    with open(AUDIT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_audit_log():
    if not AUDIT_FILE.exists():
        return []
    records = []
    with open(AUDIT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def get_models():
    if not OLLAMA_AVAILABLE:
        return ["(ollama not installed)"]
    try:
        return get_available_models() if RAG_AVAILABLE else []
    except Exception:
        return ["gemma2:2b", "gemma3:4b", "llama3.2:3b"]


def ingest_pdf(uploaded_file):
    """Save uploaded PDF and ingest into ChromaDB."""
    save_path = PDF_FOLDER / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    source_name = save_path.stem
    text = extract_text_from_pdf(str(save_path))
    chunks = chunk_text(text)
    if not chunks:
        return 0, "No text extracted."
    embeddings = embed_chunks(chunks)
    add_chunks(chunks, embeddings, source=source_name)
    log_audit("pdf_ingested", {"filename": uploaded_file.name, "chunks": len(chunks)})
    return len(chunks), None


def ingest_image(uploaded_file):
    """Save image and return a basic description (no vision model needed)."""
    save_path = PDF_FOLDER / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    log_audit("image_uploaded", {"filename": uploaded_file.name})
    return str(save_path)


def chat_with_image(image_path: str, question: str, model: str):
    """Send image + question to Ollama (vision model)."""
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode()
    response = _ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": question,
            "images": [img_b64],
        }],
    )
    return response["message"]["content"]


# ── Session state defaults ────────────────────────────────────────────────────
for key, default in {
    "messages": [],
    "agent_trace": [],
    "selected_model": None,
    "image_path": None,
    "uploaded_docs": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/3/3b/MRPL_Logo.png", width=120)
    st.title("AI Workbench")
    st.caption("SIH 26117 · On-Premise · Sovereign")
    st.divider()

    # ── Model selection ──────────────────────────────────────────────────────
    st.subheader("🤖 LLM Model")
    available_models = get_models()
    if available_models:
        if st.session_state.selected_model not in available_models:
            st.session_state.selected_model = available_models[0]
        selected_model = st.selectbox(
            "Select model", available_models,
            index=available_models.index(st.session_state.selected_model),
        )
        st.session_state.selected_model = selected_model
    else:
        st.warning("No Ollama models found. Run `ollama pull gemma2:2b`")
        selected_model = "gemma2:2b"

    st.divider()

    # ── Document ingestion ───────────────────────────────────────────────────
    st.subheader("📄 Upload Documents")
    pdf_files = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True,
        help="PDFs are stored locally and never leave the server.",
    )
    if pdf_files:
        if st.button("Ingest PDFs", type="primary"):
            if not RAG_AVAILABLE:
                st.error(f"RAG stack not available: {_RAG_ERROR}")
            else:
                for f in pdf_files:
                    if f.name not in st.session_state.uploaded_docs:
                        with st.spinner(f"Ingesting {f.name}…"):
                            n, err = ingest_pdf(f)
                        if err:
                            st.error(f"{f.name}: {err}")
                        else:
                            st.success(f"{f.name} → {n} chunks stored")
                            st.session_state.uploaded_docs.append(f.name)

    if st.session_state.uploaded_docs:
        st.caption("Ingested docs:")
        for d in st.session_state.uploaded_docs:
            st.markdown(f"- `{d}`")

    st.divider()

    # ── Image upload for vision ──────────────────────────────────────────────
    st.subheader("🖼️ Upload Image (Vision)")
    img_file = st.file_uploader(
        "Upload image for analysis", type=["png", "jpg", "jpeg", "webp"],
    )
    if img_file:
        img_path = ingest_image(img_file)
        st.session_state.image_path = img_path
        st.image(img_file, caption=img_file.name, use_container_width=True)
        st.caption("This image will be attached to your next message.")

    if st.session_state.image_path and not img_file:
        if st.button("Clear image"):
            st.session_state.image_path = None

    st.divider()

    # ── Settings ─────────────────────────────────────────────────────────────
    st.subheader("⚙️ Settings")
    n_results = st.slider("Retrieved chunks (RAG)", 1, 10, 3)
    show_trace = st.toggle("Show agent trace", value=True)

    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.session_state.agent_trace = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA — tabs
# ══════════════════════════════════════════════════════════════════════════════
tab_chat, tab_docs, tab_audit, tab_about = st.tabs(
    ["💬 Chat", "📚 Documents", "🔍 Audit Log", "ℹ️ About"]
)

# ── TAB: Chat ─────────────────────────────────────────────────────────────────
with tab_chat:
    st.header("Industrial AI Assistant")

    if not RAG_AVAILABLE:
        st.error(
            f"⚠️ RAG stack failed to load: `{_RAG_ERROR}`\n\n"
            "Make sure ChromaDB, sentence-transformers, and ollama are installed."
        )

    # render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Sources used"):
                    for s in msg["sources"]:
                        st.markdown(f"- {s}")

    # Agent trace panel (collapsible)
    if show_trace and st.session_state.agent_trace:
        with st.expander("🔬 Agent trace (last query)", expanded=False):
            for step in st.session_state.agent_trace:
                st.markdown(f"**{step['agent']}** · `{step['time']:.2f}s`")
                st.code(step["detail"], language="text")

    # Chat input
    user_input = st.chat_input(
        "Ask a question about your industrial documents…",
        disabled=(not RAG_AVAILABLE and not OLLAMA_AVAILABLE),
    )

    if user_input:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # ── Build answer ──────────────────────────────────────────────────
        trace = []
        sources = []

        with st.chat_message("assistant"):
            status_box = st.empty()

            # Check if we have an image attached → vision mode
            if st.session_state.image_path and OLLAMA_AVAILABLE:
                status_box.markdown("🖼️ *Analysing image…*")
                t0 = time.time()
                try:
                    answer = chat_with_image(
                        st.session_state.image_path, user_input,
                        st.session_state.selected_model,
                    )
                    trace.append({
                        "agent": "Vision Agent",
                        "time": time.time() - t0,
                        "detail": f"Image: {st.session_state.image_path}\nQ: {user_input}",
                    })
                    st.session_state.image_path = None  # clear after use
                except Exception as e:
                    answer = f"⚠️ Vision error: {e}\n\n(Make sure you use a vision model like `llava` or `gemma3:4b`)"

            elif RAG_AVAILABLE:
                # Planner stage
                status_box.markdown("🧠 *Planner agent thinking…*")
                t0 = time.time()
                try:
                    answer = answer_question(
                        user_input,
                        st.session_state.messages[:-1],  # history without current
                        n_results,
                        st.session_state.selected_model,
                    )
                    trace.append({
                        "agent": "Planner → Researcher → Answer",
                        "time": time.time() - t0,
                        "detail": f"Model: {st.session_state.selected_model}\nQ: {user_input}",
                    })
                    # try to get sources from collection metadata
                    try:
                        q_emb = embed_model.encode(user_input)
                        results = collection.query(
                            query_embeddings=[q_emb.tolist()], n_results=n_results,
                            include=["metadatas"],
                        )
                        metas = results.get("metadatas", [[]])[0]
                        sources = list({m.get("source", "unknown") for m in metas})
                    except Exception:
                        sources = []
                except Exception as e:
                    answer = f"⚠️ Error: {e}"

            elif OLLAMA_AVAILABLE:
                # Fallback: direct LLM (no RAG)
                status_box.markdown("💬 *Generating response (no docs loaded)…*")
                t0 = time.time()
                try:
                    resp = _ollama.chat(
                        model=st.session_state.selected_model,
                        messages=[{"role": "user", "content": user_input}],
                    )
                    answer = resp["message"]["content"]
                    trace.append({
                        "agent": "Direct LLM (no RAG)",
                        "time": time.time() - t0,
                        "detail": f"Model: {st.session_state.selected_model}",
                    })
                except Exception as e:
                    answer = f"⚠️ Ollama error: {e}"
            else:
                answer = "⚠️ Neither RAG stack nor Ollama is available. Please check your installation."

            status_box.empty()
            st.markdown(answer)
            if sources:
                with st.expander("📎 Sources used"):
                    for s in sources:
                        st.markdown(f"- `{s}`")

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
        st.session_state.agent_trace = trace

        log_audit("query", {
            "question": user_input,
            "model": st.session_state.selected_model,
            "sources": sources,
        })


# ── TAB: Documents ────────────────────────────────────────────────────────────
with tab_docs:
    st.header("📚 Document Library")

    pdf_paths = list(PDF_FOLDER.glob("*.pdf"))
    img_paths = list(PDF_FOLDER.glob("*.png")) + \
                list(PDF_FOLDER.glob("*.jpg")) + \
                list(PDF_FOLDER.glob("*.jpeg"))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"PDFs ({len(pdf_paths)})")
        if pdf_paths:
            for p in pdf_paths:
                st.markdown(f"📄 `{p.name}` — {p.stat().st_size // 1024} KB")
        else:
            st.info("No PDFs uploaded yet. Use the sidebar to upload.")

    with col2:
        st.subheader(f"Images ({len(img_paths)})")
        if img_paths:
            for p in img_paths:
                st.markdown(f"🖼️ `{p.name}` — {p.stat().st_size // 1024} KB")
        else:
            st.info("No images uploaded yet.")

    st.divider()

    # ChromaDB stats
    if RAG_AVAILABLE:
        try:
            count = collection.count()
            st.metric("Total chunks in vector store", count)
        except Exception as e:
            st.warning(f"Could not read ChromaDB: {e}")


# ── TAB: Audit Log ────────────────────────────────────────────────────────────
with tab_audit:
    st.header("🔍 Audit Log")
    st.caption("All actions are logged locally. Nothing leaves the server.")

    records = load_audit_log()

    if not records:
        st.info("No audit records yet.")
    else:
        col_dl, col_clr = st.columns([1, 1])
        with col_dl:
            st.download_button(
                "⬇️ Download log (JSONL)",
                data="\n".join(json.dumps(r) for r in records),
                file_name="audit_log.jsonl",
                mime="application/jsonl",
            )
        with col_clr:
            if st.button("🗑️ Clear audit log"):
                AUDIT_FILE.unlink(missing_ok=True)
                st.success("Audit log cleared.")
                st.rerun()

        # Display as table
        import pandas as pd
        df = pd.DataFrame(records)
        st.dataframe(df, use_container_width=True)


# ── TAB: About ────────────────────────────────────────────────────────────────
with tab_about:
    st.header("ℹ️ About this Workbench")
    st.markdown("""
## SIH 26117 — Sovereign On-Premise Agentic AI Workbench

**Organisation:** Mangalore Refinery and Petrochemicals Limited (MRPL)

### What it does
This workbench allows industrial employees to query **confidential internal documents** 
using state-of-the-art open-weight LLMs — **entirely on-premise**, with zero data leaving 
the organisation's network.

### Architecture

| Layer | Technology |
|---|---|
| LLM inference | [Ollama](https://ollama.com) (Gemma, Llama, Qwen, …) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | [ChromaDB](https://www.trychroma.com) (persistent, local) |
| Document ingestion | pypdf |
| Agentic pipeline | Custom Planner → Researcher → Answerer |
| Web UI | [Streamlit](https://streamlit.io) |
| Audit trail | Local JSONL file |

### How to use
1. **Upload PDFs** from the sidebar and click **Ingest PDFs**.
2. **Select a model** from the dropdown (models must be pulled with `ollama pull <model>`).
3. **Ask questions** in the Chat tab.
4. For image analysis, upload an image and ask a question (requires a vision model like `llava` or `gemma3:4b`).
5. Check the **Audit Log** tab for a complete action history.

### Data privacy
- All data stays on your machine.
- No telemetry, no external API calls.
- ChromaDB and PDFs are stored in `data/`.

### Recommended models (pull with ollama)
```
ollama pull gemma2:2b       # fast, small
ollama pull gemma3:4b       # multimodal (vision)
ollama pull llama3.2:3b     # general purpose
ollama pull llava:7b        # vision
```
    """)
