"""
SIH 26117 — Sovereign On-Premise Agentic AI Workbench
Main Streamlit application — no emojis, full feature set.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

# ── Page config (must come first) ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Workbench - SIH 26117",
    page_icon="assets/favicon.ico" if Path("assets/favicon.ico").exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Core modules (lightweight, zero external AI dependencies) ─────────────────
from src.config import PDF_FOLDER, AUDIT_FILE, N_RESULTS as CFG_N_RESULTS, DEFAULT_MODEL, USERS
from src.auth import check_credentials, register_user, get_user_role
from src.session_store import (
    create_session, get_sessions, save_message,
    get_messages, delete_session, rename_session,
)

# ── Heavy backend imports (lazy / guarded for resilience) ────────────────────
try:
    import ollama as _ollama
    OLLAMA_OK = True
except ImportError:
    OLLAMA_OK = False

try:
    from src.rag import (
        prepare_rag_context, stream_final_answer,
        stream_vision_answer, get_available_models,
    )
    from src.embed import embed_chunks, model as embed_model
    from src.extract import extract_text_from_pdf
    from src.chunker import chunk_text
    from src.vectorstore import (
        add_chunks, list_sources, delete_source, get_chunk_count, collection,
    )
    BACKEND_OK = True
    BACKEND_ERR = ""
except Exception as _e:
    BACKEND_OK = False
    BACKEND_ERR = str(_e)

PDF_FOLDER = Path(PDF_FOLDER)
PDF_FOLDER.mkdir(parents=True, exist_ok=True)
AUDIT_FILE = Path(AUDIT_FILE)
AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def log_audit(event: str, detail: dict):
    record = {"timestamp": datetime.now().isoformat(), "event": event, **detail}
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


@st.cache_data(ttl=20)
def cached_list_sources():
    if not BACKEND_OK:
        return []
    return list_sources()


@st.cache_data(ttl=20)
def cached_get_models():
    if not OLLAMA_OK:
        return ["(ollama not installed)"]
    try:
        return get_available_models()
    except Exception:
        return [DEFAULT_MODEL]


def ingest_pdf(uploaded_file, force: bool = False):
    """Save and ingest a PDF. Returns (chunk_count, error_or_None)."""
    save_path = PDF_FOLDER / uploaded_file.name
    source = save_path.stem

    existing = cached_list_sources()
    if source in existing and not force:
        return 0, f"'{source}' is already ingested. Enable 'Force re-ingest' to overwrite."

    with open(save_path, "wb") as fh:
        fh.write(uploaded_file.getbuffer())

    if source in existing:
        delete_source(source)

    text = extract_text_from_pdf(str(save_path))
    chunks = chunk_text(text)
    if not chunks:
        return 0, "No text could be extracted from this PDF."

    embeddings = embed_chunks(chunks)
    add_chunks(chunks, embeddings, source=source)
    cached_list_sources.clear()
    log_audit("pdf_ingested", {"filename": uploaded_file.name, "chunks": len(chunks), "force": force})
    return len(chunks), None


def save_image(uploaded_file) -> str:
    save_path = PDF_FOLDER / uploaded_file.name
    with open(save_path, "wb") as fh:
        fh.write(uploaded_file.getbuffer())
    log_audit("image_uploaded", {"filename": uploaded_file.name})
    return str(save_path)


def check_ollama_status():
    if not OLLAMA_OK:
        return False, "ollama package not installed"
    try:
        info = _ollama.list()
        count = len(info.get("models", []) if isinstance(info, dict) else info.models)
        return True, f"{count} model(s) available"
    except Exception as e:
        return False, str(e)


# ── Session state defaults ────────────────────────────────────────────────────
_defaults = {
    "authenticated": False,
    "username": "",
    "user_role": "Engineer",
    "session_id": None,
    "messages": [],       # in-memory cache of current session messages
    "agent_trace": [],    # last query trace
    "selected_model": DEFAULT_MODEL,
    "image_path": None,
    "source_filter": [],  # [] = all sources
    "ingested_this_run": [],
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ══════════════════════════════════════════════════════════════════════════════
#  LOGIN & REGISTRATION PAGE
# ══════════════════════════════════════════════════════════════════════════════

def render_login():
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown("## AI Workbench")
        st.markdown("**SIH 26117 — MRPL Sovereign On-Premise Platform**")
        st.caption("Secure Industrial Knowledge Assistant · Confidential Air-Gapped Operation")
        st.divider()

        auth_tab_login, auth_tab_register, auth_tab_network = st.tabs(
            ["Sign In", "Create Account", "Network Access"]
        )

        # ── TAB: Sign In ──────────────────────────────────────────────────────
        with auth_tab_login:
            with st.form("login_form"):
                username = st.text_input("Username", value="admin")
                password = st.text_input("Password", type="password", value="admin123")
                submitted = st.form_submit_button("Sign In", use_container_width=True)

            if submitted:
                if check_credentials(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username.strip()
                    st.session_state.user_role = get_user_role(username.strip())
                    log_audit("login", {"username": username.strip(), "role": st.session_state.user_role})
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

            st.divider()
            st.info(
                "**Pre-configured Demo Accounts:**\n"
                "- Administrator: `admin` / `admin123`\n"
                "- Plant Engineer: `engineer` / `eng456`\n\n"
                "*Or create your own account using the 'Create Account' tab.*"
            )

        # ── TAB: Create Account (Self-Registration) ───────────────────────────
        with auth_tab_register:
            st.markdown("##### Register New User")
            st.caption("Create a new local account. Credentials are stored securely on-premise.")
            with st.form("register_form"):
                new_username = st.text_input("Choose Username")
                new_password = st.text_input("Choose Password", type="password")
                new_role = st.selectbox(
                    "Select Role",
                    ["Engineer", "Plant Operator", "Safety Inspector", "Auditor", "Administrator"],
                )
                reg_submitted = st.form_submit_button("Create Account", use_container_width=True)

            if reg_submitted:
                ok, msg = register_user(new_username, new_password, new_role)
                if ok:
                    st.success(msg)
                    log_audit("user_registered", {"username": new_username, "role": new_role})
                else:
                    st.error(msg)

        # ── TAB: How other devices connect ────────────────────────────────────
        with auth_tab_network:
            st.markdown("##### Accessing from another Laptop / Phone")
            st.markdown("""
To allow teammates, evaluators, or plant engineers to access this workbench from their own device:

1. **Ensure you are on the same Wi-Fi / LAN network**
2. **Start the server bound to all network interfaces:**
   ```bash
   streamlit run app.py --server.address 0.0.0.0
   ```
3. **Open the Network URL on any browser:**
   ```
   http://<YOUR_MACHINE_IP>:8501
   ```
   *(Find your IP by running `ipconfig` in Command Prompt or PowerShell)*
4. Each user can sign up with their own username and have **private chat sessions**.
            """)

        st.divider()
        st.caption("All data and inference remain strictly within your local network perimeter.")


if not st.session_state.authenticated:
    render_login()
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"**AI Workbench** &nbsp;|&nbsp; `{st.session_state.username}`")
    st.caption(f"Role: **{st.session_state.user_role}** · On-Premise · Sovereign")
    st.divider()

    # ── Model ──────────────────────────────────────────────────────────────────
    col_m_title, col_m_btn = st.columns([3, 1])
    with col_m_title:
        st.subheader("Model")
    with col_m_btn:
        if st.button("Reload", help="Refresh installed Ollama models"):
            cached_get_models.clear()
            st.rerun()

    all_models = cached_get_models()
    if st.session_state.selected_model not in all_models and all_models:
        st.session_state.selected_model = all_models[0]
    if all_models:
        st.session_state.selected_model = st.selectbox(
            "LLM model", all_models,
            index=all_models.index(st.session_state.selected_model)
            if st.session_state.selected_model in all_models else 0,
        )
    else:
        st.warning("No models detected. Ensure Ollama is running, then click 'Reload'.")

    st.divider()

    # ── Sessions ───────────────────────────────────────────────────────────────
    st.subheader("Sessions")

    def _load_session(sid: str):
        st.session_state.session_id = sid
        msgs = get_messages(sid) if BACKEND_OK else []
        st.session_state.messages = [
            {"role": m["role"], "content": m["content"], "sources": m["sources"]}
            for m in msgs
        ]

    if BACKEND_OK:
        sessions = get_sessions(st.session_state.username)

        if st.button("New session", use_container_width=True):
            sid = create_session(
                f"Session {len(sessions) + 1}",
                st.session_state.username,
            )
            _load_session(sid)
            st.session_state.agent_trace = []
            st.rerun()

        for sess in sessions:
            is_active = sess["id"] == st.session_state.session_id
            label = f"[active] {sess['name']}" if is_active else sess["name"]
            col_s, col_d = st.columns([4, 1])
            with col_s:
                if st.button(label, key=f"sess_{sess['id']}", use_container_width=True):
                    _load_session(sess["id"])
                    st.rerun()
            with col_d:
                if st.button("X", key=f"del_{sess['id']}"):
                    delete_session(sess["id"])
                    if st.session_state.session_id == sess["id"]:
                        st.session_state.session_id = None
                        st.session_state.messages = []
                    st.rerun()

        if not st.session_state.session_id and sessions:
            _load_session(sessions[0]["id"])

        if not st.session_state.session_id:
            sid = create_session("Session 1", st.session_state.username)
            _load_session(sid)

    st.divider()

    # ── Document upload ────────────────────────────────────────────────────────
    st.subheader("Upload Documents")
    force_reingest = st.checkbox("Force re-ingest (overwrite existing)")
    pdf_files = st.file_uploader(
        "PDF files", type=["pdf"], accept_multiple_files=True,
    )
    if pdf_files and st.button("Ingest PDFs", type="primary", use_container_width=True):
        if not BACKEND_OK:
            st.error(f"Backend error: {BACKEND_ERR}")
        else:
            for f in pdf_files:
                if f.name in st.session_state.ingested_this_run and not force_reingest:
                    st.info(f"{f.name}: already done this run.")
                    continue
                with st.spinner(f"Ingesting {f.name}..."):
                    n, err = ingest_pdf(f, force=force_reingest)
                if err:
                    st.warning(err)
                else:
                    st.success(f"{f.name}: {n} chunks stored.")
                    st.session_state.ingested_this_run.append(f.name)

    st.divider()

    # ── Source filter ──────────────────────────────────────────────────────────
    st.subheader("Source Filter")
    available_sources = cached_list_sources()
    if available_sources:
        st.session_state.source_filter = st.multiselect(
            "Limit search to documents (blank = all)",
            options=available_sources,
            default=st.session_state.source_filter,
        )
    else:
        st.caption("No documents ingested yet.")

    st.divider()

    # ── Image upload ───────────────────────────────────────────────────────────
    st.subheader("Image Analysis")
    img_file = st.file_uploader(
        "Upload image (requires vision model)", type=["png", "jpg", "jpeg", "webp"],
    )
    if img_file:
        st.session_state.image_path = save_image(img_file)
        st.image(img_file, caption=img_file.name, use_container_width=True)
        st.caption("Image will be sent with your next message.")

    if st.session_state.image_path and not img_file:
        if st.button("Clear image"):
            st.session_state.image_path = None

    st.divider()

    # ── Settings ───────────────────────────────────────────────────────────────
    st.subheader("Settings")
    n_results = st.slider("Retrieved chunks", 1, 10, CFG_N_RESULTS)
    show_trace = st.toggle("Show agent trace", value=False)
    show_evidence = st.toggle("Show retrieved evidence", value=False)

    st.divider()
    if st.button("Sign out", use_container_width=True):
        log_audit("logout", {"username": st.session_state.username})
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_chat, tab_docs, tab_status, tab_audit, tab_about = st.tabs(
    ["Chat", "Documents", "Status", "Audit Log", "About"]
)


# ── TAB: Chat ─────────────────────────────────────────────────────────────────
with tab_chat:
    if not BACKEND_OK:
        st.error(f"Backend failed to load: {BACKEND_ERR}")
        st.info("Make sure all packages in requirements.txt are installed and Ollama is running.")
        st.stop()

    if not st.session_state.session_id:
        st.info("Create or select a session from the sidebar to start chatting.")
        st.stop()

    st.subheader("Industrial AI Assistant")
    if st.session_state.source_filter:
        st.caption(f"Searching in: {', '.join(st.session_state.source_filter)}")
    else:
        st.caption("Searching across all documents.")

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources used"):
                    for s in msg["sources"]:
                        st.markdown(f"- `{s}`")

    # Agent trace (last query)
    if show_trace and st.session_state.agent_trace:
        with st.expander("Agent trace (last query)"):
            for step in st.session_state.agent_trace:
                st.markdown(f"**{step['agent']}** — {step['time']:.2f}s")
                st.code(step["detail"], language="text")

    # Chat input
    user_input = st.chat_input("Ask a question about your documents...")

    if user_input:
        # Save & render user message
        st.session_state.messages.append({"role": "user", "content": user_input, "sources": []})
        save_message(st.session_state.session_id, "user", user_input)
        with st.chat_message("user"):
            st.markdown(user_input)

        trace = []
        sources = []
        answer = ""

        # ── Vision mode ───────────────────────────────────────────────────────
        if st.session_state.image_path:
            status_msg = st.empty()
            status_msg.markdown("*Analysing image...*")
            t0 = time.time()
            try:
                with st.chat_message("assistant"):
                    gen = stream_vision_answer(
                        st.session_state.image_path,
                        user_input,
                        st.session_state.selected_model,
                    )
                    answer = st.write_stream(gen)
                trace.append({
                    "agent": "Vision Agent",
                    "time": time.time() - t0,
                    "detail": f"Image: {st.session_state.image_path}\nQuestion: {user_input}",
                })
            except Exception as ex:
                answer = (
                    f"Vision error: {ex}\n\n"
                    "Make sure you are using a vision-capable model such as `llava` or `gemma3:4b`."
                )
                with st.chat_message("assistant"):
                    st.warning(answer)
            status_msg.empty()
            st.session_state.image_path = None

        # ── RAG mode ──────────────────────────────────────────────────────────
        else:
            status_msg = st.empty()
            evidence = []

            try:
                # Stage 1: Planner + Researcher
                status_msg.markdown("*Planner agent analyzing question...*")
                t0 = time.time()
                source_filter = st.session_state.source_filter or None
                plan, context, evidence = prepare_rag_context(
                    user_input,
                    [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]],
                    n_results,
                    st.session_state.selected_model,
                    source_filter,
                )
                prep_time = time.time() - t0
                trace.append({
                    "agent": "Planner + Researcher",
                    "time": prep_time,
                    "detail": json.dumps(plan, indent=2),
                })

                # Stage 2: Stream final answer
                status_msg.markdown("*Generating response...*")
                with st.chat_message("assistant"):
                    gen = stream_final_answer(
                        user_input, plan, context,
                        [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]],
                        st.session_state.selected_model,
                    )
                    answer = st.write_stream(gen)

                    # Show evidence if enabled
                    if show_evidence and evidence:
                        with st.expander("Retrieved evidence"):
                            for i, chunk in enumerate(evidence[:5]):
                                st.markdown(f"**Chunk {i + 1}:**")
                                st.markdown(chunk[:300] + ("..." if len(chunk) > 300 else ""))
                                st.divider()

                # Extract sources from evidence metadata
                try:
                    q_emb = embed_model.encode(user_input)
                    results = collection.query(
                        query_embeddings=[q_emb.tolist()],
                        n_results=n_results,
                        include=["metadatas"],
                    )
                    metas = results.get("metadatas", [[]])[0]
                    sources = sorted({m.get("source", "unknown") for m in metas})
                except Exception:
                    sources = []

                if sources:
                    with st.chat_message("assistant"):
                        pass  # sources shown inside the chat bubble above

            except Exception as ex:
                answer = f"Error: {ex}"
                with st.chat_message("assistant"):
                    st.error(answer)

            status_msg.empty()

        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        save_message(st.session_state.session_id, "assistant", answer, sources)
        st.session_state.agent_trace = trace
        log_audit("query", {
            "username": st.session_state.username,
            "session_id": st.session_state.session_id,
            "question": user_input[:200],
            "model": st.session_state.selected_model,
            "sources": sources,
        })


# ── TAB: Documents ────────────────────────────────────────────────────────────
with tab_docs:
    st.subheader("Document Library")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**PDFs on disk**")
        pdfs = sorted(PDF_FOLDER.glob("*.pdf"))
        if pdfs:
            for p in pdfs:
                size_kb = p.stat().st_size // 1024
                st.markdown(f"- `{p.name}` &nbsp; ({size_kb} KB)")
        else:
            st.info("No PDFs uploaded yet.")

    with col_r:
        st.markdown("**Images on disk**")
        imgs = sorted(
            f for f in PDF_FOLDER.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        )
        if imgs:
            for p in imgs:
                size_kb = p.stat().st_size // 1024
                st.markdown(f"- `{p.name}` &nbsp; ({size_kb} KB)")
        else:
            st.info("No images uploaded yet.")

    st.divider()

    if BACKEND_OK:
        st.markdown("**Ingested sources in vector store**")
        total_chunks = get_chunk_count()
        st.metric("Total chunks", total_chunks)

        sources = list_sources()
        if sources:
            for src in sources:
                col_name, col_btn = st.columns([5, 1])
                with col_name:
                    st.markdown(f"`{src}`")
                with col_btn:
                    if st.button("Delete", key=f"delsrc_{src}"):
                        n = delete_source(src)
                        # Also remove from disk
                        for ext in [".pdf"]:
                            fp = PDF_FOLDER / (src + ext)
                            if fp.exists():
                                fp.unlink()
                        cached_list_sources.clear()
                        log_audit("source_deleted", {
                            "source": src,
                            "chunks_removed": n,
                            "username": st.session_state.username,
                        })
                        st.success(f"Deleted '{src}' ({n} chunks).")
                        st.rerun()
        else:
            st.info("No documents ingested. Upload PDFs from the sidebar.")


# ── TAB: Status ───────────────────────────────────────────────────────────────
with tab_status:
    st.subheader("System Status")

    col1, col2, col3 = st.columns(3)

    # Ollama
    with col1:
        st.markdown("**Ollama (LLM server)**")
        ok, msg = check_ollama_status()
        if ok:
            st.success(f"Online — {msg}")
        else:
            st.error(f"Offline — {msg}")
            st.caption("Start Ollama: `ollama serve`")

    # ChromaDB
    with col2:
        st.markdown("**ChromaDB (Vector store)**")
        if BACKEND_OK:
            try:
                count = get_chunk_count()
                st.success(f"Online — {count} chunks stored")
            except Exception as ex:
                st.error(f"Error: {ex}")
        else:
            st.error("Not available")

    # Backend
    with col3:
        st.markdown("**Python backend**")
        if BACKEND_OK:
            st.success("All modules loaded")
        else:
            st.error(f"Import error")
            st.caption(BACKEND_ERR)

    st.divider()

    if OLLAMA_OK:
        st.markdown("**Available models**")
        models = cached_get_models()
        if models:
            for m in models:
                st.markdown(f"- `{m}`")
        else:
            st.info("No models pulled yet. Run: `ollama pull gemma2:2b`")

    st.divider()

    st.markdown("**Data directories**")
    data_dir = Path("data")
    if data_dir.exists():
        for item in sorted(data_dir.rglob("*")):
            if item.is_file():
                rel = item.relative_to(data_dir)
                size_kb = item.stat().st_size // 1024
                st.markdown(f"- `data/{rel}` &nbsp; ({size_kb} KB)")
    else:
        st.info("data/ directory not found.")

    if st.button("Refresh status"):
        cached_get_models.clear()
        cached_list_sources.clear()
        st.rerun()


# ── TAB: Audit Log ────────────────────────────────────────────────────────────
with tab_audit:
    st.subheader("Audit Log")
    st.caption("All actions are logged locally. Nothing is sent externally.")

    records = load_audit_log()

    if not records:
        st.info("No audit records yet.")
    else:
        col_dl, col_clr, _ = st.columns([1, 1, 3])
        with col_dl:
            st.download_button(
                "Download log (JSONL)",
                data="\n".join(json.dumps(r) for r in records),
                file_name="audit_log.jsonl",
                mime="application/jsonl",
            )
        with col_clr:
            if st.button("Clear log"):
                AUDIT_FILE.unlink(missing_ok=True)
                st.success("Audit log cleared.")
                st.rerun()

        import pandas as pd
        df = pd.DataFrame(records)
        st.dataframe(df, use_container_width=True)


# ── TAB: About ────────────────────────────────────────────────────────────────
with tab_about:
    st.subheader("About — SIH 26117")
    st.markdown("""
**Organisation:** Mangalore Refinery and Petrochemicals Limited (MRPL)

**Title:** Sovereign On-Premise Agentic AI Workbench using Open-Weight Multimodal LLMs for Confidential Industrial Work

---

### Architecture

| Layer | Technology |
|---|---|
| LLM inference | Ollama (Gemma, Llama, Qwen, DeepSeek, ...) |
| Embeddings | sentence-transformers / all-MiniLM-L6-v2 |
| Vector store | ChromaDB (persistent, local) |
| Document ingestion | pypdf |
| Agentic pipeline | Planner → Researcher → Streamed Answer |
| Session storage | SQLite (local) |
| Audit trail | Local JSONL |
| Web UI | Streamlit |
| Container | Docker + docker-compose |

---

### How to use

1. Upload PDFs from the sidebar and click **Ingest PDFs**.
2. Select a model from the dropdown (models are pulled with `ollama pull <name>`).
3. Optionally filter which documents to search using the **Source Filter**.
4. Ask questions in the **Chat** tab — answers stream token by token.
5. For image analysis, upload an image in the sidebar (requires a vision model like `llava` or `gemma3:4b`).
6. Switch between sessions using the **Sessions** panel in the sidebar.
7. View the **Audit Log** tab for a full history of all actions.

---

### Recommended models

```
ollama pull gemma2:2b        # fast, low memory
ollama pull gemma3:4b        # vision + text
ollama pull llama3.2:3b      # general purpose
ollama pull llava:7b         # dedicated vision model
ollama pull qwen2.5:7b       # strong reasoning
```

---

### Data privacy

- All data (PDFs, embeddings, sessions, audit log) stays on this machine.
- No telemetry. No external API calls. No cloud.
- To deploy in a fully air-gapped environment, pre-pull models before disconnecting.

---

### Docker deployment

```bash
docker-compose up --build
```

Ollama and the workbench run as separate containers.
Persistent data is stored in the local `./data/` directory.
    """)
