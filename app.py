import os
import json
import shutil
import hashlib
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

from langchain_core.documents import Document
from rag_engine.loader import load_codebase
from rag_engine.chunker import chunk_repo_texts, suggest_chunk_config, preview_chunks
from rag_engine.vector_store import store_in_chroma, load_chroma
from rag_engine.query_engine import get_llm_chain
from rag_engine.utils import clone_github_repo, load_webpage_as_document

# ───────────────────────────────────────────────
load_dotenv()

def is_docker():
    return os.path.exists("/.dockerenv") or os.getenv("DOCKERIZED", "false").lower() == "true"

def is_render():
    return (
        os.getenv("RENDER", "false").lower() == "true"
        or "onrender.com" in os.environ.get("RENDER_EXTERNAL_HOSTNAME", "")
    )

def extract_answer(response):
    if isinstance(response, str):
        return response
    if hasattr(response, "content"):
        return response.content
    if isinstance(response, dict):
        if "content" in response:
            return response["content"]
        if "answer" in response:
            return response["answer"]
    return str(response)

def make_db_name(source, chunk_size, chunk_overlap):
    hash_part = hashlib.md5(source.encode()).hexdigest()[:8]
    return f"{hash_part}_c{chunk_size}_o{chunk_overlap}"

docker_mode = is_docker()
render_mode = is_render()

# --- Remove Ollama/LLM state in cloud (session safety) ---
if render_mode:
    for k in list(st.session_state.keys()):
        if "ollama" in k.lower() or "llm" in k.lower():
            del st.session_state[k]

if "history" not in st.session_state:
    st.session_state.history = []
if "last_github_url" not in st.session_state:
    st.session_state.last_github_url = None

st.set_page_config(page_title="DevHelper AI 🤖", layout="wide")
st.title("🧠 DevHelper AI - Chat with Your Codebase")

# ─────────── SOURCE SELECTION ───────────
source_option = st.radio(
    "📦 Load project from:", 
    ["📁 Local Folder", "🌐 GitHub Repo", "🔗 Website"], 
    key="source_option"
)
path_input, docs = "", []

if source_option == "📁 Local Folder":
    use_mounted = st.checkbox("📦 Use mounted Docker volume (`/mounted`)", value=docker_mode, key="mounted_checkbox")
    if use_mounted:
        base_mount = "/mounted"
        try:
            folders = [f for f in os.listdir(base_mount) if os.path.isdir(os.path.join(base_mount, f))]
        except Exception:
            folders = []
        st.markdown("📁 Choose subfolder inside `/mounted`")
        selected_folder = st.selectbox("Subdirectory:", options=[""] + folders, key="folder_selectbox")
        path_input = os.path.join(base_mount, selected_folder) if selected_folder else base_mount
        st.success(f"📁 Using: `{path_input}`")
    else:
        path_input = st.text_input("📁 Enter full path to your local folder:", key="local_path_input")

elif source_option == "🌐 GitHub Repo":
    github_url = st.text_input("🌐 Enter GitHub repo URL:", key="gh_url_input")
    force_reindex = st.checkbox("🔁 Force re-index this repo", value=False, key="force_reindex_checkbox")
    repo_changed = github_url != st.session_state.last_github_url
    should_reclone = force_reindex or not os.path.exists("cloned_repo") or repo_changed
    if github_url:
        if should_reclone:
            if os.path.exists("cloned_repo"):
                shutil.rmtree("cloned_repo")
            try:
                with st.spinner("🔄 Cloning GitHub repo..."):
                    path_input = clone_github_repo(github_url, dest_folder="cloned_repo")
                st.session_state.last_github_url = github_url
                st.success("✅ Repo cloned successfully.")
            except Exception as e:
                st.error(str(e))
                path_input = ""
        else:
            path_input = "cloned_repo"

elif source_option == "🔗 Website":
    url = st.text_input("🔗 Enter website URL:", key="web_url_input")
    if url:
        with st.spinner("🌐 Scraping website..."):
            doc = load_webpage_as_document(url)
            docs = [doc]
            path_input = "web_loaded"

exclude_dirs = st.text_area(
    "🚫 Folders to exclude (comma-separated)", 
    ".venv, node_modules, __pycache__", 
    key="exclude_dirs_textarea"
).split(",")

# --- LLM Engine Selection ---
if render_mode:
    llm_engine = "openai"
    st.info("🌐 Running in online mode: Only OpenAI is available.")
else:
    llm_engine = st.radio(
        "🧠 Choose LLM Engine", 
        ["OpenAI", "Ollama"], 
        index=0, 
        key="llm_radio"
    ).lower()

# --- Chunking Config ---
st.subheader("🔧 Chunking Configuration")
smart_mode = st.checkbox("🧠 Auto-Tune Chunk Size", value=True, key="autotune_checkbox")
chunk_size, chunk_overlap = 800, 100

if smart_mode and path_input and os.path.isdir(path_input):
    suggested_size, suggested_overlap = suggest_chunk_config(path_input)
    st.success(f"✨ Auto-Tune: {suggested_size} chunks | {suggested_overlap} overlap")
    chunk_size, chunk_overlap = suggested_size, suggested_overlap
else:
    chunk_size = st.slider("🧩 Chunk Size", 100, 2000, chunk_size, 100, key="chunk_slider")
    chunk_overlap = st.slider("🔁 Chunk Overlap", 0, 500, chunk_overlap, 50, key="overlap_slider")

# --- MAIN LOGIC ---
if path_input and (os.path.isdir(path_input) or path_input == "web_loaded"):
    try:
        if source_option == "🌐 GitHub Repo":
            source_key = github_url.strip()
        elif source_option == "🔗 Website":
            source_key = url.strip()
        else:
            source_key = path_input.strip()
        db_name = make_db_name(source_key, chunk_size, chunk_overlap)
        chroma_path = os.path.join("chroma_store", db_name)

        force_reindex_other = False
        if source_option != "🌐 GitHub Repo":
            force_reindex_other = st.checkbox("🔁 Force re-index this repo", value=False, key="force_reindex_other")

        with st.spinner("⚙️ Processing project..."):
            do_reindex = force_reindex if source_option == "🌐 GitHub Repo" else force_reindex_other

            if os.path.exists(chroma_path) and not do_reindex:
                vectordb = load_chroma(chroma_path)
                st.success("✅ Loaded existing vector DB")
                chunks = None
            else:
                if not docs:
                    docs = load_codebase(path_input, exclude_dirs=exclude_dirs)
                chunks = chunk_repo_texts(docs, chunk_size=chunk_size, overlap=chunk_overlap)
                chunks = [
                    doc if isinstance(doc, Document) else
                    Document(page_content=doc.get("page_content", ""), metadata=doc.get("metadata", {}))
                    for doc in chunks
                ]
                vectordb = store_in_chroma(chunks, persist_path=chroma_path)
                st.success("✅ New vector store created")

            # --- SAFETY: Always use OpenAI on Render/cloud ---
            qa_chain = get_llm_chain(
                vectordb,
                engine="openai" if render_mode else llm_engine
            )

        if st.checkbox("📜 Preview Chunked Content", key="preview_checkbox"):
            if 'chunks' not in locals() or chunks is None:
                docs = load_codebase(path_input, exclude_dirs=exclude_dirs)
                chunks = chunk_repo_texts(docs, chunk_size=chunk_size, overlap=chunk_overlap)
            preview_chunks(chunks)

        query = st.text_input("💬 Ask something about the codebase or page:", key="query_input")
        if query:
            with st.spinner("🔍 Thinking..."):
                try:
                    response = qa_chain.invoke({"query": query})
                    answer = extract_answer(response)
                except Exception as e:
                    answer = f"❌ Error: {e}"
                st.markdown("**🧠 Answer:**")
                st.write(answer)
                st.session_state.history.append({"q": query, "a": answer})

        if st.session_state.history:
            hist_json = json.dumps(st.session_state.history, indent=2)
            st.download_button(
                "💾 Download Chat Log", 
                hist_json, 
                file_name="devhelper_chat.json", 
                key="dl_button"
            )

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("👈 Please enter a valid folder, GitHub repo, or website.")
