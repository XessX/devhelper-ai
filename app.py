import os
import json
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

docker_mode = is_docker()

if "history" not in st.session_state:
    st.session_state.history = []

st.set_page_config(page_title="DevHelper AI 🤖", layout="wide")
st.title("🧠 DevHelper AI - Chat with Your Codebase")

# ───────────────────────────────────────────────
source_option = st.radio("📦 Load project from:", ["📁 Local Folder", "🌐 GitHub Repo", "🔗 Website"])
path_input, docs = "", []

if source_option == "📁 Local Folder":
    use_mounted = st.checkbox("📦 Use mounted Docker volume (`/mounted`)", value=docker_mode)

    if use_mounted:
        base_mount = "/mounted"
        try:
            folders = [f for f in os.listdir(base_mount) if os.path.isdir(os.path.join(base_mount, f))]
        except Exception:
            folders = []

        st.markdown("📁 Choose subfolder inside `/mounted`")
        selected_folder = st.selectbox("Subdirectory:", options=[""] + folders)
        path_input = os.path.join(base_mount, selected_folder) if selected_folder else base_mount
        st.success(f"📁 Using: `{path_input}`")
    else:
        path_input = st.text_input("📁 Enter full path to your local folder:")

elif source_option == "🌐 GitHub Repo":
    github_url = st.text_input("🌐 Enter GitHub repo URL:")
    if github_url:
        with st.spinner("🔄 Cloning GitHub repo..."):
            path_input = clone_github_repo(github_url, dest_folder="cloned_repo")

elif source_option == "🔗 Website":
    url = st.text_input("🔗 Enter website URL:")
    if url:
        with st.spinner("🌐 Scraping website..."):
            doc = load_webpage_as_document(url)
            docs = [doc]
            path_input = "web_loaded"

# ───────────────────────────────────────────────
exclude_dirs = st.text_area("🚫 Folders to exclude (comma-separated)", ".venv, node_modules, __pycache__").split(",")
llm_engine = st.radio("🧠 Choose LLM Engine", ["OpenAI", "Ollama"], index=0)

st.subheader("🔧 Chunking Configuration")
smart_mode = st.checkbox("🧠 Auto-Tune Chunk Size", value=True)
chunk_size, chunk_overlap = 800, 100

if smart_mode and path_input and os.path.isdir(path_input):
    suggested_size, suggested_overlap = suggest_chunk_config(path_input)
    st.success(f"✨ Auto-Tune: {suggested_size} chunks | {suggested_overlap} overlap")
    chunk_size, chunk_overlap = suggested_size, suggested_overlap
else:
    chunk_size = st.slider("🧩 Chunk Size", 100, 2000, chunk_size, 100)
    chunk_overlap = st.slider("🔁 Chunk Overlap", 0, 500, chunk_overlap, 50)

# ───────────────────────────────────────────────
if path_input and (os.path.isdir(path_input) or path_input == "web_loaded"):
    try:
        with st.spinner("⚙️ Processing project..."):
            docs_dir = Path(path_input).name
            db_name = f"{docs_dir}_c{chunk_size}_o{chunk_overlap}".replace("/", "_")
            chroma_path = os.path.join("chroma_store", db_name)

            if os.path.exists(chroma_path):
                vectordb = load_chroma(chroma_path)
                st.success("✅ Loaded existing vector DB")
            else:
                if not docs:
                    docs = load_codebase(path_input, exclude_dirs=exclude_dirs)
                chunks = chunk_repo_texts(docs, chunk_size=chunk_size, overlap=chunk_overlap)

                # Ensure all are Document type
                chunks = [
                    doc if isinstance(doc, Document) else
                    Document(page_content=doc.get("page_content", ""), metadata=doc.get("metadata", {}))
                    for doc in chunks
                ]

                vectordb = store_in_chroma(chunks, persist_path=chroma_path)
                st.success("✅ New vector store created")

            qa_chain = get_llm_chain(vectordb, engine=llm_engine.lower())

        if st.checkbox("📜 Preview Chunked Content"):
            preview_chunks(chunks)

        query = st.text_input("💬 Ask something about the codebase or page:")
        if query:
            with st.spinner("🔍 Thinking..."):
                try:
                    response = qa_chain.invoke({"query": query})
                except Exception as e:
                    response = f"❌ Error: {e}"
                st.markdown("**🧠 Answer:**")
                st.write(response)
                st.session_state.history.append({"q": query, "a": response})

        if st.session_state.history:
            hist_json = json.dumps(st.session_state.history, indent=2)
            st.download_button("💾 Download Chat Log", hist_json, file_name="devhelper_chat.json")

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("👈 Please enter a valid folder, GitHub repo, or website.")
