from langchain_chroma import Chroma
from langchain_core.documents import Document

# Try importing Ollama, guard if not installed
try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    OllamaEmbeddings = None

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    OpenAIEmbeddings = None

import os

def is_render():
    return (
        os.getenv("RENDER", "false").lower() == "true"
        or "onrender.com" in os.environ.get("RENDER_EXTERNAL_HOSTNAME", "")
    )

def _get_embedding(engine: str = "openai"):
    """
    Returns the correct embedding function for the environment.
    Default: OpenAI on Render/cloud, Ollama if local and requested.
    """
    if is_render() or engine == "openai":
        if OpenAIEmbeddings is None:
            raise ImportError("OpenAIEmbeddings not installed.")
        return OpenAIEmbeddings()
    elif engine == "ollama":
        if OllamaEmbeddings is None:
            raise ImportError("OllamaEmbeddings not installed.")
        return OllamaEmbeddings(model="llama3")
    else:
        raise ValueError("Unknown embedding engine: choose 'openai' or 'ollama'")

def store_in_chroma(chunks, persist_path="chroma_store", embedding_engine="openai"):
    embedding = _get_embedding(embedding_engine)

    # Ensure all items are Document instances
    valid_docs = []
    for doc in chunks:
        if isinstance(doc, Document):
            valid_docs.append(doc)
        elif isinstance(doc, dict):
            valid_docs.append(Document(
                page_content=doc.get("page_content", ""),
                metadata=doc.get("metadata", {})
            ))
        else:
            raise ValueError("Chunk must be a Document or dict.")

    # Build and persist vector store
    return Chroma.from_documents(
        documents=valid_docs,
        embedding=embedding,
        persist_directory=persist_path
    )

def load_chroma(persist_path="chroma_store", embedding_engine="openai"):
    embedding = _get_embedding(embedding_engine)
    return Chroma(
        embedding_function=embedding,
        persist_directory=persist_path
    )
