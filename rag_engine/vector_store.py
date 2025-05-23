import os
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Optional imports, guarded for environments where not installed
try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    OpenAIEmbeddings = None

try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    OllamaEmbeddings = None

def is_render():
    """
    Returns True if running on Render/cloud.
    """
    return (
        os.getenv("RENDER", "false").lower() == "true"
        or "onrender.com" in os.environ.get("RENDER_EXTERNAL_HOSTNAME", "")
    )

def _get_embedding(engine: str = "openai"):
    """
    Returns the correct embedding instance.
    - Always uses OpenAI on Render/cloud, regardless of user choice.
    - Locally, uses OpenAI or Ollama as requested.
    """
    if is_render() or engine == "openai":
        if OpenAIEmbeddings is None:
            raise ImportError("OpenAIEmbeddings is not installed. Please install langchain_openai.")
        return OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    elif engine == "ollama":
        if OllamaEmbeddings is None:
            raise ImportError("OllamaEmbeddings is not installed. Please install langchain_ollama.")
        return OllamaEmbeddings(model="llama3")
    else:
        raise ValueError("Unknown embedding engine: choose 'openai' or 'ollama'.")

def store_in_chroma(chunks, persist_path="chroma_store", embedding_engine="openai"):
    """
    Stores document chunks in a Chroma vector DB using the correct embedding engine.
    """
    embedding = _get_embedding(embedding_engine)

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

    return Chroma.from_documents(
        documents=valid_docs,
        embedding=embedding,
        persist_directory=persist_path
    )

def load_chroma(persist_path="chroma_store", embedding_engine="openai"):
    """
    Loads a Chroma vector DB using the correct embedding engine.
    """
    embedding = _get_embedding(embedding_engine)
    return Chroma(
        embedding_function=embedding,
        persist_directory=persist_path
    )
