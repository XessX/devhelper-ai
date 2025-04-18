from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

def store_in_chroma(chunks, persist_path="chroma_store"):
    embedding = OllamaEmbeddings(model="llama3")

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

def load_chroma(persist_path="chroma_store"):
    embedding = OllamaEmbeddings(model="llama3")
    return Chroma(
        embedding_function=embedding,
        persist_directory=persist_path
    )
