from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
import streamlit as st

def chunk_repo_texts(docs, chunk_size=600, overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    readme_chunks = []
    other_chunks = []

    for doc in docs:
        if not isinstance(doc, Document):
            doc = Document(
                page_content=doc.get("page_content", ""),
                metadata=doc.get("metadata", {})
            )

        if "readme.md" in doc.metadata.get("source", "").lower():
            readme_chunks.extend(splitter.split_documents([doc]) * 2)
        else:
            other_chunks.extend(splitter.split_documents([doc]))

    return readme_chunks + other_chunks

def suggest_chunk_config(path):
    total_lines = 0
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith((".py", ".md", ".txt", ".js", ".ts", ".json")):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                        total_lines += len(f.readlines())
                except:
                    continue
    if total_lines < 500:
        return 400, 50
    elif total_lines < 1500:
        return 800, 100
    else:
        return 1200, 150

def preview_chunks(chunks):
    st.markdown(f"📦 Showing {len(chunks)} chunks:")
    for i, c in enumerate(chunks[:5]):
        st.code(c.page_content, language='text')
        st.caption(f"📄 {c.metadata.get('source', 'Unknown')} | Chunk #{i+1}")
