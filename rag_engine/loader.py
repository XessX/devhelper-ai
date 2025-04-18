import os
from typing import List
from langchain_core.documents import Document

VALID_EXTENSIONS = (".py", ".md", ".txt", ".js", ".ts", ".jsx", ".tsx", ".env", ".yaml", ".gitignore")

def is_binary(file_path):
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            if b'\x00' in chunk or chunk.startswith((b'\xff\xfe', b'\xfe\xff')):
                return True
    except:
        return True
    return False

def load_codebase(base_path: str, include_exts=None, exclude_dirs=None) -> List[Document]:
    include_exts = include_exts or VALID_EXTENSIONS
    exclude_dirs = [d.strip() for d in (exclude_dirs or []) if d.strip()]
    docs = []

    for root, _, files in os.walk(base_path):
        if any(part in root.split(os.sep) for part in exclude_dirs):
            continue

        for file in files:
            file_path = os.path.join(root, file)

            if not file.lower().endswith(tuple(include_exts)):
                continue

            if is_binary(file_path):
                print(f"⛔ Skipped binary or non-UTF file: {file_path}")
                continue

            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()

                docs.append(Document(
                    page_content=content,
                    metadata={"source": os.path.relpath(file_path, base_path)}
                ))
                print(f"✅ Loaded: {file_path}")
            except Exception as e:
                print(f"⚠️ Skipped {file_path}: {e}")

    print(f"📄 Total documents loaded: {len(docs)}")
    return docs
