import os
import requests
import zipfile
import io
import shutil
from bs4 import BeautifulSoup
from langchain_core.documents import Document

def clone_github_repo(repo_url: str, dest_folder="cloned_repo"):
    repo_url = repo_url.rstrip("/")
    repo_name = repo_url.split("/")[-1]
    zip_url = f"{repo_url}/archive/refs/heads/main.zip"

    response = requests.get(zip_url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(dest_folder)

    inner_folder = os.path.join(dest_folder, f"{repo_name}-main")
    for root, _, files in os.walk(inner_folder):
        for file in files:
            src = os.path.join(root, file)
            dst_dir = root.replace(inner_folder, dest_folder)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.move(src, os.path.join(dst_dir, file))
    shutil.rmtree(inner_folder)
    return dest_folder

def load_webpage_as_document(url: str) -> Document:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()
    return Document(page_content=text, metadata={"source": url})
