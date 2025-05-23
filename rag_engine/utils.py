import os
import requests
import zipfile
import io
import shutil
from bs4 import BeautifulSoup
from langchain_core.documents import Document

def is_valid_github_repo_url(repo_url: str):
    if not repo_url.startswith("https://github.com/"):
        return False
    parts = repo_url[len("https://github.com/"):].strip("/").split("/")
    return len(parts) == 2 and all(parts)

def get_default_branch(repo_url):
    parts = repo_url.rstrip("/").split("/")
    if len(parts) < 2:
        raise ValueError("❌ Invalid GitHub repo URL.")
    user, repo = parts[-2], parts[-1]
    api_url = f"https://api.github.com/repos/{user}/{repo}"
    response = requests.get(api_url, headers={"Accept": "application/vnd.github.v3+json"})
    if response.status_code != 200:
        raise ValueError("❌ Could not access repo details. Is it public?")
    return response.json().get("default_branch", "main")

def clone_github_repo(repo_url: str, dest_folder="cloned_repo", force_delete=False):
    if not is_valid_github_repo_url(repo_url):
        raise ValueError("❌ Please enter a valid GitHub **repository** URL: https://github.com/<user>/<repo>")

    # Always clean dest folder before cloning to avoid stale files
    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)

    repo_url = repo_url.rstrip("/")
    repo_name = repo_url.split("/")[-1]

    try:
        default_branch = get_default_branch(repo_url)
    except Exception as e:
        raise ValueError(str(e))

    zip_url = f"{repo_url}/archive/refs/heads/{default_branch}.zip"

    response = requests.get(zip_url)
    if response.status_code != 200:
        raise ValueError(
            f"❌ Could not fetch repo ZIP for branch '{default_branch}'. "
            "Please check that the repository exists and is public."
        )

    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(dest_folder)
    except zipfile.BadZipFile:
        raise ValueError("❌ Failed to extract ZIP file from GitHub. The repo might be empty or private.")

    inner_folder = os.path.join(dest_folder, f"{repo_name}-{default_branch}")
    if not os.path.isdir(inner_folder):
        raise ValueError("❌ Failed to find extracted repo folder inside ZIP.")

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
