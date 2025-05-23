# 🧠 DevHelper AI

**Chat with your codebase using LLMs!**  
Analyze local projects, GitHub repos, or entire websites with **OpenAI** (online/cloud) or **Ollama** (local/dev, optional).

🌐 **Live Demo:** [https://devhelper-ai.onrender.com](https://devhelper-ai.onrender.com)

---

## 🚀 Features

- 🔍 **RAG** (Retrieval-Augmented Generation) + Chunking + Vector Search
- 📂 Local folders or Docker-mounted volumes
- 🌐 GitHub repo & Website content support
- 🧠 Supports **OpenAI** (cloud, always) and **Ollama** (local, optional)
- 🧱 ChromaDB persistence (vector store)
- 💬 Streamlit-based chat UI
- 📥 Export chat history (JSON)

---

## ⚙️ Quick Start

### 🏗️ **Local/Dev Mode** (OpenAI or Ollama)

```bash
git clone https://github.com/XessX/devhelper-ai
cd devhelper-ai
cp .env.example .env
# Add your OpenAI API key to .env
pip install -r requirements.txt
streamlit run app.py
Optionally, for Docker/Ollama (local RAG):

# PowerShell
.\run-devhelper.ps1

# OR, if on bash:
docker build -t devhelper-ai .
docker run -it -p 8501:8501 -v "$(pwd):/mounted" --env-file .env devhelper-ai
☁️ Cloud/Render Deployment (OpenAI-only)
Push to GitHub:
https://github.com/XessX/devhelper-ai

One-click deploy using render.yaml

Set your Render environment variable:

OPENAI_API_KEY=your-openai-key
⚠️ Note
On Render (and any online cloud host), only OpenAI is supported.

Ollama is only available for local Docker/dev use.

The app will auto-select OpenAI on Render/cloud; no user/accidental Ollama usage.

💡 Use Cases
Chat with unfamiliar codebases

Understand legacy repositories

Explore or debug GitHub projects interactively

Extract architecture or README details

Scrape & summarize documentation sites

📁 Project Structure

devhelper-ai/
├── app.py                  # Streamlit frontend
├── rag_engine/             # Code loaders, chunkers, RAG logic
│   ├── chunker.py
│   ├── loader.py
│   ├── query_engine.py
│   ├── utils.py
│   └── vector_store.py
├── Dockerfile              # Docker container (local, dev)
├── run-devhelper.ps1       # PowerShell launcher for Docker
├── requirements.txt        # Python dependencies
├── .env.example            # Env variable template
├── render.yaml             # Render deploy spec
└── README.md               # You’re reading it!
👨‍💻 Author
DevHelper AI by Al Jubair Hossain

GitHub: @XessX

LinkedIn: al-jubair-hossain

🙌 Acknowledgments
LangChain

Ollama

ChromaDB

Streamlit

🛡️ LLM Engine Policy
OpenAI: Always available, required for online/cloud (Render, etc.)

Ollama: Only available in local Docker/dev environments.

On Render/cloud, Ollama is disabled & cannot be selected.

The app will auto-select OpenAI on Render; no user/accidental Ollama usage.

📥 Export/Import
Download chat logs as .json for research or reuse.

Enjoy your AI-powered codebase assistant!