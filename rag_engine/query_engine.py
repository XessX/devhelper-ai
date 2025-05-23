import os
from dotenv import load_dotenv

load_dotenv()

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def is_render():
    """
    Detect if running in Render (cloud) environment.
    """
    return (
        os.getenv("RENDER", "false").lower() == "true"
        or "onrender.com" in os.environ.get("RENDER_EXTERNAL_HOSTNAME", "")
    )

def load_llm(engine: str = None):
    """
    Load the appropriate LLM engine (OpenAI or Ollama), bulletproof against cloud/Ollama conflicts.
    """
    # 🚨 Force OpenAI in Render/cloud mode (never allow Ollama).
    if is_render():
        engine = "openai"

    # Default selection if not specified: use .env locally, otherwise OpenAI
    if engine is None:
        use_env = os.getenv("USE_OLLAMA", "0") == "1"
        engine = "ollama" if use_env else "openai"

    # 🚨 Double-check: absolutely never allow Ollama in Render/cloud
    if is_render() and engine == "ollama":
        raise RuntimeError("Ollama is not supported in online/cloud mode. Please select OpenAI.")

    # --- Import ONLY what you need ---
    if engine == "ollama":
        # Only import Ollama client locally, never in cloud
        try:
            from langchain_ollama import OllamaLLM
        except ImportError:
            raise ImportError("langchain_ollama is not installed. Please install it to use Ollama locally.")
        return OllamaLLM(model="llama3")
    else:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("langchain_openai is not installed. Please install it to use OpenAI.")
        return ChatOpenAI(
            model="gpt-4",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

def get_llm_chain(vectordb, engine: str = None):
    """
    Returns a chain that handles retrieval-augmented QA and project summary, using the right LLM.
    """
    llm = load_llm(engine)
    retriever = vectordb.as_retriever()

    class SmartChain:
        def __init__(self):
            self.qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        def invoke(self, inputs: dict):
            query = inputs.get("query") or inputs.get("question")
            if not query:
                return "❌ No question provided."

            # Triggers for project summary
            summary_triggers = [
                "readme", "project", "repo", "what this repo", "what is this repo",
                "what this codebase", "what does this repo do", "describe this repository",
                "summary of this repo"
            ]
            if any(k in query.lower() for k in summary_triggers):
                docs = retriever.invoke("project overview")
                if not docs:
                    return "No files found to summarize."
                docs = docs[:12]  # Limit for context size
                file_list = "\n".join(doc.metadata.get("source", "") for doc in docs)
                code_snippets = "\n\n".join(doc.page_content[:1500] for doc in docs)
                prompt = PromptTemplate.from_template("""
You are an expert software assistant.

Below are the main files and key code/documentation snippets from a software project:

Files:
{file_list}

File Contents:
{code_snippets}

User Question:
{question}

Based on the above files and their content, give a clear, practical summary of what this repository or codebase does, what kind of project it is, its main components, and what its main files or code blocks implement. If possible, infer the purpose from the filenames and code. Do not guess; use only the provided code and files.
""")
                prompt_text = prompt.format_prompt(
                    file_list=file_list.strip(),
                    code_snippets=code_snippets.strip(),
                    question=query.strip()
                ).to_string()
                return llm.invoke(prompt_text)
            # Default: use vanilla QA
            return self.qa_chain.invoke({"query": query})

    return SmartChain()
