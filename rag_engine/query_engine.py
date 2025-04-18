import os
from dotenv import load_dotenv

# Load .env vars (OPENAI_API_KEY, USE_OLLAMA, DOCKERIZED)
load_dotenv()

# Docker → Host Ollama patch
if os.getenv("DOCKERIZED", "false").lower() == "true":
    os.environ["OLLAMA_BASE_URL"] = "http://host.docker.internal:11434"

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def load_llm(engine: str = None):
    """
    Load LLM from OpenAI or Ollama, optionally based on argument or .env
    """
    use_env = os.getenv("USE_OLLAMA", "0") == "1"
    use_ollama = engine == "ollama" or (engine is None and use_env)

    if use_ollama:
        from langchain_ollama import OllamaLLM
        return OllamaLLM(model="llama3")
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )


def get_llm_chain(vectordb, engine: str = None):
    retriever = vectordb.as_retriever()
    llm = load_llm(engine)

    class SmartChain:
        def __init__(self):
            self.qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        def invoke(self, inputs: dict):
            query = inputs.get("query") or inputs.get("question")
            if not query:
                return "❌ No question provided."

            # Handle README special case
            if any(k in query.lower() for k in ["readme", "project", "repo"]):
                results = retriever.invoke(query)
                readme_docs = [
                    doc for doc in results
                    if "readme.md" in doc.metadata.get("source", "").lower()
                ]
                if not readme_docs:
                    return "⚠️ README not found."

                content = "\n".join(doc.page_content for doc in readme_docs)
                prompt = PromptTemplate.from_template("""
You are a helpful assistant that understands software projects.

README:
{readme}

Question:
{question}

Answer:
""")
                prompt_text = prompt.format_prompt(
                    readme=content.strip(),
                    question=query.strip()
                ).to_string()
                return llm.invoke(prompt_text)

            return self.qa_chain.invoke({"query": query})

    return SmartChain()
