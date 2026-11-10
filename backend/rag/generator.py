import httpx

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen3.5:2b"


def generate_answer(question: str, context: str):

    prompt = f"""
You are an AI infrastructure troubleshooting assistant.

Answer the user's question using the provided context.

If the context does not contain enough information,
clearly say that.

Context:
{context}

Question:
{question}

Provide a clear, technically accurate answer.
"""

    response = httpx.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        },
        timeout=180.0
    )

    response.raise_for_status()

    return response.json()["response"]
