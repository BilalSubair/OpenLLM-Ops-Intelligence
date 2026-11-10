
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from pathlib import Path
import httpx
import shutil
import uuid
import time

from rag.hybrid_search import hybrid_search
from rag.reranker import rerank_documents
from rag.generator import generate_answer
from rag.upload import process_uploaded_file


# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

app = FastAPI(
    title="OpenLLM Ops Intelligence",
    description="AI-powered LLM monitoring and troubleshooting platform",
    version="1.0.0"
)

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen3.5:2b"

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class ChatRequest(BaseModel):
    question: str


class RAGRequest(BaseModel):
    question: str


# --------------------------------------------------
# HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "project": "OpenLLM Ops Intelligence",
        "status": "running"
    }


# --------------------------------------------------
# HEALTH CHECK ENDPOINT
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "backend"
    }


# --------------------------------------------------
# NORMAL LLM CHAT ENDPOINT
# --------------------------------------------------

@app.post("/api/llm/chat")
async def chat(request: ChatRequest):

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": request.question
            }
        ],
        "stream": False
    }

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:

            response = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload
            )

            response.raise_for_status()
            result = response.json()

            return {
                "question": request.question,
                "model": MODEL_NAME,
                "answer": result["message"]["content"]
            }

    except httpx.HTTPStatusError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned an error: {error.response.text}"
        )

    except httpx.RequestError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Ollama. Check whether Ollama is running."
        )


# --------------------------------------------------
# HYBRID RAG + RERANKING ENDPOINT
# --------------------------------------------------


@app.post("/api/rag/ask")
async def ask_rag(request: RAGRequest):

    # Start total request timer
    total_start = time.perf_counter()

    try:
        # ------------------------------------------
        # STEP 1: HYBRID SEARCH
        # ------------------------------------------

        retrieval_start = time.perf_counter()

        hybrid_results = hybrid_search(
            request.question,
            results_count=5
        )

        retrieval_end = time.perf_counter()

        if not hybrid_results:
            return {
                "question": request.question,
                "answer": "No relevant documentation found.",
                "sources": [],
                "performance": {
                    "retrieval_ms": round(
                        (retrieval_end - retrieval_start) * 1000, 2
                    )
                }
            }

        # ------------------------------------------
        # STEP 2: RERANKING
        # ------------------------------------------

        reranking_start = time.perf_counter()

        reranked_results = rerank_documents(
            question=request.question,
            documents=hybrid_results,
            top_k=3
        )

        reranking_end = time.perf_counter()

        if not reranked_results:
            return {
                "question": request.question,
                "answer": "No relevant documentation found.",
                "sources": []
            }

        # ------------------------------------------
        # STEP 3: PREPARE CONTEXT
        # ------------------------------------------

        documents = [
            result["document"]
            for result in reranked_results
        ]

        context = "\n\n".join(documents)

        # ------------------------------------------
        # STEP 4: LLM GENERATION
        # ------------------------------------------

        generation_start = time.perf_counter()

        answer = generate_answer(
            request.question,
            context
        )

        generation_end = time.perf_counter()

        # ------------------------------------------
        # STEP 5: TOTAL PERFORMANCE
        # ------------------------------------------

        total_end = time.perf_counter()

        retrieval_ms = (retrieval_end - retrieval_start) * 1000
        reranking_ms = (reranking_end - reranking_start) * 1000
        generation_ms = (generation_end - generation_start) * 1000
        total_ms = (total_end - total_start) * 1000

        return {
            "question": request.question,
            "answer": answer,
            "model": MODEL_NAME,
            "retrieval_method": "Hybrid Search + BGE Reranking",
            "sources": [
                {
                    "id": result["id"],
                    "rerank_score": result["rerank_score"]
                }
                for result in reranked_results
            ],
            "performance": {
                "retrieval_ms": round(retrieval_ms, 2),
                "reranking_ms": round(reranking_ms, 2),
                "generation_ms": round(generation_ms, 2),
                "total_ms": round(total_ms, 2)
            }
        }

    except httpx.RequestError:
        raise HTTPException(
            status_code=503,
            detail="Unable to connect to Ollama."
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# --------------------------------------------------
# DOCUMENT UPLOAD ENDPOINT
# --------------------------------------------------

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):

    allowed_extensions = {
        ".pdf",
        ".txt",
        ".md",
        ".markdown"
    }

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required"
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format"
        )

    unique_name = f"{uuid.uuid4().hex}{extension}"
    file_path = UPLOAD_DIR / unique_name

    try:

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = process_uploaded_file(str(file_path))

        return result

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )

    finally:
        await file.close()

