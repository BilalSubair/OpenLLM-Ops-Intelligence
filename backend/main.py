import time
import shutil
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Response
from pydantic import BaseModel
import httpx

# Existing RAG imports
from rag.hybrid_search import hybrid_search
from rag.reranker import rerank_documents
from rag.generator import generate_answer
from rag.upload import process_uploaded_file

# Monitoring & Observability imports
from monitoring.gpu_monitor import get_gpu_metrics
from monitoring.system_monitor import get_system_metrics
from monitoring.inference_metrics import inference_metrics_collector
from monitoring.prometheus_exporter import generate_prometheus_metrics

# Benchmarking & Quantization imports
from benchmarking.quantization_analyzer import analyze_quantization_tradeoffs
from benchmarking.benchmark_runner import BenchmarkRunner

# Copilot & Multi-Agent Architecture imports
from copilot.orchestrator import copilot_orchestrator
from copilot.tools.log_analyzer import LogAnalyzer
from copilot.tools.config_inspector import ConfigInspector
from copilot.tools.searxng_client import SearXNGClient

# Model Registry, Health & Fallbacks imports
from registry.model_registry import model_registry
from registry.health_monitor import health_monitor
from registry.fallback_router import fallback_router
from registry.audit_logger import audit_logger

# Advanced RAG Enhancements imports
from rag_advanced.metadata_filtering import search_with_metadata_filter
from rag_advanced.evaluation import evaluate_retrieval_accuracy
from rag_advanced.hallucination import assess_hallucination


# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

app = FastAPI(
    title="OpenLLM Ops Intelligence",
    description="AI-powered LLM infrastructure monitoring, management and troubleshooting platform",
    version="2.0.0"
)

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen3.5:2b"

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

benchmark_runner = BenchmarkRunner()
log_analyzer = LogAnalyzer()
config_inspector = ConfigInspector()
searxng_client = SearXNGClient()


# --------------------------------------------------
# REQUEST & RESPONSE MODELS
# --------------------------------------------------

class ChatRequest(BaseModel):
    question: str
    model: Optional[str] = None


class RAGRequest(BaseModel):
    question: str
    category: Optional[str] = None
    results_count: int = 5
    top_k_rerank: int = 3


class BenchmarkRequest(BaseModel):
    engines: List[str] = ["ollama"]
    prompt: Optional[str] = None
    iterations: int = 2
    max_tokens: int = 100


class CopilotDiagnoseRequest(BaseModel):
    problem_description: str
    logs: Optional[str] = None
    config_manifest: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class LogAnalysisRequest(BaseModel):
    logs: str


class ConfigInspectRequest(BaseModel):
    config_type: str = "kubernetes"  # "kubernetes" or "docker"
    content: str


class FallbackInferenceRequest(BaseModel):
    prompt: str
    preferred_engine: Optional[str] = "ollama"
    model_id: Optional[str] = None
    max_tokens: int = 128


class RetrievalEvalRequest(BaseModel):
    query: str
    expected_doc_ids: List[str]
    k: int = 3


class HallucinationCheckRequest(BaseModel):
    context: str
    generated_answer: str


# --------------------------------------------------
# HOME & HEALTH ENDPOINTS
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "project": "OpenLLM Ops Intelligence",
        "version": "2.0.0",
        "status": "running",
        "description": "AI-Powered LLM Infrastructure Monitoring, Management and Troubleshooting Platform"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "backend",
        "version": "2.0.0"
    }


# --------------------------------------------------
# NORMAL LLM CHAT ENDPOINT
# --------------------------------------------------

@app.post("/api/llm/chat")
async def chat(request: ChatRequest):
    target_model = request.model or MODEL_NAME
    payload = {
        "model": target_model,
        "messages": [
            {
                "role": "user",
                "content": request.question
            }
        ],
        "stream": False
    }

    start_time = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            latency_ms = (time.perf_counter() - start_time) * 1000
            answer = result["message"]["content"]
            comp_tokens = len(answer.split())

            # Record metrics & audit
            inference_metrics_collector.record_request(
                model=target_model,
                engine="ollama",
                latency_ms=latency_ms,
                prompt_tokens=len(request.question.split()),
                completion_tokens=comp_tokens,
                success=True
            )
            audit_logger.log_event(
                event_type="LLM_CHAT",
                model=target_model,
                engine="ollama",
                latency_ms=latency_ms,
                status="SUCCESS"
            )

            return {
                "question": request.question,
                "model": target_model,
                "answer": answer,
                "latency_ms": round(latency_ms, 2)
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
    total_start = time.perf_counter()

    try:
        # STEP 1: HYBRID SEARCH
        retrieval_start = time.perf_counter()
        hybrid_results = hybrid_search(
            request.question,
            results_count=request.results_count
        )
        retrieval_end = time.perf_counter()

        if not hybrid_results:
            return {
                "question": request.question,
                "answer": "No relevant documentation found.",
                "sources": [],
                "performance": {
                    "retrieval_ms": round((retrieval_end - retrieval_start) * 1000, 2)
                }
            }

        # STEP 2: RERANKING
        reranking_start = time.perf_counter()
        reranked_results = rerank_documents(
            question=request.question,
            documents=hybrid_results,
            top_k=request.top_k_rerank
        )
        reranking_end = time.perf_counter()

        if not reranked_results:
            return {
                "question": request.question,
                "answer": "No relevant documentation found after reranking.",
                "sources": []
            }

        # STEP 3: PREPARE CONTEXT
        documents = [result["document"] for result in reranked_results]
        context = "\n\n".join(documents)

        # STEP 4: LLM GENERATION
        generation_start = time.perf_counter()
        answer = generate_answer(request.question, context)
        generation_end = time.perf_counter()

        # STEP 5: TOTAL PERFORMANCE
        total_end = time.perf_counter()

        retrieval_ms = (retrieval_end - retrieval_start) * 1000
        reranking_ms = (reranking_end - reranking_start) * 1000
        generation_ms = (generation_end - generation_start) * 1000
        total_ms = (total_end - total_start) * 1000

        # Record metrics & audit
        inference_metrics_collector.record_request(
            model=MODEL_NAME,
            engine="ollama",
            latency_ms=total_ms,
            prompt_tokens=len(context.split()) + len(request.question.split()),
            completion_tokens=len(answer.split()),
            success=True
        )
        audit_logger.log_event(
            event_type="RAG_ASK",
            model=MODEL_NAME,
            engine="ollama",
            latency_ms=total_ms,
            status="SUCCESS",
            metadata={"top_source": reranked_results[0]["id"]}
        )

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
    allowed_extensions = {".pdf", ".txt", ".md", ".markdown"}

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    extension = Path(file.filename).suffix.lower()
    if extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    unique_name = f"{uuid.uuid4().hex}{extension}"
    file_path = UPLOAD_DIR / unique_name

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = process_uploaded_file(str(file_path))
        return result
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
    finally:
        await file.close()


# --------------------------------------------------
# MONITORING & OBSERVABILITY ENDPOINTS
# --------------------------------------------------

@app.get("/api/monitoring/gpu")
def get_gpu_telemetry():
    """Real-time GPU compute utilization, VRAM allocation, temperature and hardware devices."""
    return get_gpu_metrics()


@app.get("/api/monitoring/system")
def get_system_telemetry():
    """Host CPU load, system RAM, swap, and disk capacity metrics."""
    return get_system_metrics()


@app.get("/api/monitoring/inference")
def get_inference_telemetry():
    """Inference telemetry: latency percentiles (p50/p95/p99), throughput, TTFT, and tokens/sec."""
    return inference_metrics_collector.get_summary()


@app.get("/metrics")
def get_prometheus_metrics():
    """Prometheus exposition format for scraper & Grafana dashboard."""
    metrics_text = generate_prometheus_metrics()
    return Response(content=metrics_text, media_type="text/plain; version=0.0.4")


# --------------------------------------------------
# BENCHMARKING & QUANTIZATION ENDPOINTS
# --------------------------------------------------

@app.post("/api/benchmark/run")
async def run_benchmark(request: BenchmarkRequest):
    """Executes latency, throughput, and token generation benchmarks across inference engines."""
    prompt = request.prompt or "Explain GPU memory paging mechanisms in large language model inference."
    results = await benchmark_runner.compare_engines(
        engines=request.engines,
        prompt=prompt,
        iterations=request.iterations,
        max_tokens=request.max_tokens
    )
    return results


@app.get("/api/benchmark/quantization")
def get_quantization_analysis(
    params_b: float = Query(8.0, description="Model parameters in billions (e.g. 7, 8, 14, 70)"),
    context_length: int = Query(4096, description="Context window length"),
    gpu_type: str = Query("NVIDIA_RTX_4090", description="Reference GPU target")
):
    """GGUF model quantization trade-off analysis (Q4_K_M vs Q8_0, VRAM savings, and perplexity)."""
    return analyze_quantization_tradeoffs(
        params_billion=params_b,
        context_length=context_length,
        gpu_type=gpu_type
    )


# --------------------------------------------------
# AI INFRASTRUCTURE TROUBLESHOOTING COPILOT
# --------------------------------------------------

@app.post("/api/copilot/diagnose")
async def copilot_diagnose(request: CopilotDiagnoseRequest):
    """
    Multi-Agent Copilot Troubleshooting endpoint.
    Coordinates GPU, Kubernetes, Docker, and RAG specialists for root-cause analysis and remediation.
    """
    context = request.context or {}
    if request.logs:
        context["logs"] = request.logs
    if request.config_manifest:
        context["manifest"] = request.config_manifest

    diagnosis = await copilot_orchestrator.troubleshoot(
        problem_description=request.problem_description,
        context=context
    )
    return diagnosis


@app.post("/api/copilot/analyze-log")
def analyze_logs(request: LogAnalysisRequest):
    """Parses infrastructure logs for CUDA OOM, driver crashes, K8s OOMKilled, and Docker errors."""
    return log_analyzer.analyze_log(request.logs)


@app.post("/api/copilot/inspect-config")
def inspect_configuration(request: ConfigInspectRequest):
    """Lints Kubernetes manifests or Docker configs for GPU resource allocation and /dev/shm issues."""
    if request.config_type.lower() == "kubernetes":
        return config_inspector.inspect_kubernetes_manifest(request.content)
    else:
        return config_inspector.inspect_docker_config(request.content)


@app.get("/api/copilot/search")
async def search_knowledge(
    q: str = Query(..., description="Query technical documentation or error string"),
    max_results: int = Query(5, description="Max results to retrieve")
):
    """Queries SearXNG self-hosted search engine with offline knowledge fallback."""
    return await searxng_client.search(query=q, max_results=max_results)


# --------------------------------------------------
# MODEL REGISTRY, HEALTH & RESILIENT ROUTING
# --------------------------------------------------

@app.get("/api/registry/models")
def list_registered_models():
    """Lists registered models, quantization profiles, and recommended hardware tiers."""
    return {"models": model_registry.list_models()}


@app.get("/api/registry/health")
async def check_engine_health():
    """Active health and latency probe across all inference engines."""
    return await health_monitor.probe_all(force_refresh=True)


@app.post("/api/registry/route")
async def route_resilient_inference(request: FallbackInferenceRequest):
    """
    Executes resilient inference with automatic failover (e.g. vLLM -> Ollama -> llama.cpp).
    """
    result = await fallback_router.route_and_generate(
        prompt=request.prompt,
        preferred_engine=request.preferred_engine,
        model_id=request.model_id,
        max_tokens=request.max_tokens
    )
    if "error" in result and not result.get("text"):
        raise HTTPException(status_code=502, detail=result)
    return result


@app.get("/api/audit/logs")
def get_audit_trail(limit: int = Query(50, description="Number of recent logs to fetch")):
    """Returns recent structured audit events for queries and failovers."""
    return {"audit_events": audit_logger.get_recent_logs(limit=limit)}


# --------------------------------------------------
# ADVANCED RAG ENHANCEMENTS
# --------------------------------------------------

@app.post("/api/rag/search-filtered")
def search_filtered_rag(
    question: str,
    category: Optional[str] = None,
    source: Optional[str] = None,
    results_count: int = 5
):
    """Filtered vector search in ChromaDB by metadata category or source."""
    return search_with_metadata_filter(
        question=question,
        category=category,
        source=source,
        results_count=results_count
    )


@app.post("/api/rag/evaluate")
def evaluate_retrieval(request: RetrievalEvalRequest):
    """Evaluates retrieval accuracy metrics: Hit Rate@K, MRR, Context Relevance."""
    # Retrieve top chunks for evaluation
    hybrid_results = hybrid_search(request.query, results_count=max(5, request.k))
    retrieved_ids = [r["id"] for r in hybrid_results]
    retrieved_contexts = [r["document"] for r in hybrid_results]

    return evaluate_retrieval_accuracy(
        query=request.query,
        retrieved_ids=retrieved_ids,
        retrieved_contexts=retrieved_contexts,
        expected_doc_ids=request.expected_doc_ids,
        k=request.k
    )


@app.post("/api/rag/check-hallucination")
def check_hallucination(request: HallucinationCheckRequest):
    """Calculates faithfulness and flags unsupported claims in generated text."""
    return assess_hallucination(
        context=request.context,
        generated_answer=request.generated_answer
    )
