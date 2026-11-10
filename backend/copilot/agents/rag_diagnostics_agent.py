"""
RAG & Knowledge Diagnostics Specialist Agent.
Diagnoses hybrid search retrieval accuracy, context starvation,
embedding dimension mismatch, and Ollama generator connectivity.
"""

from typing import Dict, Any, Optional
from copilot.agents.base_agent import BaseSpecialistAgent
from copilot.tools.log_analyzer import LogAnalyzer

try:
    from rag.vector_store import collection
except Exception:
    collection = None


class RAGDiagnosticsAgent(BaseSpecialistAgent):
    def __init__(self):
        super().__init__(
            name="RAG-Diagnostics-Agent",
            specialty="RAG Pipeline, Hybrid Search (Dense + BM25), Cross-Encoder Reranking, and Context Health",
            system_prompt="You are a RAG systems architect specializing in vector databases, lexical retrieval, and context optimization."
        )
        self.log_analyzer = LogAnalyzer()

    async def diagnose(self, problem_description: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.clear_steps()
        context = context or {}

        # Step 1: Inspect Vector DB Health
        self.log_step(
            thought="Inspecting ChromaDB collection status to verify document counts and index accessibility.",
            action="collection.count"
        )
        doc_count = 0
        if collection:
            try:
                doc_count = collection.count()
                self.steps[-1].observation = {"chroma_collection": "infrastructure_knowledge", "total_documents": doc_count}
            except Exception as e:
                self.steps[-1].observation = {"error": str(e)}
        else:
            self.steps[-1].observation = {"status": "ChromaDB offline or module uninitialized"}

        # Step 2: Log analysis
        log_data = context.get("logs") or problem_description
        self.log_step(
            thought="Checking logs for embedding generation timeouts or ChromaDB ingestion failures.",
            action="LogAnalyzer.analyze_log"
        )
        log_findings = self.log_analyzer.analyze_log(log_data)
        self.steps[-1].observation = {
            "issues_count": log_findings["issues_found"],
            "primary": log_findings.get("primary_diagnosis", {}).get("error_type") if log_findings.get("primary_diagnosis") else None
        }

        # Step 3: Diagnostic reasoning
        primary = log_findings.get("primary_diagnosis")
        if primary and primary.get("error_type") == "OLLAMA_CONNECTION_REFUSED":
            root_cause = "Embedding & LLM Service Down: Backend cannot connect to Ollama on port 11434."
            severity = "CRITICAL"
            remediation = [
                "1. Start Ollama: 'ollama serve' or check container status.",
                "2. Ensure embedding model 'nomic-embed-text' and LLM 'qwen3.5:2b' are pulled ('ollama pull nomic-embed-text').",
                "3. Verify backend OLLAMA_URL points to the correct network host."
            ]
        elif collection is not None and doc_count == 0:
            root_cause = "Empty Vector Database: The ChromaDB collection contains 0 documents. RAG queries cannot retrieve relevant context."
            severity = "HIGH"
            remediation = [
                "1. Ingest infrastructure runbooks or documentation using /api/documents/upload.",
                "2. Run 'python test_pipeline.py' to populate initial troubleshooting documents."
            ]
        elif "empty context" in problem_description.lower() or "no relevant documentation" in problem_description.lower():
            root_cause = "Retrieval Starvation / Vocabulary Mismatch: Neither dense vector similarity nor BM25 keyword matching crossed relevance thresholds."
            severity = "MEDIUM"
            remediation = [
                "1. Increase hybrid search results_count from 3 to 10 before reranking.",
                "2. Adjust chunk_size (current: 500) and chunk_overlap (current: 100) to capture broader context.",
                "3. Check if query contains typos or technical acronyms not present in indexed documents."
            ]
        else:
            root_cause = "RAG pipeline operational issue or low context precision."
            severity = "LOW"
            remediation = [
                "Evaluate retrieval accuracy using the /api/rag/evaluate endpoint.",
                "Verify Cross-Encoder BGE reranker threshold settings.",
                "Check for hallucination risk with /api/rag/check-hallucination."
            ]

        return {
            "agent": self.name,
            "specialty": self.specialty,
            "root_cause_analysis": root_cause,
            "severity": severity,
            "vector_store_health": {
                "collection_name": "infrastructure_knowledge",
                "total_indexed_chunks": doc_count
            },
            "detected_log_issues": log_findings.get("all_detected_issues", []),
            "remediation_steps": remediation,
            "reasoning_steps": [s.to_dict() for s in self.steps]
        }
