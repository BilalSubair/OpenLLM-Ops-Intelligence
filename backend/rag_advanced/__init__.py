"""
Advanced RAG Enhancements Module.
Includes metadata filtering, retrieval accuracy evaluation (Hit Rate, MRR),
and hallucination / faithfulness assessment.
"""

from rag_advanced.metadata_filtering import search_with_metadata_filter
from rag_advanced.evaluation import evaluate_retrieval_accuracy, RetrievalEvaluator
from rag_advanced.hallucination import assess_hallucination, HallucinationDetector
