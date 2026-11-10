"""
SearXNG Client Module.
Provides privacy-preserving, self-hosted web search augmentation for the Copilot.
Queries self-hosted SearXNG meta-search engine with graceful local knowledge fallback.
"""

import os
from typing import Dict, Any, List, Optional

try:
    import httpx
except ImportError:
    httpx = None


class SearXNGClient:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or os.getenv("SEARXNG_URL", "http://localhost:8082")).rstrip("/")

    async def search(self, query: str, categories: str = "it", max_results: int = 5) -> Dict[str, Any]:
        """
        Query SearXNG JSON API endpoint (/search?format=json).
        """
        if httpx:
            params = {
                "q": query,
                "format": "json",
                "categories": categories,
                "language": "en"
            }

            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    res = await client.get(f"{self.base_url}/search", params=params)
                    if res.status_code == 200:
                        data = res.json()
                        results = []
                        for item in data.get("results", [])[:max_results]:
                            results.append({
                                "title": item.get("title", ""),
                                "url": item.get("url", ""),
                                "snippet": item.get("content", ""),
                                "engine": item.get("engine", "")
                            })
                        return {
                            "status": "online",
                            "provider": "SearXNG",
                            "query": query,
                            "results_count": len(results),
                            "results": results
                        }
            except Exception:
                pass

        # Resilient offline knowledge base fallback for common AI Ops errors
        return self._offline_search_fallback(query)

    def _offline_search_fallback(self, query: str) -> Dict[str, Any]:
        """
        Fallback search results from curated offline infrastructure knowledge base.
        """
        q_lower = query.lower()
        curated_db = [
            {
                "keywords": ["cuda", "out of memory", "oom", "vram"],
                "title": "NVIDIA PyTorch CUDA Memory Management & OOM Resolution",
                "url": "https://pytorch.org/docs/stable/notes/cuda.html#memory-management",
                "snippet": "Use torch.cuda.empty_cache(), set max_split_size_mb, enable gradient checkpointing or lower batch size.",
                "engine": "Curated Docs"
            },
            {
                "keywords": ["xid", "gpu fell off bus", "nvidia-smi"],
                "title": "NVIDIA Driver Xid Error Reference & Troubleshooting",
                "url": "https://docs.nvidia.com/deploy/xid-errors/index.html",
                "snippet": "Xid messages indicate hardware, PCIe bus, or memory errors detected by the NVIDIA graphics driver.",
                "engine": "NVIDIA Enterprise"
            },
            {
                "keywords": ["vllm", "shm", "bus error"],
                "title": "vLLM Production Deployment and Shared Memory Configuration",
                "url": "https://docs.vllm.ai/en/latest/serving/deploying_with_docker.html",
                "snippet": "Ensure docker is started with --ipc=host or --shm-size=10.24gb to prevent DataLoader and NCCL crashes.",
                "engine": "vLLM Documentation"
            },
            {
                "keywords": ["kubernetes", "gpu", "operator", "insufficient"],
                "title": "NVIDIA GPU Operator on Kubernetes Installation Guide",
                "url": "https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/overview.html",
                "snippet": "NVIDIA GPU Operator manages NVIDIA drivers, Container Toolkit, and Kubernetes device plugin daemons.",
                "engine": "Kubernetes Docs"
            }
        ]

        matched = []
        for doc in curated_db:
            if any(k in q_lower for k in doc["keywords"]):
                matched.append(doc)

        return {
            "status": "offline_fallback",
            "provider": "Local SearXNG Cache",
            "query": query,
            "results_count": len(matched),
            "results": matched
        }
