"""
Inference Engine Connector Suite.
Provides unified clients for Ollama, vLLM, llama.cpp, TGI, and LMDeploy.
"""

import time
from typing import Dict, Any, Optional

try:
    import httpx
except ImportError:
    httpx = None


class BaseInferenceEngine:
    def __init__(self, name: str, base_url: str, default_model: str):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model

    async def check_health(self) -> Dict[str, Any]:
        raise NotImplementedError

    async def generate(self, prompt: str, model: Optional[str] = None, max_tokens: int = 128) -> Dict[str, Any]:
        raise NotImplementedError


class OllamaEngine(BaseInferenceEngine):
    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = "qwen3.5:2b"):
        super().__init__("Ollama", base_url, default_model)

    async def check_health(self) -> Dict[str, Any]:
        if not httpx:
            return {"status": "offline", "engine": self.name, "error": "httpx package not installed"}
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                if res.status_code == 200:
                    models = [m["name"] for m in res.json().get("models", [])]
                    return {"status": "online", "engine": self.name, "latency_ms": latency_ms, "available_models": models}
        except Exception as e:
            return {"status": "offline", "engine": self.name, "error": str(e)}
        return {"status": "offline", "engine": self.name}

    async def generate(self, prompt: str, model: Optional[str] = None, max_tokens: int = 128) -> Dict[str, Any]:
        if not httpx:
            raise RuntimeError("httpx is required for live Ollama HTTP generation")
        target_model = model or self.default_model
        payload = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens}
        }
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=120.0) as client:
            res = await client.post(f"{self.base_url}/api/generate", json=payload)
            res.raise_for_status()
            elapsed_sec = time.perf_counter() - start
            data = res.json()

            prompt_eval_count = data.get("prompt_eval_count", len(prompt.split()))
            eval_count = data.get("eval_count", len(data.get("response", "").split()))
            tps = round(eval_count / max(elapsed_sec, 0.001), 2)
            ttft_ms = round((data.get("prompt_eval_duration", 0) / 1e6), 2) if "prompt_eval_duration" in data else 0.0

            return {
                "engine": self.name,
                "model": target_model,
                "text": data.get("response", ""),
                "prompt_tokens": prompt_eval_count,
                "completion_tokens": eval_count,
                "total_time_ms": round(elapsed_sec * 1000, 2),
                "ttft_ms": ttft_ms,
                "tokens_per_second": tps
            }


class VLLMEngine(BaseInferenceEngine):
    def __init__(self, base_url: str = "http://localhost:8000", default_model: str = "meta-llama/Llama-3.2-3B-Instruct"):
        super().__init__("vLLM", base_url, default_model)

    async def check_health(self) -> Dict[str, Any]:
        if not httpx:
            return {"status": "offline", "engine": self.name, "error": "httpx package not installed"}
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{self.base_url}/health")
                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                if res.status_code == 200:
                    return {"status": "online", "engine": self.name, "latency_ms": latency_ms}
        except Exception as e:
            return {"status": "offline", "engine": self.name, "error": str(e)}
        return {"status": "offline", "engine": self.name}

    async def generate(self, prompt: str, model: Optional[str] = None, max_tokens: int = 128) -> Dict[str, Any]:
        if not httpx:
            raise RuntimeError("httpx is required for live vLLM HTTP generation")
        target_model = model or self.default_model
        payload = {
            "model": target_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.2
        }
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=120.0) as client:
            res = await client.post(f"{self.base_url}/v1/chat/completions", json=payload)
            res.raise_for_status()
            elapsed_sec = time.perf_counter() - start
            data = res.json()
            usage = data.get("usage", {})
            prompt_toks = usage.get("prompt_tokens", len(prompt.split()))
            comp_toks = usage.get("completion_tokens", max_tokens)
            content = data["choices"][0]["message"]["content"]

            return {
                "engine": self.name,
                "model": target_model,
                "text": content,
                "prompt_tokens": prompt_toks,
                "completion_tokens": comp_toks,
                "total_time_ms": round(elapsed_sec * 1000, 2),
                "ttft_ms": round(elapsed_sec * 100, 2),
                "tokens_per_second": round(comp_toks / max(elapsed_sec, 0.001), 2)
            }


class LlamaCppEngine(BaseInferenceEngine):
    def __init__(self, base_url: str = "http://localhost:8080", default_model: str = "gguf-model"):
        super().__init__("llama.cpp", base_url, default_model)

    async def check_health(self) -> Dict[str, Any]:
        if not httpx:
            return {"status": "offline", "engine": self.name, "error": "httpx package not installed"}
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{self.base_url}/health")
                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                return {"status": "online" if res.status_code == 200 else "offline", "engine": self.name, "latency_ms": latency_ms}
        except Exception as e:
            return {"status": "offline", "engine": self.name, "error": str(e)}

    async def generate(self, prompt: str, model: Optional[str] = None, max_tokens: int = 128) -> Dict[str, Any]:
        if not httpx:
            raise RuntimeError("httpx is required for live llama.cpp HTTP generation")
        payload = {"prompt": prompt, "n_predict": max_tokens, "stream": False}
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=120.0) as client:
            res = await client.post(f"{self.base_url}/completion", json=payload)
            res.raise_for_status()
            elapsed_sec = time.perf_counter() - start
            data = res.json()
            text = data.get("content", "")
            comp_toks = data.get("tokens_predicted", len(text.split()))

            return {
                "engine": self.name,
                "model": model or self.default_model,
                "text": text,
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": comp_toks,
                "total_time_ms": round(elapsed_sec * 1000, 2),
                "ttft_ms": 0.0,
                "tokens_per_second": round(comp_toks / max(elapsed_sec, 0.001), 2)
            }


class TGIEngine(BaseInferenceEngine):
    def __init__(self, base_url: str = "http://localhost:8081", default_model: str = "tgi-model"):
        super().__init__("TGI", base_url, default_model)

    async def check_health(self) -> Dict[str, Any]:
        if not httpx:
            return {"status": "offline", "engine": self.name, "error": "httpx package not installed"}
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{self.base_url}/health")
                return {"status": "online" if res.status_code == 200 else "offline", "engine": self.name}
        except Exception as e:
            return {"status": "offline", "engine": self.name, "error": str(e)}

    async def generate(self, prompt: str, model: Optional[str] = None, max_tokens: int = 128) -> Dict[str, Any]:
        if not httpx:
            raise RuntimeError("httpx is required for live TGI HTTP generation")
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_tokens}}
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=120.0) as client:
            res = await client.post(f"{self.base_url}/generate", json=payload)
            res.raise_for_status()
            elapsed_sec = time.perf_counter() - start
            data = res.json()
            text = data.get("generated_text", "")
            comp_toks = len(text.split())

            return {
                "engine": self.name,
                "model": model or self.default_model,
                "text": text,
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": comp_toks,
                "total_time_ms": round(elapsed_sec * 1000, 2),
                "ttft_ms": 0.0,
                "tokens_per_second": round(comp_toks / max(elapsed_sec, 0.001), 2)
            }


class LMDeployEngine(BaseInferenceEngine):
    def __init__(self, base_url: str = "http://localhost:23333", default_model: str = "lmdeploy-model"):
        super().__init__("LMDeploy", base_url, default_model)

    async def check_health(self) -> Dict[str, Any]:
        if not httpx:
            return {"status": "offline", "engine": self.name, "error": "httpx package not installed"}
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{self.base_url}/health")
                return {"status": "online" if res.status_code == 200 else "offline", "engine": self.name}
        except Exception as e:
            return {"status": "offline", "engine": self.name, "error": str(e)}

    async def generate(self, prompt: str, model: Optional[str] = None, max_tokens: int = 128) -> Dict[str, Any]:
        if not httpx:
            raise RuntimeError("httpx is required for live LMDeploy HTTP generation")
        payload = {"model": model or self.default_model, "prompt": prompt, "max_tokens": max_tokens}
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=120.0) as client:
            res = await client.post(f"{self.base_url}/v1/completions", json=payload)
            res.raise_for_status()
            elapsed_sec = time.perf_counter() - start
            data = res.json()
            text = data["choices"][0]["text"]
            comp_toks = len(text.split())

            return {
                "engine": self.name,
                "model": model or self.default_model,
                "text": text,
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": comp_toks,
                "total_time_ms": round(elapsed_sec * 1000, 2),
                "ttft_ms": 0.0,
                "tokens_per_second": round(comp_toks / max(elapsed_sec, 0.001), 2)
            }


class EngineRegistry:
    def __init__(self):
        self.engines: Dict[str, BaseInferenceEngine] = {
            "ollama": OllamaEngine(),
            "vllm": VLLMEngine(),
            "llama.cpp": LlamaCppEngine(),
            "tgi": TGIEngine(),
            "lmdeploy": LMDeployEngine()
        }

    def get_engine(self, engine_name: str) -> Optional[BaseInferenceEngine]:
        return self.engines.get(engine_name.lower())

    async def get_all_health(self) -> Dict[str, Any]:
        results = {}
        for name, engine in self.engines.items():
            results[name] = await engine.check_health()
        return results
