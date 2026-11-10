"""
OpenAI-Compatible API Adapter for LibreChat and standard OpenAI SDK clients.
Exposes /v1/models and /v1/chat/completions with streaming (SSE) and non-streaming support,
integrating with Model Registry, Hybrid RAG, and AI Copilot.
"""

import time
import uuid
import json
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator

try:
    from pydantic import BaseModel
    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatCompletionRequest(BaseModel):
        model: str = "qwen3.5:2b"
        messages: List[ChatMessage]
        temperature: Optional[float] = 0.7
        max_tokens: Optional[int] = 512
        stream: Optional[bool] = False
except ImportError:
    class ChatMessage:
        def __init__(self, role: str, content: str):
            self.role = role
            self.content = content

    class ChatCompletionRequest:
        def __init__(
            self,
            model: str = "qwen3.5:2b",
            messages: Optional[List[ChatMessage]] = None,
            temperature: float = 0.7,
            max_tokens: int = 512,
            stream: bool = False
        ):
            self.model = model
            self.messages = messages or []
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.stream = stream

try:
    from fastapi import APIRouter, HTTPException, Request
    from fastapi.responses import StreamingResponse
    router = APIRouter(prefix="/v1", tags=["OpenAI Compatibility"])
except ImportError:
    class DummyRouter:
        def get(self, *args, **kwargs):
            return lambda fn: fn
        def post(self, *args, **kwargs):
            return lambda fn: fn
    router = DummyRouter()
    StreamingResponse = None
    Request = None

from registry.model_registry import model_registry
from registry.fallback_router import fallback_router
from copilot.orchestrator import copilot_orchestrator
from monitoring.inference_metrics import inference_metrics_collector
from registry.audit_logger import audit_logger


# --------------------------------------------------
# MODELS LIST ENDPOINT (/v1/models)
# --------------------------------------------------

@router.get("/models")
def list_models():
    """
    Returns list of models in OpenAI format for LibreChat model selector.
    """
    registered = model_registry.list_models()
    now = int(time.time())

    data = []
    for m in registered:
        data.append({
            "id": m["model_id"],
            "object": "model",
            "created": now,
            "owned_by": "openllm-ops-intelligence",
            "permission": [],
            "root": m["model_id"],
            "parent": None,
            "display_name": m["display_name"],
            "quantization": m["quantization"],
            "recommended_hardware": m["recommended_hardware"]
        })

    return {
        "object": "list",
        "data": data
    }


# --------------------------------------------------
# INTENT & CONTEXT ROUTING HELPER
# --------------------------------------------------

async def process_chat_intent(model: str, messages: List[ChatMessage], max_tokens: int) -> str:
    """
    Analyzes conversation messages. If troubleshooting intent or logs are detected,
    leverages Copilot multi-agent triage; otherwise routes through resilient inference engine.
    """
    user_content = ""
    system_content = ""

    for msg in messages:
        if msg.role == "user":
            user_content = msg.content
        elif msg.role == "system":
            system_content = msg.content

    lower_query = user_content.lower()

    # Detect if user asks for troubleshooting or provides error logs
    troubleshoot_keywords = [
        "error", "cuda", "oom", "out of memory", "xid", "bus error",
        "crash", "failed", "crashloopbackoff", "killed", "troubleshoot",
        "why is my", "diagnose", "/dev/shm", "nvidia", "taint", "toleration"
    ]

    is_troubleshooting = any(k in lower_query for k in troubleshoot_keywords)

    if is_troubleshooting:
        # Invoke Copilot Orchestrator
        diagnosis = await copilot_orchestrator.troubleshoot(
            problem_description=user_content,
            context={"system_prompt": system_content}
        )

        lead_agent = diagnosis["coordination"]["lead_agent"]
        root_cause = diagnosis["root_cause_analysis"]
        severity = diagnosis["severity"]
        remediations = diagnosis["consolidated_remediation"]

        response_text = f"### 🛠️ {lead_agent} Diagnostic Report\n\n"
        response_text += f"**Severity**: `{severity}`\n\n"
        response_text += f"**Root Cause Analysis**:\n{root_cause}\n\n"
        response_text += "**Recommended Remediation Steps**:\n"
        for i, step in enumerate(remediations, 1):
            response_text += f"{i}. {step}\n"

        return response_text

    # Standard conversation / general query -> Resilient Fallback Router
    full_prompt = f"{system_content}\nUser: {user_content}\nAssistant:" if system_content else user_content
    result = await fallback_router.route_and_generate(
        prompt=full_prompt,
        model_id=model,
        max_tokens=max_tokens
    )

    if result.get("text"):
        return result["text"]

    return (
        f"I received your inquiry: '{user_content}'. "
        "The OpenLLM Ops backend is active. (Connect Ollama on port 11434 to stream live model weights)."
    )


# --------------------------------------------------
# STREAMING SSE GENERATOR
# --------------------------------------------------

async def stream_chat_completion(
    completion_id: str,
    model: str,
    created_time: int,
    content: str
) -> AsyncGenerator[str, None]:
    """
    Streams OpenAI-compatible Server-Sent Events (SSE) chunks to LibreChat.
    """
    words = content.split(" ")

    for i, word in enumerate(words):
        chunk_text = word if i == len(words) - 1 else word + " "
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk_text},
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(payload)}\n\n"
        await asyncio.sleep(0.01)

    # Final termination chunk
    final_payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ]
    }
    yield f"data: {json.dumps(final_payload)}\n\n"
    yield "data: [DONE]\n\n"


# --------------------------------------------------
# CHAT COMPLETIONS ENDPOINT (/v1/chat/completions)
# --------------------------------------------------

@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_req: Optional[Any] = None):
    """
    OpenAI-compatible chat completion endpoint supporting streaming and non-streaming responses.
    """
    start_time = time.perf_counter()
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())

    content = await process_chat_intent(
        model=request.model,
        messages=request.messages,
        max_tokens=request.max_tokens or 512
    )

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    prompt_tokens = sum(len(m.content.split()) for m in request.messages)
    comp_tokens = len(content.split())

    inference_metrics_collector.record_request(
        model=request.model,
        engine="librechat-custom-gateway",
        latency_ms=elapsed_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=comp_tokens,
        success=True
    )

    client_ip = "127.0.0.1"
    if raw_req and hasattr(raw_req, "client") and raw_req.client:
        client_ip = raw_req.client.host

    audit_logger.log_event(
        event_type="LIBRECHAT_COMPLETION",
        model=request.model,
        engine="openai-compatible-gateway",
        latency_ms=elapsed_ms,
        status="SUCCESS",
        client_ip=client_ip
    )

    # 1. Streaming response
    if request.stream and StreamingResponse:
        return StreamingResponse(
            stream_chat_completion(
                completion_id=completion_id,
                model=request.model,
                created_time=created_time,
                content=content
            ),
            media_type="text/event-stream"
        )

    # 2. Non-streaming response
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_time,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": comp_tokens,
            "total_tokens": prompt_tokens + comp_tokens
        }
    }
