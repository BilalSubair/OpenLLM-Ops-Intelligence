"""
Centralized Model Registry.
Maintains model metadata, quantization profiles, engine bindings,
and hardware tier recommendations.
"""

from typing import Dict, Any, List, Optional

try:
    from pydantic import BaseModel
    class RegisteredModel(BaseModel):
        model_id: str
        display_name: str
        parameters_billion: float
        quantization: str
        preferred_engine: str
        context_window: int
        vram_minimum_gb: float
        recommended_hardware: str
        status: str = "active"
except ImportError:
    class RegisteredModel:
        def __init__(
            self,
            model_id: str,
            display_name: str,
            parameters_billion: float,
            quantization: str,
            preferred_engine: str,
            context_window: int,
            vram_minimum_gb: float,
            recommended_hardware: str,
            status: str = "active"
        ):
            self.model_id = model_id
            self.display_name = display_name
            self.parameters_billion = parameters_billion
            self.quantization = quantization
            self.preferred_engine = preferred_engine
            self.context_window = context_window
            self.vram_minimum_gb = vram_minimum_gb
            self.recommended_hardware = recommended_hardware
            self.status = status

        def dict(self) -> Dict[str, Any]:
            return {
                "model_id": self.model_id,
                "display_name": self.display_name,
                "parameters_billion": self.parameters_billion,
                "quantization": self.quantization,
                "preferred_engine": self.preferred_engine,
                "context_window": self.context_window,
                "vram_minimum_gb": self.vram_minimum_gb,
                "recommended_hardware": self.recommended_hardware,
                "status": self.status
            }


class ModelRegistry:
    def __init__(self):
        self.models: Dict[str, RegisteredModel] = {}
        self._seed_default_models()

    def _seed_default_models(self):
        defaults = [
            RegisteredModel(
                model_id="qwen3.5:2b",
                display_name="Qwen 3.5 2B (Local Default)",
                parameters_billion=2.0,
                quantization="Q4_K_M",
                preferred_engine="ollama",
                context_window=32768,
                vram_minimum_gb=3.5,
                recommended_hardware="Consumer GPU / Apple Silicon (8GB+)",
                status="active"
            ),
            RegisteredModel(
                model_id="llama-3.2-3b-instruct-q4",
                display_name="Llama 3.2 3B Instruct (Q4_K_M)",
                parameters_billion=3.2,
                quantization="Q4_K_M",
                preferred_engine="llama.cpp",
                context_window=8192,
                vram_minimum_gb=4.5,
                recommended_hardware="Consumer GPU (8GB+)",
                status="active"
            ),
            RegisteredModel(
                model_id="llama-3.1-8b-instruct-q8",
                display_name="Llama 3.1 8B Instruct (Q8_0 High Precision)",
                parameters_billion=8.0,
                quantization="Q8_0",
                preferred_engine="vllm",
                context_window=16384,
                vram_minimum_gb=12.0,
                recommended_hardware="NVIDIA RTX 4080 / 4090 / A10G (16GB+)",
                status="active"
            ),
            RegisteredModel(
                model_id="qwen2.5-coder-7b",
                display_name="Qwen 2.5 Coder 7B",
                parameters_billion=7.6,
                quantization="Q5_K_M",
                preferred_engine="vllm",
                context_window=32768,
                vram_minimum_gb=9.5,
                recommended_hardware="NVIDIA RTX 4070Ti / A4000 (12GB+)",
                status="active"
            ),
            RegisteredModel(
                model_id="mistral-nemo-12b-awq",
                display_name="Mistral NeMo 12B AWQ",
                parameters_billion=12.2,
                quantization="AWQ-4bit",
                preferred_engine="tgi",
                context_window=16384,
                vram_minimum_gb=14.0,
                recommended_hardware="NVIDIA RTX 4090 / A100 (24GB+)",
                status="standby"
            )
        ]
        for m in defaults:
            self.models[m.model_id] = m

    def register_model(self, model: RegisteredModel):
        self.models[model.model_id] = model

    def get_model(self, model_id: str) -> Optional[RegisteredModel]:
        return self.models.get(model_id)

    def list_models(self) -> List[Dict[str, Any]]:
        return [m.dict() for m in self.models.values()]


# Global model registry instance
model_registry = ModelRegistry()
