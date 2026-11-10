"""
Base Agent Module.
Defines agent interface, multi-step reasoning scratchpad, and tool-calling execution.
"""

from typing import Dict, Any, List, Optional
import time


class AgentReasoningStep:
    def __init__(self, step_number: int, thought: str, action: Optional[str] = None, observation: Optional[Any] = None):
        self.step_number = step_number
        self.thought = thought
        self.action = action
        self.observation = observation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step_number,
            "thought": self.thought,
            "action": self.action,
            "observation": self.observation
        }


class BaseSpecialistAgent:
    def __init__(self, name: str, specialty: str, system_prompt: str):
        self.name = name
        self.specialty = specialty
        self.system_prompt = system_prompt
        self.steps: List[AgentReasoningStep] = []

    def log_step(self, thought: str, action: Optional[str] = None, observation: Optional[Any] = None) -> AgentReasoningStep:
        step = AgentReasoningStep(len(self.steps) + 1, thought, action, observation)
        self.steps.append(step)
        return step

    def clear_steps(self):
        self.steps.clear()

    async def diagnose(self, problem_description: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raise NotImplementedError
