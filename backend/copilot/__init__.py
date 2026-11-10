"""
AI Infrastructure Troubleshooting Copilot.
Features multi-agent architecture (GPU, Kubernetes, Docker, RAG Diagnostics),
automated log analysis, configuration auditing, and SearXNG search.
"""

from copilot.orchestrator import copilot_orchestrator, CopilotOrchestrator
from copilot.tools.log_analyzer import LogAnalyzer
from copilot.tools.config_inspector import ConfigInspector
from copilot.tools.searxng_client import SearXNGClient
