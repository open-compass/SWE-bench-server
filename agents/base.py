"""
Abstract base class for SWE-bench agent runners.

This module provides:
- BaseAgentRunner: Abstract base class for inference/agent runners
- create_agent_runner: Factory function to create agent runners by type
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from utils import ImageLRUCache

logger = logging.getLogger(__name__)


class BaseAgentRunner(ABC):
    """
    Abstract base class for SWE-bench agent runners.

    Subclasses implement the actual agent logic (e.g., mini-swe-agent, swe-agent).
    The base class provides shared image cache management.
    """

    def __init__(self, image_cache: ImageLRUCache):
        """
        Initialize the agent runner.

        Args:
            image_cache: LRU cache for Docker images
        """
        self.image_cache = image_cache

    @abstractmethod
    def get_image_key(self, instance: dict[str, Any], benchmark: str) -> str:
        """
        Get the Docker image key for the given instance.

        Args:
            instance: SWE-bench instance dictionary
            benchmark: Benchmark type ("swebench_verified", "swebench", "swebench_lite", "swebench_multilingual", or "swebench_pro")

        Returns:
            Docker image key string
        """
        pass

    @abstractmethod
    def run(
        self,
        instance: dict[str, Any],
        llm_config: dict[str, Any],
        step_limit: int = 250,
        cost_limit: float = 3.0,
        request_timeout: float | None = None,
        benchmark: str = "swebench_verified",
    ) -> dict[str, Any]:
        """
        Run the agent on the given SWE-bench instance.

        Args:
            instance: SWE-bench instance data
            llm_config: LLM configuration dictionary
            step_limit: Maximum number of agent steps
            cost_limit: Maximum cost limit in dollars
            request_timeout: Timeout for LLM API requests
            benchmark: Benchmark type ("swebench_verified", "swebench", "swebench_lite", "swebench_multilingual", or "swebench_pro")

        Returns:
            Dictionary with:
                - content: Generated patch (git diff)
                - model_name: Name of the model used
                - call_stat: API call statistics
                - messages: Agent conversation messages
                - exit_status: Agent exit status
        """
        pass


def create_agent_runner(agent_type: str, image_cache: ImageLRUCache) -> BaseAgentRunner:
    """
    Factory function to create an agent runner for the given agent type.

    Args:
        agent_type: Agent type ("mini_swe_agent" or "swe_agent")
        image_cache: LRU cache for Docker images

    Returns:
        Appropriate agent runner instance

    Raises:
        ValueError: If agent type is unknown
    """
    if agent_type == "mini_swe_agent":
        from agents.mini_swe_agent import MiniSweAgentRunner

        return MiniSweAgentRunner(image_cache)
    elif agent_type == "swe_agent":
        from agents.swe_agent import SweAgentRunner

        return SweAgentRunner(image_cache)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
