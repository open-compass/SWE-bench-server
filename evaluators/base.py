"""
Abstract base class for SWE-bench evaluators.

This module provides:
- BaseEvaluator: Abstract base class for evaluation logic
- create_evaluator: Factory function to create evaluators by benchmark type
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

from utils import ImageLRUCache, ensure_image_loaded, get_docker_client

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """
    Abstract base class for SWE-bench evaluators.

    Subclasses implement the actual evaluation logic for different benchmark types.
    The base class provides shared image cache management.
    """

    def __init__(self, image_cache: ImageLRUCache):
        """
        Initialize the evaluator.

        Args:
            image_cache: LRU cache for Docker images
        """
        self.image_cache = image_cache

    @abstractmethod
    def get_image_key(self, instance: dict[str, Any]) -> str:
        """
        Get the Docker image key for the given instance.

        Args:
            instance: SWE-bench instance dictionary

        Returns:
            Docker image key string
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        instance: dict[str, Any],
        model_name: str,
        patch: str,
    ) -> dict[str, Any]:
        """
        Evaluate the given patch for the SWE-bench instance.

        Args:
            instance: SWE-bench instance dictionary
            model_name: Name of the model that generated the patch
            patch: Generated patch content (git diff format)

        Returns:
            Dictionary with at least:
                - resolved: bool indicating if the patch resolves the issue
        """
        pass

    def _with_image_context(
        self,
        image_key: str,
        evaluate_fn,
    ) -> dict[str, Any]:
        """
        Execute evaluation with proper image cache management.

        Handles:
        - Image cache protection (LRU order update + reference counting)
        - Image loading if needed
        - Cleanup of old images after evaluation

        Args:
            image_key: Docker image key
            evaluate_fn: Callable that performs the actual evaluation,
                         receives the docker client as argument

        Returns:
            Result from evaluate_fn
        """
        with get_docker_client() as client:
            with self.image_cache.use(image_key):
                # Ensure the image is loaded
                ensure_image_loaded(client, image_key)

                # Run the evaluation
                result = evaluate_fn(client)

            # Clean up old images if needed
            cleanup_enabled = os.getenv(
                "IMAGE_CACHE_CLEANUP_ENABLED", "false"
            ).lower() in {"true", "1", "yes"}
            if cleanup_enabled:
                self.image_cache.cleanup_old_images(client)

            return result


def create_evaluator(benchmark: str, image_cache: ImageLRUCache) -> BaseEvaluator:
    """
    Factory function to create an evaluator for the given benchmark type.

    Args:
        benchmark: Benchmark type ("swebench_verified", "swebench", "swebench_lite", "swebench_multilingual", or "swebench_pro")
        image_cache: LRU cache for Docker images

    Returns:
        Appropriate evaluator instance

    Raises:
        ValueError: If benchmark type is unknown
    """
    if benchmark in (
        "swebench_verified",
        "swebench",
        "swebench_lite",
        "swebench_multilingual",
    ):
        from evaluators.swebench import SWEbenchEvaluator

        return SWEbenchEvaluator(image_cache)
    elif benchmark == "swebench_pro":
        from evaluators.swebench_pro import SWEbenchProEvaluator

        return SWEbenchProEvaluator(image_cache)
    else:
        raise ValueError(f"Unknown benchmark type: {benchmark}")
