"""
SWE-bench evaluator implementation.

This module provides the SWEbenchEvaluator class for evaluating patches
against standard SWE-bench instances.
"""

import logging
import uuid
from typing import Any

from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
)
import swebench.harness.test_spec.python as _ts_python
from swebench.harness.run_evaluation import run_instance
from swebench.harness.test_spec.test_spec import make_test_spec

# Patch out network calls in test_spec
if hasattr(_ts_python, "get_environment_yml"):
    _ts_python.get_environment_yml = lambda *_: ""
if hasattr(_ts_python, "get_requirements"):
    _ts_python.get_requirements = lambda *_: ""

from evaluators.base import BaseEvaluator
from utils import ImageLRUCache, cleanup_logs

logger = logging.getLogger(__name__)


class SWEbenchEvaluator(BaseEvaluator):
    """
    Evaluator for standard SWE-bench instances.
    
    Uses the swebench harness to run tests and determine if patches
    correctly resolve the issues.
    """
    
    def __init__(self, image_cache: ImageLRUCache):
        """
        Initialize the SWE-bench evaluator.
        
        Args:
            image_cache: LRU cache for Docker images
        """
        super().__init__(image_cache)
    
    def get_image_key(self, instance: dict[str, Any]) -> str:
        """
        Get the Docker image key for the given instance.
        
        Args:
            instance: SWE-bench instance dictionary
        
        Returns:
            Docker image key string
        """
        test_spec = make_test_spec(
            instance, namespace="swebench", instance_image_tag="latest"
        )
        return test_spec.instance_image_key
    
    def evaluate(
        self,
        instance: dict[str, Any],
        model_name: str,
        patch: str,
    ) -> dict[str, Any]:
        """
        Evaluate a patch for the given SWE-bench instance.
        
        Uses the swebench harness to apply the patch and run tests.
        
        Args:
            instance: SWE-bench instance dictionary
            model_name: Name of the model that generated the patch
            patch: Generated patch content (git diff format)
        
        Returns:
            Dictionary with evaluation results including 'resolved' boolean
        """
        # Generate random run_id to avoid conflicts in concurrent execution
        run_id = f"swe_bench_{uuid.uuid4().hex}"
        
        try:
            test_spec = make_test_spec(
                instance, namespace="swebench", instance_image_tag="latest"
            )
            image_key = test_spec.instance_image_key
            
            pred_dict = {
                KEY_INSTANCE_ID: instance.get("instance_id", "unknown"),
                KEY_MODEL: model_name,
                KEY_PREDICTION: patch,
            }
            
            instance_id = instance.get("instance_id", "unknown")
            logger.info(f"Starting evaluation for instance_id={instance_id}")
            
            def do_evaluation(client):
                return run_instance(
                    test_spec=test_spec,
                    pred=pred_dict,
                    rm_image=False,
                    force_rebuild=False,
                    client=client,
                    run_id=run_id,
                    timeout=1800,
                    rewrite_reports=False,
                )
            
            result = self._with_image_context(image_key, do_evaluation)
            return result if result else {}
        finally:
            # Clean up logs after evaluation completes
            cleanup_logs(run_id)
