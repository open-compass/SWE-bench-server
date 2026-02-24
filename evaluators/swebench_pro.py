"""
SWE-bench Pro evaluator implementation.

This module provides the SWEbenchProEvaluator class for evaluating patches
against SWE-bench Pro instances.
"""

import logging
from typing import Any

from evaluators.base import BaseEvaluator
from swebench_pro_utils import (
    assemble_workspace_files,
    get_swebench_pro_image_uri,
    parse_swebench_pro_result,
    run_swebench_pro_container,
)
from utils import ImageLRUCache

logger = logging.getLogger(__name__)


class SWEbenchProEvaluator(BaseEvaluator):
    """
    Evaluator for SWE-bench Pro instances.
    
    Unlike standard SWE-bench, this uses custom run_script.sh and parser.py
    per instance, and determines resolution based on fail_to_pass/pass_to_pass sets.
    """
    
    def __init__(self, image_cache: ImageLRUCache):
        """
        Initialize the SWE-bench Pro evaluator.
        
        Args:
            image_cache: LRU cache for Docker images
        """
        super().__init__(image_cache)
    
    def get_image_key(self, instance: dict[str, Any]) -> str:
        """
        Get the Docker image key for the given instance.
        
        Args:
            instance: SWE-bench Pro instance dictionary
        
        Returns:
            Docker image key string
        """
        instance_id = instance.get("instance_id", "unknown")
        repo = instance.get("repo", "")
        return get_swebench_pro_image_uri(instance_id, repo)
    
    def evaluate(
        self,
        instance: dict[str, Any],
        model_name: str,
        patch: str,
    ) -> dict[str, Any]:
        """
        Evaluate a patch for the given SWE-bench Pro instance.
        
        Args:
            instance: SWE-bench Pro instance dictionary with additional fields:
                - before_repo_set_cmd, selected_test_files_to_run, base_commit
                - fail_to_pass, pass_to_pass
            model_name: Name of the model that generated the patch
            patch: Generated patch content (git diff format)
        
        Returns:
            Dictionary with evaluation results including 'resolved' boolean
        """
        instance_id = instance.get("instance_id", "unknown")
        
        try:
            image_key = self.get_image_key(instance)
            
            logger.info(f"Starting SWE-bench Pro evaluation for instance_id={instance_id}")
            
            # Assemble workspace files
            try:
                workspace_files = assemble_workspace_files(instance, patch)
            except FileNotFoundError as e:
                logger.error(f"Failed to load scripts for {instance_id}: {e}")
                return {"resolved": False, "error": str(e)}
            
            def do_evaluation(client):
                # Run container and get output
                output = run_swebench_pro_container(
                    client, image_key, workspace_files, timeout=1800
                )
                
                # Parse results and determine resolution
                return parse_swebench_pro_result(output, instance)
            
            result = self._with_image_context(image_key, do_evaluation)
            
            logger.info(f"SWE-bench Pro evaluation completed for {instance_id}: resolved={result.get('resolved')}")
            
            return result
        
        except Exception as e:
            logger.error(f"SWE-bench Pro evaluation failed for {instance_id}: {e}", exc_info=True)
            return {"resolved": False, "error": str(e)}
