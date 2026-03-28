"""
rjob-based SWE-bench evaluator implementation.

Submits an rjob that runs evaluation inside the SWE-bench instance
container (applies patch, runs tests, grades result).
"""

import logging
import os
from typing import Any

from evaluators.base import BaseEvaluator
from rjob.rjob_runner import (
    RJOB_CONFIG_PATH,
    get_rjob_instance_image,
    load_rjob_config,
    submit_job_and_wait,
)
from utils import ImageLRUCache

logger = logging.getLogger(__name__)


class SWEbenchRJobEvaluator(BaseEvaluator):
    """
    Evaluator that submits evaluation work to an rjob cluster scheduler.

    The rjob container IS the SWE-bench instance image.
    Evaluation runs in-place inside the container (no Docker-in-Docker).
    """

    def __init__(self, image_cache: ImageLRUCache):
        # image_cache accepted for interface compatibility but not used
        super().__init__(image_cache)

    def get_image_key(self, instance: dict[str, Any]) -> str:
        config = load_rjob_config(RJOB_CONFIG_PATH)
        return get_rjob_instance_image(instance["instance_id"], config)

    def evaluate(
        self,
        instance: dict[str, Any],
        model_name: str,
        patch: str,
    ) -> dict[str, Any]:
        config = load_rjob_config(RJOB_CONFIG_PATH)
        image = get_rjob_instance_image(instance["instance_id"], config)

        instance_id = instance.get("instance_id", "unknown")
        logger.info("Submitting rjob eval task for %s, image=%s", instance_id, image)

        task_payload = {
            "mode": "eval",
            "instance": instance,
            "model_name": model_name,
            "patch": patch,
            "eval_timeout": int(os.getenv("EVAL_TIMEOUT", 0)) or 1800,
        }

        result = submit_job_and_wait(
            config_path=RJOB_CONFIG_PATH,
            task_payload=task_payload,
            image=image,
            job_name_prefix="swebench-eval",
        )

        if not result.get("ok"):
            error = result.get("error", "Unknown rjob error")
            logger.error("rjob eval failed for %s: %s", instance_id, error)
            return {"resolved": False, "error": error}

        return {
            "resolved": result.get("resolved", False),
            "completed": result.get("completed", False),
        }
