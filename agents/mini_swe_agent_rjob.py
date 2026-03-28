"""
rjob-based mini-swe-agent runner implementation.

Submits an rjob that runs mini-swe-agent inside the SWE-bench instance
container using LocalEnvironment (no Docker-in-Docker needed).
"""

import logging
from typing import Any

from agents.base import BaseAgentRunner
from rjob.rjob_runner import (
    RJOB_CONFIG_PATH,
    get_rjob_instance_image,
    load_rjob_config,
    submit_job_and_wait,
)
from utils import ImageLRUCache

logger = logging.getLogger(__name__)


class MiniSweAgentRJobRunner(BaseAgentRunner):
    """
    Agent runner that submits work to an rjob cluster scheduler.

    The rjob container IS the SWE-bench instance image (repo at /testbed).
    The agent runs locally inside the container using LocalEnvironment.
    """

    def __init__(self, image_cache: ImageLRUCache):
        # image_cache accepted for interface compatibility but not used
        super().__init__(image_cache)

    def get_image_key(self, instance: dict[str, Any], benchmark: str) -> str:
        config = load_rjob_config(RJOB_CONFIG_PATH)
        return get_rjob_instance_image(instance["instance_id"], config)

    def run(
        self,
        instance: dict[str, Any],
        llm_config: dict[str, Any],
        step_limit: int = 250,
        cost_limit: float = 3.0,
        request_timeout: float | None = None,
        benchmark: str = "swebench_verified",
    ) -> dict[str, Any]:
        config = load_rjob_config(RJOB_CONFIG_PATH)
        image = get_rjob_instance_image(instance["instance_id"], config)

        instance_id = instance.get("instance_id", "unknown")
        logger.info("Submitting rjob agent task for %s, image=%s", instance_id, image)

        task_payload = {
            "mode": "agent",
            "instance": instance,
            "llm_config": llm_config,
            "step_limit": step_limit,
            "cost_limit": cost_limit,
            "request_timeout": request_timeout,
        }

        result = submit_job_and_wait(
            config_path=RJOB_CONFIG_PATH,
            task_payload=task_payload,
            image=image,
            job_name_prefix="mini-swe-agent",
        )

        if not result.get("ok"):
            error = result.get("error", "Unknown rjob error")
            tb = result.get("traceback", "")
            raise RuntimeError(f"rjob agent failed: {error}\n{tb}")

        return {
            "content": result.get("content", ""),
            "model_name": result.get("model_name", ""),
            "call_stat": result.get("call_stat", {}),
            "messages": result.get("messages", []),
            "exit_status": result.get("exit_status", "unknown"),
        }
