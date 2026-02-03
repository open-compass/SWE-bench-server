"""FastAPI service for running SWE-bench tasks."""

import argparse
import asyncio
import copy
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.models.litellm_model import LitellmModel
from pydantic import BaseModel
from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    SWEbenchInstance,
)
from swebench.harness.run_evaluation import run_instance
from swebench.harness.test_spec.test_spec import make_test_spec

from swebench_agent_config import SWEBENCH_AGENT_CONFIG
from utils import ImageLRUCache, cleanup_logs, ensure_image_loaded, get_docker_client

logger = logging.getLogger(__name__)


# Lifespan context manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize resources
    logger.info(f"ImageLRUCache initialized: max_size={image_cache.max_size}")
    logger.info(f"ThreadPoolExecutor initialized with max_workers={thread_pool_max_workers}")
    yield
    # Shutdown: Clean up resources
    logger.info("Shutting down thread pool...")
    task_executor.shutdown(wait=True)
    logger.info("Thread pool shut down complete.")


app = FastAPI(
    title="SWE-bench LLM Service with Docker Evaluation",
    version="1.1.0",
    lifespan=lifespan,
)

# Global image cache instance
image_cache = ImageLRUCache()

# Global thread pool for operations
thread_pool_max_workers = int(os.getenv("THREAD_POOL_MAX_WORKERS", "1"))
task_executor = ThreadPoolExecutor(max_workers=thread_pool_max_workers)


class TaskRequest(BaseModel):
    params: dict[str, Any] | None = None
    benchmark: str | None = None
    llm_config: dict[str, Any] | None = None
    modality: str | None = None


class TaskResponse(BaseModel):
    final_answer: str
    trajectory: list | None = None
    call_stat: dict[str, Any] | None = None


def _run_mini_swe_agent(
    instance: SWEbenchInstance,
    llm_config: dict[str, Any],
    step_limit: int = 250,
    cost_limit: float = 3.0,
) -> dict[str, Any]:
    """
    Run mini-swe-agent on a SWE-bench instance.

    Args:
        instance: SWE-bench instance data containing problem_statement, repo, etc.
        llm_config: LLM configuration dictionary with model_name, api_key, url, etc.
        step_limit: Maximum number of agent steps (default: 250)
        cost_limit: Maximum cost limit in dollars (default: 3.0)

    Returns:
        Dictionary with content (git diff), model_name, call_stat, messages, exit_status
    """
    # Get the image key using make_test_spec
    test_spec = make_test_spec(
        instance, namespace="swebench", instance_image_tag="latest"
    )
    image_key = test_spec.instance_image_key

    # Use context manager to protect image and update LRU order
    with image_cache.use(image_key):
        # Ensure the image is loaded
        with get_docker_client() as client:
            ensure_image_loaded(client, image_key)

        # Extract model configuration
        model_name = llm_config.get("model_name") or os.getenv("OPENAI_MODEL")
        api_key = llm_config.get("api_key") or os.getenv("OPENAI_API_KEY")
        api_base = llm_config.get("url") or os.getenv("OPENAI_BASE_URL")
        model_config = llm_config.get("model_infer_params") or {}
        temperature = model_config.get("temperature", 0.0)
        top_p = model_config.get("top_p", 1.0)

        # Build model_kwargs for litellm
        model_kwargs = {
            "drop_params": True,
            "temperature": temperature,
            "top_p": top_p,
        }

        model_kwargs["extra_body"] = {
            "model_infer_params": copy.deepcopy(model_config),
        }

        if api_key:
            model_kwargs["api_key"] = api_key
        if api_base:
            model_kwargs["api_base"] = api_base

        model_kwargs["custom_llm_provider"] = "openai"
        model_kwargs["input_cost_per_token"] = 0.0
        model_kwargs["output_cost_per_token"] = 0.0

        # Create LitellmModel
        model = LitellmModel(
            model_name=model_name,
            model_kwargs=model_kwargs,
            cost_tracking="ignore_errors",
        )

        # Create DockerEnvironment
        env_config = SWEBENCH_AGENT_CONFIG["environment"]
        env = DockerEnvironment(
            image=image_key,
            cwd=env_config["cwd"],
            timeout=env_config["timeout"],
            env=env_config["env"],
        )

        try:
            # Create DefaultAgent with swebench config
            agent = DefaultAgent(
                model,
                env,
                system_template=SWEBENCH_AGENT_CONFIG["system_template"],
                instance_template=SWEBENCH_AGENT_CONFIG["instance_template"],
                action_observation_template=SWEBENCH_AGENT_CONFIG["action_observation_template"],
                format_error_template=SWEBENCH_AGENT_CONFIG["format_error_template"],
                step_limit=step_limit,
                cost_limit=cost_limit,
            )

            # Get task from instance
            task = instance.get("problem_statement")
            if not task:
                raise ValueError("Missing problem_statement in instance")

            instance_id = instance.get("instance_id", "unknown")
            logger.info(f"Starting mini-swe-agent for instance_id={instance_id}")
            logger.info(f"Using model: {model_name}")

            # Run the agent
            exit_status, result = agent.run(task)

            logger.info(f"Agent finished with exit_status={exit_status}")
            logger.info(f"Agent made {agent.model.n_calls} API calls, cost: ${agent.model.cost:.4f}")

            # Build call_stat
            call_stat = {
                "model": model_name,
                "api_calls": agent.model.n_calls,
                "total_cost": agent.model.cost,
                "exit_status": exit_status,
            }

            return {
                "content": result,
                "model_name": model_name,
                "call_stat": call_stat,
                "messages": agent.messages,
                "exit_status": exit_status,
            }
        finally:
            # Clean up the environment (container)
            env.cleanup()


def _run_docker_evaluation(
    instance: SWEbenchInstance, model_name: str, pred: str
) -> dict[str, Any]:
    """
    Run Docker-based evaluation for the generated patch.
    This is a simplified version of SWE-bench evaluation.
    Uses LRU cache to manage images for better performance.
    """
    # Generate random run_id to avoid conflicts in concurrent execution
    run_id = f"swe_bench_{uuid.uuid4().hex}"

    try:
        with get_docker_client() as client:
            test_spec = make_test_spec(
                instance, namespace="swebench", instance_image_tag="latest"
            )
            image_key = test_spec.instance_image_key

            # Use context manager to protect image and update LRU order
            with image_cache.use(image_key):
                # Ensure the image is loaded
                ensure_image_loaded(client, image_key)

                pred_dict = {
                    KEY_INSTANCE_ID: instance.get("instance_id", "unknown"),
                    KEY_MODEL: model_name,
                    KEY_PREDICTION: pred,
                }

                logger.info(f"Starting evaluation for instance_id={instance.get('instance_id', 'unknown')}")

                # Run evaluation
                res = run_instance(
                    test_spec=test_spec,
                    pred=pred_dict,
                    rm_image=False,
                    force_rebuild=False,
                    client=client,
                    run_id=run_id,
                    timeout=1800,
                    rewrite_reports=False,
                )

            # Clean up old images if the cache is full
            # Controlled by IMAGE_CACHE_CLEANUP_ENABLED environment variable (default: true)
            cleanup_enabled = os.getenv("IMAGE_CACHE_CLEANUP_ENABLED", "true").lower() in {"true", "1", "yes"}
            if cleanup_enabled:
                image_cache.cleanup_old_images(client)

            return res if res else {}
    finally:
        # Clean up logs after evaluation completes
        cleanup_logs(run_id)


@app.post("/api/tasks", response_model=TaskResponse)
async def run_swebench_task(request: TaskRequest):
    """Run SWE-bench task using mini-swe-agent and return the result with evaluation."""
    payload = request.model_dump()

    # Extract SWE-bench instance data
    params = payload.get("params", {})
    instance = params.get("metadata")

    if not instance or not instance.get("repo"):
        raise HTTPException(status_code=400, detail="Missing repository information")

    try:
        # Get LLM configuration
        llm_config = payload.get("llm_config", {})

        # Get optional step_limit and cost_limit from params
        step_limit = params.get("step_limit", SWEBENCH_AGENT_CONFIG["step_limit"])
        cost_limit = params.get("cost_limit", SWEBENCH_AGENT_CONFIG["cost_limit"])

        # Validate task input: prefer problem_statement, fallback to question
        task = instance.get("problem_statement") or params.get("question")
        if not task:
            raise HTTPException(
                status_code=400,
                detail="Missing problem_statement in instance or question in params",
            )
        if not instance.get("problem_statement"):
            instance["problem_statement"] = task

        # Run mini-swe-agent
        result = await asyncio.get_running_loop().run_in_executor(
            task_executor, _run_mini_swe_agent, instance, llm_config, step_limit, cost_limit
        )

        # Extract patch from the agent result (git diff)
        result_content = result["content"]
        model_name = result["model_name"]

        # Run evaluation
        evaluation_result = {}
        try:
            evaluation_result = await asyncio.get_running_loop().run_in_executor(
                task_executor,
                _run_docker_evaluation,
                instance,
                model_name,
                result_content,
            )
        except Exception as docker_error:
            logger.error(
                f"Docker evaluation failed, continuing without evaluation: {docker_error}",
                exc_info=True,
            )

        call_stat = result["call_stat"]

        # Create the trajectory from agent messages
        trajectory = [
            {
                "step": i + 1,
                "role": msg.get("role", "unknown"),
                "content": msg.get("content", ""),
            }
            for i, msg in enumerate(result.get("messages", []))
        ]

        # Add evaluation
        trajectory.append(
            {
                "step": len(trajectory) + 1,
                "action": "evaluation",
                "input": f"Patch content length: {len(result_content) if result_content else 0}",
                "output": f"Evaluation result: {evaluation_result}",
            }
        )

        instance_id = instance.get("instance_id", "unknown")
        logger.info(f"Successfully completed SWE-bench task: {instance_id}")
        logger.info(f"Evaluation result: {evaluation_result}")

        # Extract the resolved status and set the correct flag
        resolved = isinstance(evaluation_result, dict) and evaluation_result.get("resolved", False)
        final_answer = str(resolved)

        return TaskResponse(
            final_answer=final_answer,
            trajectory=trajectory,
            call_stat=call_stat,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SWE-bench task execution failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"SWE-bench task execution failed: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "SWE-bench LLM with Evaluation"}


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="FastAPI server for SWE-bench with direct LLM and Docker evaluation"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port to listen on (default: 8080)"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes (default: 1)"
    )
    args = parser.parse_args()

    module_name = Path(__file__).stem
    app_target = f"{module_name}:app"

    swe_bench_images_path = os.getenv("SWE_BENCH_IMAGES_PATH")
    if swe_bench_images_path:
        logger.info(f"SWE_BENCH_IMAGES_PATH is set to: {swe_bench_images_path}")
    else:
        logger.info("SWE_BENCH_IMAGES_PATH is not set, images will be pulled from Docker Hub")

    logger.info(f"Starting server with {args.workers} uvicorn worker(s)")
    uvicorn.run(app_target, host=args.host, port=args.port, workers=args.workers)
