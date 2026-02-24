"""FastAPI service for running SWE-bench tasks."""

import argparse
import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agents.base import create_agent_runner
from evaluators.base import create_evaluator
from swebench_agent_config import SWEBENCH_AGENT_CONFIG
from utils import ImageLRUCache

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
    benchmark_type: str | None = None
    agent_type: str | None = None
    max_steps: int | None = None
    llm_config: dict[str, Any] | None = None
    modality: str | None = None


class TaskResponse(BaseModel):
    final_answer: str
    trajectory: list | None = None
    call_stat: dict[str, Any] | None = None


@app.post("/api/tasks", response_model=TaskResponse)
async def run_swebench_task(request: TaskRequest):
    """Run SWE-bench task using mini-swe-agent and return the result with evaluation."""
    payload = request.model_dump()

    # --- Extract and validate all inputs ---
    params = payload.get("params") or {}
    instance = params.get("metadata")

    if not instance or not instance.get("repo"):
        raise HTTPException(status_code=400, detail="Missing repository information")

    # Validate benchmark_type
    benchmark_type = payload.get("benchmark_type")
    valid_benchmark_types = {"swebench_verified", "swebench", "swebench_lite", "swebench_multilingual", "swebench_pro"}
    if not benchmark_type:
        raise HTTPException(status_code=400, detail=f"Missing benchmark_type. Must be one of {sorted(valid_benchmark_types)}")
    if benchmark_type not in valid_benchmark_types:
        raise HTTPException(status_code=400, detail=f"Invalid benchmark_type: {benchmark_type}. Must be one of {sorted(valid_benchmark_types)}")

    # Validate agent_type
    agent_type = payload.get("agent_type")
    valid_agent_types = {"mini_swe_agent", "swe_agent"}
    if not agent_type:
        raise HTTPException(status_code=400, detail=f"Missing agent_type. Must be one of {sorted(valid_agent_types)}")
    if agent_type not in valid_agent_types:
        raise HTTPException(status_code=400, detail=f"Invalid agent_type: {agent_type}. Must be one of {sorted(valid_agent_types)}")

    # Validate task input: prefer problem_statement, fallback to question
    task = instance.get("problem_statement") or params.get("question")
    if not task:
        raise HTTPException(
            status_code=400,
            detail="Missing problem_statement in instance or question in params",
        )
    if not instance.get("problem_statement"):
        instance["problem_statement"] = task

    # Extract optional parameters with defaults
    llm_config = payload.get("llm_config", {})
    step_limit = payload.get("max_steps", SWEBENCH_AGENT_CONFIG["step_limit"])
    cost_limit = params.get("cost_limit", SWEBENCH_AGENT_CONFIG["cost_limit"])
    request_timeout = params.get("request_timeout")

    try:
        # Run agent using the abstracted runner
        loop = asyncio.get_running_loop()
        agent_runner = create_agent_runner(agent_type, image_cache)
        result = await loop.run_in_executor(
            task_executor,
            lambda: agent_runner.run(
                instance=instance,
                llm_config=llm_config,
                step_limit=step_limit,
                cost_limit=cost_limit,
                request_timeout=request_timeout,
                benchmark=benchmark_type,
            ),
        )

        # Extract patch from the agent result (git diff)
        result_content = result["content"]
        model_name = result["model_name"]

        # Run evaluation using the abstracted evaluator
        evaluation_result = {}
        try:
            evaluator = create_evaluator(benchmark_type, image_cache)
            evaluation_result = await loop.run_in_executor(
                task_executor,
                lambda: evaluator.evaluate(
                    instance=instance,
                    model_name=model_name,
                    patch=result_content,
                ),
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
