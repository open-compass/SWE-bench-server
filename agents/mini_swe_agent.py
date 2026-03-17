"""
Mini-SWE-Agent runner implementation.

This module provides the MiniSweAgentRunner class that uses mini-swe-agent
to generate patches for SWE-bench instances.
"""

import copy
import logging
import os
from pathlib import Path
from typing import Any

import minisweagent
import yaml
from packaging.version import Version

_MINISWEAGENT_VERSION = Version(minisweagent.__version__)
_IS_V2 = _MINISWEAGENT_VERSION.major >= 2

import swebench.harness.test_spec.python as _ts_python
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.models.litellm_model import LitellmModel
from swebench.harness.test_spec.test_spec import make_test_spec

# Patch out network calls in test_spec
if hasattr(_ts_python, "get_environment_yml"):
    _ts_python.get_environment_yml = lambda *_: ""
if hasattr(_ts_python, "get_requirements"):
    _ts_python.get_requirements = lambda *_: ""

from agents.base import BaseAgentRunner
from swebench_pro_utils import get_swebench_pro_image_uri
from utils import ImageLRUCache, ensure_image_loaded, get_docker_client

# ---------------------------------------------------------------------------
# SWE-bench agent configuration (previously in swebench_agent_config.py)
# ---------------------------------------------------------------------------

_MINISWEAGENT_DIR = Path(minisweagent.__file__).parent
if _IS_V2:
    _SWEBENCH_YAML_PATH = _MINISWEAGENT_DIR / "config" / "benchmarks" / "swebench.yaml"
else:
    _SWEBENCH_YAML_PATH = _MINISWEAGENT_DIR / "config" / "extra" / "swebench.yaml"


def _load_swebench_config() -> dict:
    """Load the SWE-bench agent configuration from the official YAML file."""
    try:
        with open(_SWEBENCH_YAML_PATH, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find swebench.yaml at {_SWEBENCH_YAML_PATH}. "
            "Make sure mini-swe-agent is installed in the expected location."
        ) from None

    agent_config = config.get("agent", {})
    env_config = config.get("environment", {})
    env_config.pop("environment_class", None)
    model_config = config.get("model", {})

    if _IS_V2:
        observation_template = model_config.get("observation_template", "")
        format_error_template = model_config.get("format_error_template", "")
    else:
        observation_template = agent_config.get("action_observation_template", "")
        format_error_template = agent_config.get("format_error_template", "")

    return {
        "system_template": agent_config.get("system_template", ""),
        "instance_template": agent_config.get("instance_template", ""),
        "observation_template": observation_template,
        "format_error_template": format_error_template,
        "step_limit": int(agent_config.get("step_limit", 250)),
        "cost_limit": float(agent_config.get("cost_limit", 3.0)),
        "environment": env_config,
    }


SWEBENCH_AGENT_CONFIG = _load_swebench_config()

logger = logging.getLogger(__name__)


class MiniSweAgentRunner(BaseAgentRunner):
    """
    Agent runner using mini-swe-agent.

    This runner uses the DefaultAgent from minisweagent package to generate
    patches for SWE-bench instances by interacting with a Docker environment.
    """

    def __init__(self, image_cache: ImageLRUCache):
        """
        Initialize the Mini-SWE-Agent runner.

        Args:
            image_cache: LRU cache for Docker images
        """
        super().__init__(image_cache)

    def get_image_key(self, instance: dict[str, Any], benchmark: str) -> str:
        """
        Get the Docker image key for the given instance.

        Args:
            instance: SWE-bench instance dictionary
            benchmark: Benchmark type ("swebench_verified", "swebench", "swebench_lite", "swebench_multilingual", or "swebench_pro")

        Returns:
            Docker image key string
        """
        if benchmark == "swebench_pro":
            instance_id = instance.get("instance_id", "")
            repo = instance.get("repo", "")
            return get_swebench_pro_image_uri(instance_id, repo)
        else:
            test_spec = make_test_spec(
                instance, namespace="swebench", instance_image_tag="latest"
            )
            return test_spec.instance_image_key

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
        Run mini-swe-agent on a SWE-bench instance.

        Args:
            instance: SWE-bench instance data containing problem_statement, repo, etc.
            llm_config: LLM configuration dictionary with model_name, api_key, url, etc.
            step_limit: Maximum number of agent steps (default: 250)
            cost_limit: Maximum cost limit in dollars (default: 3.0)
            request_timeout: Timeout in seconds for each LLM API request
            benchmark: Benchmark type ("swebench_verified", "swebench", "swebench_lite", "swebench_multilingual", or "swebench_pro")

        Returns:
            Dictionary with:
                - content: Generated patch (git diff)
                - model_name: Name of the model used
                - call_stat: API call statistics
                - messages: Agent conversation messages
                - exit_status: Agent exit status
        """
        # Get the image key
        image_key = self.get_image_key(instance, benchmark)

        # Use context manager to protect image and update LRU order
        with self.image_cache.use(image_key):
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
            if _IS_V2:
                model_kwargs["parallel_tool_calls"] = True

            # Add request timeout if specified
            if request_timeout is not None:
                model_kwargs["timeout"] = request_timeout

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
            model_init_kwargs = {
                "model_name": model_name,
                "model_kwargs": model_kwargs,
                "cost_tracking": "ignore_errors",
            }
            if _IS_V2:
                model_init_kwargs["observation_template"] = SWEBENCH_AGENT_CONFIG[
                    "observation_template"
                ]
                model_init_kwargs["format_error_template"] = SWEBENCH_AGENT_CONFIG[
                    "format_error_template"
                ]
            model = LitellmModel(**model_init_kwargs)

            # Create DockerEnvironment
            env_config = SWEBENCH_AGENT_CONFIG["environment"]
            if _IS_V2:
                # v2: pass all env_config fields so future yaml additions are auto-forwarded
                docker_env_kwargs = {**env_config, "image": image_key}
            else:
                # v1: only pass known fields to avoid dataclass TypeError
                docker_env_kwargs = {
                    "image": image_key,
                    "cwd": env_config["cwd"],
                    "timeout": env_config["timeout"],
                    "env": env_config["env"],
                }
            env = DockerEnvironment(**docker_env_kwargs)

            try:
                # Create DefaultAgent with swebench config
                agent_kwargs = {
                    "system_template": SWEBENCH_AGENT_CONFIG["system_template"],
                    "instance_template": SWEBENCH_AGENT_CONFIG["instance_template"],
                    "step_limit": step_limit,
                    "cost_limit": cost_limit,
                }
                if not _IS_V2:
                    agent_kwargs["action_observation_template"] = SWEBENCH_AGENT_CONFIG[
                        "observation_template"
                    ]
                    agent_kwargs["format_error_template"] = SWEBENCH_AGENT_CONFIG[
                        "format_error_template"
                    ]
                agent = DefaultAgent(model, env, **agent_kwargs)

                # Get task from instance
                task = instance.get("problem_statement")
                if not task:
                    raise ValueError("Missing problem_statement in instance")

                instance_id = instance.get("instance_id", "unknown")
                logger.info(f"Starting mini-swe-agent for instance_id={instance_id}")
                logger.info(f"Using model: {model_name}")

                # Run the agent
                if _IS_V2:
                    run_result = agent.run(task)
                    exit_status = run_result.get("exit_status", "unknown")
                    result = run_result.get("submission", "")
                else:
                    exit_status, result = agent.run(task)

                logger.info(f"Agent finished with exit_status={exit_status}")

                if _IS_V2:
                    n_calls = agent.n_calls
                    cost = agent.cost
                else:
                    n_calls = agent.model.n_calls
                    cost = agent.model.cost

                logger.info(f"Agent made {n_calls} API calls, cost: ${cost:.4f}")

                # Build call_stat
                call_stat = {
                    "model": model_name,
                    "api_calls": n_calls,
                    "total_cost": cost,
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
