"""
SWE-agent runner implementation.

This module provides the SweAgentRunner class that uses the official SWE-agent
to generate patches for SWE-bench instances.
"""

import copy
import logging
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

import swebench.harness.test_spec.python as _ts_python
import yaml
from pydantic import SecretStr
from swebench.harness.test_spec.test_spec import make_test_spec

# Patch out network calls in test_spec
if hasattr(_ts_python, "get_environment_yml"):
    _ts_python.get_environment_yml = lambda *_: ""
if hasattr(_ts_python, "get_requirements"):
    _ts_python.get_requirements = lambda *_: ""
from swerex.deployment.config import DockerDeploymentConfig

from agents.base import BaseAgentRunner
from swebench_pro_utils import get_swebench_pro_image_uri
from utils import ImageLRUCache, ensure_image_loaded, get_docker_client

# ---------------------------------------------------------------------------
# SWE-agent configuration loading
# ---------------------------------------------------------------------------

_CONFIG_DIR_ENV = os.getenv("SWE_AGENT_CONFIG_DIR")
_TOOLS_DIR_ENV = os.getenv("SWE_AGENT_TOOLS_DIR")
_CONFIG_ROOT_ENV = os.getenv("SWE_AGENT_CONFIG_ROOT")

_COMMON_SWE_AGENT_DIRS = [
    Path("/mnt/shared-storage-user/liqingqiu/github/SWE-agent"),
    Path.home() / "SWE-agent",
    Path.cwd() / "SWE-agent",
]

_swe_agent_root = None
for _p in _COMMON_SWE_AGENT_DIRS:
    if (_p / "config").is_dir() and (_p / "tools").is_dir():
        _swe_agent_root = _p
        break

if _swe_agent_root:
    if not _CONFIG_ROOT_ENV:
        os.environ["SWE_AGENT_CONFIG_ROOT"] = str(_swe_agent_root)
    if not _CONFIG_DIR_ENV:
        os.environ["SWE_AGENT_CONFIG_DIR"] = str(_swe_agent_root / "config")
    if not _TOOLS_DIR_ENV:
        os.environ["SWE_AGENT_TOOLS_DIR"] = str(_swe_agent_root / "tools")
    if not os.getenv("SWE_AGENT_TRAJECTORY_DIR"):
        os.environ["SWE_AGENT_TRAJECTORY_DIR"] = str(_swe_agent_root / "trajectories")

try:
    import sweagent

    _SWE_AGENT_CONFIG_DIR = sweagent.CONFIG_DIR
except (ImportError, AssertionError) as e:
    _SWE_AGENT_CONFIG_DIR = None
    warnings.warn(
        f"Could not import sweagent: {e}. "
        "Make sure to set SWE_AGENT_CONFIG_DIR, SWE_AGENT_TOOLS_DIR, and "
        "SWE_AGENT_TRAJECTORY_DIR environment variables pointing to a cloned SWE-agent repository."
    )


def _load_default_sweagent_config() -> dict:
    """Load the default SWE-agent configuration (default.yaml)."""
    if _SWE_AGENT_CONFIG_DIR is None:
        raise RuntimeError(
            "sweagent is not available. Please set SWE_AGENT_CONFIG_DIR, "
            "SWE_AGENT_TOOLS_DIR, and SWE_AGENT_TRAJECTORY_DIR environment variables."
        )
    config_path = _SWE_AGENT_CONFIG_DIR / "default.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find default.yaml at {config_path}.")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


try:
    SWE_AGENT_DEFAULT_CONFIG = _load_default_sweagent_config()
except (FileNotFoundError, RuntimeError):
    SWE_AGENT_DEFAULT_CONFIG = {}

# Import SWE-agent components
from sweagent.agent.agents import DefaultAgentConfig, get_agent_from_config
from sweagent.agent.models import GenericAPIModelConfig
from sweagent.agent.problem_statement import TextProblemStatement
from sweagent.environment.repo import PreExistingRepoConfig
from sweagent.environment.swe_env import EnvironmentConfig, SWEEnv

logger = logging.getLogger(__name__)


class SweAgentRunner(BaseAgentRunner):
    """
    Agent runner using the official SWE-agent.

    This runner uses SWE-agent to generate patches for SWE-bench instances
    by interacting with a Docker environment managed by swe-rex.
    """

    def __init__(self, image_cache: ImageLRUCache):
        """
        Initialize the SWE-agent runner.

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
            # Standard SWE-bench: use make_test_spec
            test_spec = make_test_spec(
                instance, namespace="swebench", instance_image_tag="latest"
            )
            return test_spec.instance_image_key

    def _build_model_config(
        self,
        llm_config: dict[str, Any],
        cost_limit: float,
        step_limit: int = 0,
        request_timeout: float | None = None,
    ) -> GenericAPIModelConfig:
        """
        Build SWE-agent model configuration from llm_config.

        Args:
            llm_config: LLM configuration dictionary
            cost_limit: Maximum cost limit in dollars
            step_limit: Maximum number of agent steps (maps to per_instance_call_limit)
            request_timeout: Timeout in seconds for each LLM API request

        Returns:
            GenericAPIModelConfig object
        """
        model_name = llm_config.get("model_name") or os.getenv("OPENAI_MODEL", "gpt-4o")
        api_key = llm_config.get("api_key") or os.getenv("OPENAI_API_KEY")
        api_base = llm_config.get("url") or os.getenv("OPENAI_BASE_URL")
        model_infer_params = llm_config.get("model_infer_params") or {}

        temperature = model_infer_params.get("temperature", 0.0)
        top_p = model_infer_params.get("top_p", 1.0)

        # Build completion_kwargs for additional model parameters
        completion_kwargs = {}
        if model_infer_params:
            # Pass extra body parameters
            completion_kwargs["extra_body"] = {
                "model_infer_params": copy.deepcopy(model_infer_params),
            }

        # Add timeout if specified
        if request_timeout is not None:
            completion_kwargs["timeout"] = request_timeout

        return GenericAPIModelConfig(
            name=model_name,
            per_instance_cost_limit=cost_limit,
            per_instance_call_limit=step_limit,  # Use step_limit as call limit
            temperature=temperature,
            top_p=top_p,
            api_base=api_base,
            api_key=SecretStr(api_key) if api_key else None,
            completion_kwargs=completion_kwargs,
        )

    def _build_agent_config(
        self,
        model_config: GenericAPIModelConfig,
    ) -> DefaultAgentConfig:
        """
        Build SWE-agent DefaultAgentConfig from model config.

        Uses the default.yaml configuration loaded from the sweagent package.

        Args:
            model_config: Model configuration

        Returns:
            DefaultAgentConfig object
        """
        # Start with the default config from sweagent package
        agent_dict = copy.deepcopy(SWE_AGENT_DEFAULT_CONFIG.get("agent", {}))

        # Set the model configuration
        agent_dict["model"] = model_config.model_dump()

        return DefaultAgentConfig.model_validate(agent_dict)

    def _get_repo_name(self, instance: dict[str, Any]) -> str:
        """
        Get the repository name from the instance.

        Args:
            instance: SWE-bench instance dictionary

        Returns:
            Repository name (e.g., "django" for "django/django")
        """
        repo = instance.get("repo", "")
        if "/" in repo:
            return repo.split("/")[-1]
        return repo

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
        Run SWE-agent on a SWE-bench instance.

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

            # Extract instance information
            instance_id = instance.get("instance_id", "unknown")
            problem_statement = instance.get("problem_statement")
            if not problem_statement:
                raise ValueError("Missing problem_statement in instance")

            # Match official SWE-agent SWE-bench conventions:
            # - SWE-bench classic images store the repo at `/testbed`
            # - SWE-bench Pro evaluation images store the repo at `/app`
            if benchmark in (
                "swebench_verified",
                "swebench",
                "swebench_lite",
                "swebench_multilingual",
            ):
                repo_name = "testbed"
            elif benchmark == "swebench_pro":
                repo_name = "app"
            else:
                repo_name = self._get_repo_name(instance)

            base_commit = instance.get("base_commit", "HEAD")

            logger.info(f"Starting SWE-agent for instance_id={instance_id}")

            # Build model configuration
            model_config = self._build_model_config(
                llm_config, cost_limit, step_limit, request_timeout
            )
            model_name = model_config.name
            logger.info(f"Using model: {model_name}")

            # Build agent configuration
            agent_config = self._build_agent_config(model_config)

            # Build environment configuration
            # The SWE-bench Docker images have the repo at /{repo_name}
            env_config = EnvironmentConfig(
                deployment=DockerDeploymentConfig(
                    image=image_key,
                    python_standalone_dir="/root",
                ),
                repo=PreExistingRepoConfig(
                    repo_name=repo_name,
                    base_commit=base_commit,
                ),
            )

            # Create problem statement
            problem = TextProblemStatement(
                text=problem_statement,
                id=instance_id,
            )

            # Create temporary output directory
            with tempfile.TemporaryDirectory() as output_dir:
                output_path = Path(output_dir)
                instance_output_dir = output_path / instance_id
                instance_output_dir.mkdir(parents=True, exist_ok=True)

                # Create environment and agent
                env = SWEEnv.from_config(env_config)
                agent = get_agent_from_config(agent_config)

                try:
                    # Start environment
                    env.start()

                    # Run the agent
                    result = agent.run(
                        env=env,
                        problem_statement=problem,
                        output_dir=instance_output_dir,
                    )

                    # Extract submission (patch)
                    submission = result.info.get("submission", "")
                    exit_status = result.info.get("exit_status", "unknown")

                    logger.info(f"Agent finished with exit_status={exit_status}")

                    # Get model stats
                    model_stats = result.info.get("model_stats", {})
                    api_calls = model_stats.get("api_calls", 0)
                    total_cost = model_stats.get("instance_cost", 0.0)

                    logger.info(
                        f"Agent made {api_calls} API calls, cost: ${total_cost:.4f}"
                    )

                    # Build call_stat
                    call_stat = {
                        "model": model_name,
                        "api_calls": api_calls,
                        "total_cost": total_cost,
                        "exit_status": exit_status,
                    }

                    # Get messages/history from agent
                    messages = []
                    if hasattr(agent, "history"):
                        messages = agent.history

                    return {
                        "content": submission,
                        "model_name": model_name,
                        "call_stat": call_stat,
                        "messages": messages,
                        "exit_status": exit_status,
                    }
                finally:
                    # Clean up the environment
                    try:
                        env.close()
                    except Exception as e:
                        logger.warning(f"Error closing environment: {e}")
