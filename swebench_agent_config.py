"""
SWE-bench agent configuration templates.

This module loads configuration templates directly from mini-swe-agent's
official swebench.yaml configuration file at runtime.

The YAML file is loaded from the installed minisweagent package:
    minisweagent/config/extra/swebench.yaml

These templates are used by the DefaultAgent to run SWE-bench tasks.
"""

from pathlib import Path

import minisweagent
import yaml

# Path to the official mini-swe-agent swebench.yaml config (from installed package)
_MINISWEAGENT_DIR = Path(minisweagent.__file__).parent
_SWEBENCH_YAML_PATH = _MINISWEAGENT_DIR / "config" / "extra" / "swebench.yaml"


def _load_swebench_config() -> dict:
    """Load the SWE-bench agent configuration from the official YAML file."""
    if not _SWEBENCH_YAML_PATH.exists():
        raise FileNotFoundError(
            f"Could not find swebench.yaml at {_SWEBENCH_YAML_PATH}. "
            "Make sure mini-swe-agent is installed in the expected location."
        )

    with open(_SWEBENCH_YAML_PATH, "r") as f:
        config = yaml.safe_load(f)

    agent_config = config.get("agent", {})
    env_config = config.get("environment", {})

    # Remove environment_class as we create DockerEnvironment directly
    env_config.pop("environment_class", None)

    return {
        "system_template": agent_config.get("system_template", ""),
        "instance_template": agent_config.get("instance_template", ""),
        "action_observation_template": agent_config.get("action_observation_template", ""),
        "format_error_template": agent_config.get("format_error_template", ""),
        "step_limit": int(agent_config.get("step_limit", 250)),
        "cost_limit": float(agent_config.get("cost_limit", 3.0)),
        "environment": env_config,
    }


# Load config at module import time
SWEBENCH_AGENT_CONFIG = _load_swebench_config()
