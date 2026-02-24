"""
SWE-agent configuration loader.

This module loads configuration from the installed sweagent package.

When sweagent is installed via pip (not editable install), the config and tools
directories are not included in the package. In this case, you need to either:
1. Set environment variables SWE_AGENT_CONFIG_DIR and SWE_AGENT_TOOLS_DIR
2. Clone the SWE-agent repository and set the variables to point to it

These configurations are used by SweAgentRunner to run SWE-bench tasks.
"""

import os
from pathlib import Path

import yaml

# Check if environment variables are set, otherwise provide a helpful error
_CONFIG_DIR_ENV = os.getenv("SWE_AGENT_CONFIG_DIR")
_TOOLS_DIR_ENV = os.getenv("SWE_AGENT_TOOLS_DIR")
_CONFIG_ROOT_ENV = os.getenv("SWE_AGENT_CONFIG_ROOT")

# Common SWE-agent repository locations to check
_COMMON_SWE_AGENT_DIRS = [
    Path.home() / "SWE-agent",
    Path.cwd() / "SWE-agent",
]

# Find SWE-agent repository root and set all necessary environment variables
_swe_agent_root = None
for p in _COMMON_SWE_AGENT_DIRS:
    if (p / "config").is_dir() and (p / "tools").is_dir():
        _swe_agent_root = p
        break

if _swe_agent_root:
    # Set SWE_AGENT_CONFIG_ROOT - this is critical for path resolution in Bundle
    if not _CONFIG_ROOT_ENV:
        os.environ["SWE_AGENT_CONFIG_ROOT"] = str(_swe_agent_root)
    
    if not _CONFIG_DIR_ENV:
        os.environ["SWE_AGENT_CONFIG_DIR"] = str(_swe_agent_root / "config")
    
    if not _TOOLS_DIR_ENV:
        os.environ["SWE_AGENT_TOOLS_DIR"] = str(_swe_agent_root / "tools")
    
    if not os.getenv("SWE_AGENT_TRAJECTORY_DIR"):
        os.environ["SWE_AGENT_TRAJECTORY_DIR"] = str(_swe_agent_root / "trajectories")

# Now we can safely import sweagent
try:
    import sweagent
    SWE_AGENT_CONFIG_DIR = sweagent.CONFIG_DIR
    SWE_AGENT_TOOLS_DIR = sweagent.TOOLS_DIR
    _SWEAGENT_AVAILABLE = True
except (ImportError, AssertionError) as e:
    # sweagent not available or config directories not found
    SWE_AGENT_CONFIG_DIR = None
    SWE_AGENT_TOOLS_DIR = None
    _SWEAGENT_AVAILABLE = False
    import warnings
    warnings.warn(
        f"Could not import sweagent: {e}. "
        "Make sure to set SWE_AGENT_CONFIG_DIR, SWE_AGENT_TOOLS_DIR, and "
        "SWE_AGENT_TRAJECTORY_DIR environment variables pointing to a cloned SWE-agent repository."
    )


def load_sweagent_config(config_name: str = "default.yaml") -> dict:
    """
    Load a SWE-agent configuration file.
    
    Args:
        config_name: Name of the config file (e.g., "default.yaml")
    
    Returns:
        Parsed configuration dictionary
    
    Raises:
        RuntimeError: If sweagent is not available
        FileNotFoundError: If config file not found
    """
    if not _SWEAGENT_AVAILABLE or SWE_AGENT_CONFIG_DIR is None:
        raise RuntimeError(
            "sweagent is not available. Please set SWE_AGENT_CONFIG_DIR, "
            "SWE_AGENT_TOOLS_DIR, and SWE_AGENT_TRAJECTORY_DIR environment variables."
        )
    
    config_path = SWE_AGENT_CONFIG_DIR / config_name
    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find {config_name} at {config_path}. "
            "Make sure sweagent is installed correctly."
        )
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def get_default_sweagent_config() -> dict:
    """
    Load the default SWE-agent configuration.
    
    Returns:
        Parsed default.yaml configuration
    """
    return load_sweagent_config("default.yaml")


# Pre-load default config at module import time
try:
    SWE_AGENT_DEFAULT_CONFIG = get_default_sweagent_config()
except (FileNotFoundError, RuntimeError):
    # Allow import to succeed even if config not found or sweagent unavailable
    # (useful for environments where sweagent is not fully installed)
    SWE_AGENT_DEFAULT_CONFIG = {}
