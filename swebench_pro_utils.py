"""
SWE-bench Pro evaluation utilities.

This module provides functions for running SWE-bench Pro evaluations using Docker.
It handles:
- Creating entryscripts from instance metadata
- Assembling workspace files (patch, run_script, parser, entryscript)
- Running containers and collecting results
- Parsing output.json and determining resolved status
"""

import ast
import json
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================================
# SWE-bench Pro Helpers
# ============================================================================

# Default paths for SWE-bench Pro resources
SWEBENCH_PRO_SCRIPTS_DIR = os.getenv(
    "SWEBENCH_PRO_SCRIPTS_DIR",
)
SWEBENCH_PRO_DOCKERFILES_DIR = os.getenv(
    "SWEBENCH_PRO_DOCKERFILES_DIR",
)
SWEBENCH_PRO_DOCKERHUB_USERNAME = os.getenv(
    "SWEBENCH_PRO_DOCKERHUB_USERNAME",
    "jefzda"
)


def get_swebench_pro_image_uri(instance_id: str, repo_name: str, dockerhub_username: str | None = None) -> str:
    """
    Generate Docker Hub image URI for SWE-bench Pro instances.
    
    Args:
        instance_id: The instance ID (e.g., "instance_NodeBB__NodeBB-xxx")
        repo_name: Repository name (e.g., "NodeBB/NodeBB")
        dockerhub_username: Docker Hub username (defaults to SWEBENCH_PRO_DOCKERHUB_USERNAME)
    
    Returns:
        Docker Hub image URI (e.g., "jefzda/sweap-images:nodebb.nodebb-xxx")
    """
    if dockerhub_username is None:
        dockerhub_username = SWEBENCH_PRO_DOCKERHUB_USERNAME
    
    repo_base, repo_name_only = repo_name.lower().split("/")
    hsh = instance_id.replace("instance_", "")

    # Special case handling for element-web
    if instance_id == "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan":
        repo_name_only = 'element-web'
    elif 'element-hq' in repo_name.lower() and 'element-web' in repo_name.lower():
        repo_name_only = 'element'
        if hsh.endswith('-vnan'):
            hsh = hsh[:-5]
    elif hsh.endswith('-vnan'):
        hsh = hsh[:-5]
    
    tag = f"{repo_base}.{repo_name_only}-{hsh}"
    if len(tag) > 128:
        tag = tag[:128]
    
    return f"{dockerhub_username}/sweap-images:{tag}"


def load_swebench_pro_script(instance_id: str, script_name: str, scripts_dir: str | None = None) -> str:
    """
    Load a script file from the SWE-bench Pro scripts directory.
    
    Args:
        instance_id: The instance ID
        script_name: Name of the script file (e.g., "run_script.sh", "parser.py")
        scripts_dir: Directory containing run scripts (defaults to SWEBENCH_PRO_SCRIPTS_DIR)
    
    Returns:
        Content of the script file
    
    Raises:
        FileNotFoundError: If the script file doesn't exist
    """
    if scripts_dir is None:
        scripts_dir = SWEBENCH_PRO_SCRIPTS_DIR
    
    script_path = Path(scripts_dir) / instance_id / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    
    return script_path.read_text()


def load_swebench_pro_dockerfile(instance_id: str, dockerfile_type: str, dockerfiles_dir: str | None = None) -> str:
    """
    Load a Dockerfile from the SWE-bench Pro dockerfiles directory.
    
    Args:
        instance_id: The instance ID
        dockerfile_type: Type of dockerfile ("base_dockerfile" or "instance_dockerfile")
        dockerfiles_dir: Directory containing dockerfiles (defaults to SWEBENCH_PRO_DOCKERFILES_DIR)
    
    Returns:
        Content of the Dockerfile
    
    Raises:
        FileNotFoundError: If the Dockerfile doesn't exist
    """
    if dockerfiles_dir is None:
        dockerfiles_dir = SWEBENCH_PRO_DOCKERFILES_DIR
    
    dockerfile_path = Path(dockerfiles_dir) / dockerfile_type / instance_id / "Dockerfile"
    if not dockerfile_path.exists():
        raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")
    
    return dockerfile_path.read_text()


def create_entryscript(instance: dict[str, Any], dockerfiles_dir: str | None = None) -> str:
    """
    Create the entryscript.sh content for a SWE-bench Pro instance.
    
    The entryscript:
    1. Sets up environment variables from dockerfiles
    2. Resets git to base commit
    3. Applies the patch
    4. Runs setup commands
    5. Executes tests and parses results
    
    Args:
        instance: SWE-bench Pro instance dictionary containing:
            - instance_id: Unique identifier
            - before_repo_set_cmd: Commands to run before tests
            - selected_test_files_to_run: List of test files to run
            - base_commit: Git commit to reset to
        dockerfiles_dir: Path to dockerfiles directory
    
    Returns:
        Content of entryscript.sh
    """
    if dockerfiles_dir is None:
        dockerfiles_dir = SWEBENCH_PRO_DOCKERFILES_DIR
    
    instance_id = instance.get("instance_id", "")
    before_repo_set_cmd = instance.get("before_repo_set_cmd", "").strip()
    if before_repo_set_cmd:
        before_repo_set_cmd = before_repo_set_cmd.split("\n")[-1]  # Get last line
    
    # Handle selected_test_files_to_run - can be string or list
    selected_test_files = instance.get("selected_test_files_to_run", "[]")
    if isinstance(selected_test_files, str):
        selected_test_files = ast.literal_eval(selected_test_files)
    selected_test_files_str = ",".join(selected_test_files)
    
    base_commit = instance.get("base_commit", "")
    
    # Extract ENV commands from dockerfiles
    env_cmds = []
    for dockerfile_type in ["base_dockerfile", "instance_dockerfile"]:
        try:
            dockerfile_content = load_swebench_pro_dockerfile(
                instance_id, dockerfile_type, dockerfiles_dir
            )
            for line in dockerfile_content.split("\n"):
                line = line.strip()
                if line.startswith("ENV"):
                    # Convert ENV commands to export statements
                    env_cmd = line.replace("ENV", "export", 1)
                    env_cmds.append(env_cmd)
        except FileNotFoundError:
            logger.warning(f"Dockerfile not found: {dockerfile_type}/{instance_id}")
    
    env_cmds_str = "\n".join(env_cmds)
    
    entry_script = f"""
{env_cmds_str}
# apply patch
cd /app
git reset --hard {base_commit}
git checkout {base_commit}
git apply -v /workspace/patch.diff
{before_repo_set_cmd}
# run test and save stdout and stderr to separate files
bash /workspace/run_script.sh {selected_test_files_str} > /workspace/stdout.log 2> /workspace/stderr.log
# run parsing script
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
"""
    return entry_script


def assemble_workspace_files(
    instance: dict[str, Any],
    patch: str,
    scripts_dir: str | None = None,
    dockerfiles_dir: str | None = None,
) -> dict[str, str]:
    """
    Assemble all files needed in the workspace for evaluation.
    
    Args:
        instance: SWE-bench Pro instance dictionary
        patch: The patch content (git diff format)
        scripts_dir: Path to run_scripts directory
        dockerfiles_dir: Path to dockerfiles directory
    
    Returns:
        Dictionary mapping filename to content:
            - patch.diff: The patch to apply
            - run_script.sh: Test execution script
            - parser.py: Output parsing script
            - entryscript.sh: Main execution script
    """
    if scripts_dir is None:
        scripts_dir = SWEBENCH_PRO_SCRIPTS_DIR
    if dockerfiles_dir is None:
        dockerfiles_dir = SWEBENCH_PRO_DOCKERFILES_DIR
    
    instance_id = instance.get("instance_id", "")
    
    # Load run scripts
    run_script = load_swebench_pro_script(instance_id, "run_script.sh", scripts_dir)
    parser_script = load_swebench_pro_script(instance_id, "parser.py", scripts_dir)
    
    # Create entryscript
    entryscript = create_entryscript(instance, dockerfiles_dir)
    
    return {
        "patch.diff": patch,
        "run_script.sh": run_script,
        "parser.py": parser_script,
        "entryscript.sh": entryscript,
    }


def parse_swebench_pro_result(
    output: dict[str, Any] | None,
    instance: dict[str, Any],
) -> dict[str, Any]:
    """
    Parse the evaluation output and determine if the instance is resolved.
    
    Resolution is determined by checking if all tests in fail_to_pass and
    pass_to_pass sets are in the passed tests set.
    
    Args:
        output: Parsed output.json content with test results
        instance: SWE-bench Pro instance dictionary containing fail_to_pass and pass_to_pass
    
    Returns:
        Dictionary with:
            - resolved: bool indicating if all required tests passed
            - passed_tests: set of passed test names
            - failed_tests: set of failed test names
            - fail_to_pass: set of tests that should pass after patch
            - pass_to_pass: set of tests that should still pass
            - missing_fail_to_pass: tests in fail_to_pass that didn't pass
            - missing_pass_to_pass: tests in pass_to_pass that didn't pass
    """
    if output is None:
        return {
            "resolved": False,
            "error": "No output.json produced",
            "passed_tests": [],
            "failed_tests": [],
        }
    
    # Extract passed and failed tests from output
    tests = output.get("tests", [])
    passed_tests = {t["name"] for t in tests if t.get("status") == "PASSED" and "name" in t}
    failed_tests = {t["name"] for t in tests if t.get("status") in ("FAILED", "ERROR") and "name" in t}
    
    # Get expected test sets from instance
    fail_to_pass_raw = instance.get("fail_to_pass", "[]")
    pass_to_pass_raw = instance.get("pass_to_pass", "[]")
    
    # Handle both string and list formats
    if isinstance(fail_to_pass_raw, str):
        fail_to_pass = set(ast.literal_eval(fail_to_pass_raw))
    else:
        fail_to_pass = set(fail_to_pass_raw)
    
    if isinstance(pass_to_pass_raw, str):
        pass_to_pass = set(ast.literal_eval(pass_to_pass_raw))
    else:
        pass_to_pass = set(pass_to_pass_raw)
    
    # Check if all required tests passed
    required_tests = fail_to_pass | pass_to_pass
    resolved = required_tests <= passed_tests
    
    # Calculate missing tests
    missing_fail_to_pass = fail_to_pass - passed_tests
    missing_pass_to_pass = pass_to_pass - passed_tests
    
    return {
        "resolved": resolved,
        "passed_tests": list(passed_tests),
        "failed_tests": list(failed_tests),
        "fail_to_pass": list(fail_to_pass),
        "pass_to_pass": list(pass_to_pass),
        "missing_fail_to_pass": list(missing_fail_to_pass),
        "missing_pass_to_pass": list(missing_pass_to_pass),
        "total_tests": len(tests),
        "total_passed": len(passed_tests),
        "total_failed": len(failed_tests),
    }


def run_swebench_pro_container(
    client,
    image_key: str,
    workspace_files: dict[str, str],
    timeout: int = 1800,
) -> dict[str, Any] | None:
    """
    Run a SWE-bench Pro evaluation container.
    
    Args:
        client: Docker client instance
        image_key: Docker image URI
        workspace_files: Dictionary of files to write to workspace
        timeout: Container execution timeout in seconds
    
    Returns:
        Parsed output.json content, or None if evaluation failed
    """
    # Create temporary workspace directory
    workspace_dir = tempfile.mkdtemp(prefix="swebench_pro_")
    
    try:
        # Write workspace files
        for filename, content in workspace_files.items():
            filepath = Path(workspace_dir) / filename
            filepath.write_text(content)
            # Make shell scripts executable
            if filename.endswith(".sh"):
                filepath.chmod(0o755)
        
        # Run container
        logger.info(f"Running SWE-bench Pro container with image: {image_key}")
        
        volumes = {workspace_dir: {"bind": "/workspace", "mode": "rw"}}
        
        container = None
        try:
            container = client.containers.run(
                image_key,
                command=["-c", "bash /workspace/entryscript.sh"],
                entrypoint="/bin/bash",
                volumes=volumes,
                detach=True,
                remove=False,  # Don't auto-remove so we can get logs
            )
            
            # Wait for container to finish
            result = container.wait(timeout=timeout)
            status_code = result.get("StatusCode", 1) if isinstance(result, dict) else 1
            
            if status_code != 0:
                logger.warning(f"Container exited with status code: {status_code}")
                # Get container logs for debugging
                try:
                    logs = container.logs(tail=100).decode("utf-8", errors="replace")
                    logger.debug(f"Container logs (last 100 lines):\n{logs}")
                except Exception as e:
                    logger.warning(f"Failed to get container logs: {e}")
            
        except Exception as e:
            logger.error(f"Container execution failed: {e}")
            return None
        finally:
            # Always clean up container
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass
        
        # Read output.json
        output_path = Path(workspace_dir) / "output.json"
        if output_path.exists():
            try:
                with output_path.open() as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse output.json: {e}")
                return None
        else:
            logger.warning("output.json not found in workspace")
            # Log stdout and stderr for debugging
            stdout_path = Path(workspace_dir) / "stdout.log"
            stderr_path = Path(workspace_dir) / "stderr.log"
            if stdout_path.exists():
                logger.debug(f"stdout.log:\n{stdout_path.read_text()[:2000]}")
            if stderr_path.exists():
                logger.debug(f"stderr.log:\n{stderr_path.read_text()[:2000]}")
            return None
    
    finally:
        # Clean up temporary workspace
        try:
            shutil.rmtree(workspace_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up workspace: {e}")
