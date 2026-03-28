"""
Worker that runs inside rjob containers for SWE-bench agent/eval tasks.

The container IS the SWE-bench instance image (repo at /testbed).
No Docker-in-Docker needed.

Supports two modes:
  - "agent": run mini-swe-agent locally, output patch
  - "eval": apply patch, run tests, grade result
"""

from __future__ import annotations

import copy
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any
import yaml

import swebench.harness.test_spec.python as _ts_python
from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    APPLY_PATCH_PASS,
    DOCKER_WORKDIR,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    LOG_REPORT,
    LOG_TEST_OUTPUT,
)
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import make_test_spec

# Patch out network calls in test_spec
if hasattr(_ts_python, "get_environment_yml"):
    _ts_python.get_environment_yml = lambda *_: ""
if hasattr(_ts_python, "get_requirements"):
    _ts_python.get_requirements = lambda *_: ""

import minisweagent
from minisweagent.agents.default import DefaultAgent
from minisweagent.models.litellm_model import LitellmModel
from packaging.version import Version

_MINISWEAGENT_VERSION = Version(minisweagent.__version__)
_IS_V2 = _MINISWEAGENT_VERSION.major >= 2

_MINISWEAGENT_DIR = Path(minisweagent.__file__).parent
if _IS_V2:
    _SWEBENCH_YAML_PATH = _MINISWEAGENT_DIR / "config" / "benchmarks" / "swebench.yaml"
else:
    _SWEBENCH_YAML_PATH = _MINISWEAGENT_DIR / "config" / "extra" / "swebench.yaml"


def _load_swebench_config() -> dict:
    with open(_SWEBENCH_YAML_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

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
        "model_kwargs": model_config.get("model_kwargs", {}),
    }


SWEBENCH_AGENT_CONFIG = _load_swebench_config()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("rjob_worker")

GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def create_local_env(cwd: str, timeout: int, env: dict[str, str]):
    """Create a LocalEnvironment that uses bash instead of /bin/sh.

    The upstream LocalEnvironment uses ``shell=True`` which invokes
    ``/bin/sh -c``.  The Docker path uses ``bash -c`` (via the
    ``interpreter`` config).  To keep behaviour consistent we subclass
    LocalEnvironment and override ``execute`` to run commands through
    ``["bash", "-c", command]`` explicitly.
    """
    from minisweagent.environments.local import LocalEnvironment

    class BashLocalEnvironment(LocalEnvironment):
        """LocalEnvironment variant that always uses bash."""

        def execute(self, action, cwd="", *, timeout=None):
            command = action.get("command", "")
            _cwd = cwd or self.config.cwd or os.getcwd()
            try:
                result = subprocess.run(
                    ["bash", "-c", command],
                    text=True,
                    cwd=_cwd,
                    env=os.environ | self.config.env,
                    timeout=timeout or self.config.timeout,
                    encoding="utf-8",
                    errors="replace",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                output = {
                    "output": result.stdout,
                    "returncode": result.returncode,
                    "exception_info": "",
                }
            except Exception as e:
                raw_output = getattr(e, "output", None)
                raw_output = (
                    raw_output.decode("utf-8", errors="replace")
                    if isinstance(raw_output, bytes)
                    else (raw_output or "")
                )
                output = {
                    "output": raw_output,
                    "returncode": -1,
                    "exception_info": f"An error occurred while executing the command: {e}",
                    "extra": {"exception_type": type(e).__name__, "exception": str(e)},
                }
            self._check_finished(output)
            return output

    return BashLocalEnvironment(cwd=cwd, timeout=timeout, env=env)


def run_agent(request: dict[str, Any]) -> dict[str, Any]:
    """Run mini-swe-agent in agent mode (local environment, no Docker)."""
    instance = request["instance"]
    llm_config = request.get("llm_config", {})
    step_limit = request.get("step_limit", SWEBENCH_AGENT_CONFIG["step_limit"])
    cost_limit = request.get("cost_limit", SWEBENCH_AGENT_CONFIG["cost_limit"])
    request_timeout = request.get("request_timeout")

    model_name = llm_config.get("model_name") or os.getenv("OPENAI_MODEL")
    api_key = llm_config.get("api_key") or os.getenv("OPENAI_API_KEY")
    api_base = llm_config.get("url") or os.getenv("OPENAI_BASE_URL")
    model_config = llm_config.get("model_infer_params") or {}
    temperature = model_config.get("temperature", 0.0)
    top_p = model_config.get("top_p", 1.0)

    # Build model_kwargs
    model_kwargs = {**SWEBENCH_AGENT_CONFIG.get("model_kwargs", {})}
    model_kwargs.update(
        {
            "temperature": temperature,
            "top_p": top_p,
        }
    )

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

    # Create local environment
    env_config = SWEBENCH_AGENT_CONFIG["environment"]
    cwd = env_config["cwd"]
    env_vars = env_config.get("env", {})
    timeout = int(os.getenv("AGENT_COMMAND_TIMEOUT", 0)) or env_config["timeout"]
    env = create_local_env(cwd=cwd, timeout=timeout, env=env_vars)

    try:
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

        task = instance.get("problem_statement")
        if not task:
            raise ValueError("Missing problem_statement in instance")

        instance_id = instance.get("instance_id", "unknown")
        logger.info(
            "Starting agent for instance_id=%s model=%s", instance_id, model_name
        )

        if _IS_V2:
            run_result = agent.run(task)
            exit_status = run_result.get("exit_status", "unknown")
            result = run_result.get("submission", "")
        else:
            exit_status, result = agent.run(task)

        if _IS_V2:
            n_calls = agent.n_calls
            cost = agent.cost
        else:
            n_calls = agent.model.n_calls
            cost = agent.model.cost

        logger.info(
            "Agent finished: exit_status=%s, calls=%d, cost=%.4f",
            exit_status,
            n_calls,
            cost,
        )

        call_stat = {
            "model": model_name,
            "api_calls": n_calls,
            "total_cost": cost,
            "exit_status": exit_status,
        }

        return {
            "ok": True,
            "content": result,
            "model_name": model_name,
            "call_stat": call_stat,
            "messages": agent.messages,
            "exit_status": exit_status,
        }
    finally:
        cleanup = getattr(env, "cleanup", None)
        if callable(cleanup):
            cleanup()


def run_cmd(
    cmd: str,
    *,
    cwd: str,
    timeout: int | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    proc = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        env=merged_env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, output


def run_bash_file(
    script_path: Path,
    *,
    cwd: str,
    timeout: int | None = None,
    env: dict[str, str] | None = None,
) -> tuple[str, bool, float]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    start = time.time()
    try:
        proc = subprocess.run(
            ["/bin/bash", str(script_path)],
            cwd=cwd,
            env=merged_env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        runtime = time.time() - start
        output = (proc.stdout or "") + (proc.stderr or "")
        return output, False, runtime
    except subprocess.TimeoutExpired as e:
        runtime = time.time() - start
        stdout = e.stdout if isinstance(e.stdout, str) else ""
        stderr = e.stderr if isinstance(e.stderr, str) else ""
        output = stdout + stderr
        return output, True, runtime


def run_eval(request: dict[str, Any], job_dir: str) -> dict[str, Any]:
    """Run evaluation in-place inside the container."""
    instance = request["instance"]
    patch = request["patch"]
    model_name = request.get("model_name", "unknown")
    eval_timeout = int(os.getenv("EVAL_TIMEOUT", 0)) or request.get(
        "eval_timeout", 1800
    )

    instance_id = instance.get("instance_id", "unknown")

    test_spec = make_test_spec(
        instance, namespace="swebench", instance_image_tag="latest"
    )

    log_dir = Path(job_dir) / "eval"
    log_dir.mkdir(parents=True, exist_ok=True)

    report_path = log_dir / LOG_REPORT
    test_output_path = log_dir / LOG_TEST_OUTPUT

    # Check for existing report
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return {
            "ok": True,
            "completed": True,
            "resolved": report.get(instance_id, {}).get("resolved", False),
        }

    repo_cwd = DOCKER_WORKDIR
    if not Path(repo_cwd).exists():
        raise RuntimeError(
            f"Expected repo workdir {repo_cwd} to exist inside rjob container, but it does not."
        )

    # Build prediction dict
    pred = {
        KEY_INSTANCE_ID: instance_id,
        KEY_MODEL: model_name,
        KEY_PREDICTION: patch,
    }

    # Write and apply patch
    patch_path = (log_dir / "patch.diff").resolve()
    patch_path.write_text(patch or "", encoding="utf-8")
    logger.info("Patch written to %s", patch_path)

    applied_patch = False
    last_apply_output = ""
    for git_apply_cmd in GIT_APPLY_CMDS:
        rc, output = run_cmd(
            f"{git_apply_cmd} {patch_path}",
            cwd=repo_cwd,
            timeout=eval_timeout,
        )
        last_apply_output = output
        if rc == 0:
            logger.info("%s:\n%s", APPLY_PATCH_PASS, output)
            applied_patch = True
            break
        logger.info("Failed to apply patch with command: %s", git_apply_cmd)

    if not applied_patch:
        logger.info("%s:\n%s", APPLY_PATCH_FAIL, last_apply_output)
        return {
            "ok": False,
            "completed": False,
            "resolved": False,
            "error": f"{APPLY_PATCH_FAIL}: {last_apply_output}",
        }

    # Git diff before eval
    _, git_diff_before = run_cmd(
        "git -c core.fileMode=false diff",
        cwd=repo_cwd,
        timeout=eval_timeout,
    )
    git_diff_before = git_diff_before.strip()
    logger.info("Git diff before eval:\n%s", git_diff_before)

    # Run eval script
    eval_script = test_spec.eval_script
    eval_file = (log_dir / "eval.sh").resolve()
    eval_file.write_text(eval_script, encoding="utf-8")
    eval_file.chmod(0o755)
    logger.info("Eval script written to %s", eval_file)

    test_output, timed_out, total_runtime = run_bash_file(
        eval_file,
        cwd=repo_cwd,
        timeout=eval_timeout,
    )

    logger.info("Test runtime: %.2f seconds", total_runtime)
    test_output_path.write_text(test_output, encoding="utf-8")

    if timed_out:
        with open(test_output_path, "a", encoding="utf-8") as f:
            f.write(f"\n\nTimeout error: {eval_timeout} seconds exceeded.")
        return {
            "ok": False,
            "completed": False,
            "resolved": False,
            "error": f"Test timed out after {eval_timeout} seconds.",
        }

    # Git diff after eval
    _, git_diff_after = run_cmd(
        "git -c core.fileMode=false diff",
        cwd=repo_cwd,
        timeout=eval_timeout,
    )
    git_diff_after = git_diff_after.strip()
    if git_diff_after != git_diff_before:
        logger.info("Git diff changed after running eval script")

    # Grade
    report = get_eval_report(
        test_spec=test_spec,
        prediction=pred,
        test_log_path=test_output_path,
        include_tests_status=True,
    )
    report_path.write_text(json.dumps(report, indent=4), encoding="utf-8")

    return {
        "ok": True,
        "completed": True,
        "resolved": report.get(instance_id, {}).get("resolved", False),
    }


def main(job_dir: str) -> None:
    job_path = Path(job_dir)
    request_path = job_path / "request.json"
    result_path = job_path / "result.json"
    status_path = job_path / "status.json"

    try:
        write_json(status_path, {"status": "running"})
        request = read_json(request_path)
        mode = request.get("mode")

        if mode == "agent":
            result = run_agent(request)
        elif mode == "eval":
            result = run_eval(request, job_dir)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        write_json(result_path, result)
        write_json(status_path, {"status": "done"})
    except Exception as e:
        logger.error("Worker failed: %s", e, exc_info=True)
        write_json(
            result_path,
            {
                "ok": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
        )
        write_json(status_path, {"status": "failed"})


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m rjob.rjob_worker <job_dir>")
    main(sys.argv[1])
