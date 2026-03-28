"""rjob job preparation, submission, and result polling."""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "rjob_config.yaml"
RJOB_CONFIG_PATH = os.getenv("RJOB_CONFIG_PATH", str(_DEFAULT_CONFIG_PATH))


def load_rjob_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sanitize_job_name(name: str) -> str:
    return name.replace("_", "-")


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


def get_rjob_instance_image(
    instance_id: str,
    config: dict[str, Any],
    arch: str = "x86_64",
) -> str:
    """Convert instance_id to registry image for rjob submission."""
    tag = f"sweb.eval.{arch}.{instance_id.lower()}".replace("__", "_1776_")
    image_registry = config["image_registry"]
    return f"{image_registry}:{tag}"


def build_rjob_submit_cmd(
    *,
    config: dict[str, Any],
    job_name: str,
    image: str,
    run_script_path: str,
) -> list[str]:
    resources = config["resources"]
    schedule = config["schedule"]
    mounts = config.get("mounts") or []
    submit_bin = config.get("submit_bin", "rjob")

    cmd = [
        submit_bin,
        "submit",
        f"--name={job_name}",
        f"--gpu={resources['gpu']}",
        f"--memory={resources['memory']}",
        f"--cpu={resources['cpu']}",
        f"--private-machine={schedule['private_machine']}",
        f"--charged-group={schedule['charged_group']}",
        f"--image={image}",
        f"--host-network={'true' if schedule.get('host_network', False) else 'false'}",
    ]

    for mount in mounts:
        cmd.append(f"--mount={mount}")

    cmd.extend(["--", "bash", run_script_path])
    return cmd


def prepare_job(
    *,
    config_path: str | Path,
    task_payload: dict[str, Any],
    image: str,
    job_name_prefix: str = "swebench",
    rjob_override: dict[str, Any] | None = None,
    job_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = load_rjob_config(config_path)

    if rjob_override:
        config = deep_update(config, rjob_override)

    jobs_root = Path(config["jobs_root"])
    code_root = config["code_root"]
    python_env = config["python_env"]
    env_vars = config.get("env_vars") or {}

    job_id = uuid.uuid4().hex
    instance = task_payload.get("instance", {})
    instance_id = instance.get("instance_id", "unknown")
    date_str = datetime.now().strftime("%Y%m%d")
    job_name = sanitize_job_name(f"{job_name_prefix}-{date_str}-{job_id}")

    if job_dir is None:
        resolved_job_dir = jobs_root / job_id
    else:
        resolved_job_dir = Path(job_dir)

    resolved_job_dir.mkdir(parents=True, exist_ok=True)

    request_path = resolved_job_dir / "request.json"
    result_path = resolved_job_dir / "result.json"
    status_path = resolved_job_dir / "status.json"
    stdout_path = resolved_job_dir / "stdout.log"
    stderr_path = resolved_job_dir / "stderr.log"
    run_script_path = (resolved_job_dir / "run.sh").resolve()

    write_json(request_path, task_payload)
    write_json(
        status_path,
        {
            "status": "submitted",
            "job_id": job_id,
            "instance_id": instance_id,
        },
    )

    # Build env var exports for run.sh
    env_lines = []
    for key, value in env_vars.items():
        env_lines.append(f"export {key}={shlex.quote(str(value))}")

    # Add extra env vars from task_payload (e.g., docker_env forwarded by agent/evaluator)
    extra_env_vars = task_payload.get("extra_env_vars", {})
    for key, value in extra_env_vars.items():
        env_lines.append(f"export {key}={shlex.quote(str(value))}")

    env_block = "\n".join(env_lines)

    run_script = f"""#!/usr/bin/env bash
set -euo pipefail

{env_block}

cd "{code_root}"
"{python_env}/bin/python" -m rjob.rjob_worker "{resolved_job_dir}" > "{stdout_path}" 2> "{stderr_path}"
"""
    run_script_path.write_text(run_script, encoding="utf-8")
    run_script_path.chmod(0o755)

    run_config_path = resolved_job_dir / "run_config.json"
    run_config = {
        "job_id": job_id,
        "job_name": job_name,
        "instance_id": instance_id,
        "image": image,
        "config_path": str(config_path),
        "task_payload_keys": list(task_payload.keys()),
        "llm_config": task_payload.get("llm_config", {}),
        "timestamp": time.time(),
        "run_script": run_script,
        "effective_schedule": config.get("schedule", {}),
        "rjob_override": rjob_override or {},
        "job_dir": str(resolved_job_dir),
    }
    write_json(run_config_path, run_config)

    cmd = build_rjob_submit_cmd(
        config=config,
        job_name=job_name,
        image=image,
        run_script_path=str(run_script_path),
    )

    return {
        "config": config,
        "job_id": job_id,
        "job_name": job_name,
        "job_dir": str(resolved_job_dir),
        "request_path": str(request_path),
        "result_path": str(result_path),
        "status_path": str(status_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "run_script_path": str(run_script_path),
        "run_config_path": str(run_config_path),
        "submit_cmd": cmd,
    }


def submit_job(
    *,
    config_path: str | Path,
    task_payload: dict[str, Any],
    image: str,
    job_name_prefix: str = "swebench",
    rjob_override: dict[str, Any] | None = None,
    job_dir: str | Path | None = None,
) -> dict[str, Any]:
    prepared = prepare_job(
        config_path=config_path,
        task_payload=task_payload,
        image=image,
        job_name_prefix=job_name_prefix,
        rjob_override=rjob_override,
        job_dir=job_dir,
    )

    proc = subprocess.run(prepared["submit_cmd"], text=True, capture_output=True)

    run_config_path = Path(prepared["run_config_path"])
    run_config = read_json(run_config_path)
    run_config["submit_stdout"] = proc.stdout
    run_config["submit_stderr"] = proc.stderr
    run_config["submit_returncode"] = proc.returncode
    write_json(run_config_path, run_config)

    if proc.returncode != 0:
        status_path = Path(prepared["status_path"])
        write_json(
            status_path,
            {
                "status": "submit_failed",
                "job_id": prepared["job_id"],
                "error": "rjob submit failed",
                "submit_returncode": proc.returncode,
                "submit_stdout": proc.stdout,
                "submit_stderr": proc.stderr,
            },
        )
        raise RuntimeError(
            "rjob submit failed\n"
            f"cmd: {prepared['submit_cmd']}\n"
            f"returncode: {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    status_path = Path(prepared["status_path"])
    status = read_json(status_path)
    status["status"] = "running_or_queued"
    status["submit_returncode"] = proc.returncode
    write_json(status_path, status)

    return {
        "ok": True,
        "submitted": True,
        "job_id": prepared["job_id"],
        "job_name": prepared["job_name"],
        "job_dir": prepared["job_dir"],
        "effective_schedule": prepared["config"].get("schedule", {}),
        "submit_stdout": proc.stdout,
        "submit_stderr": proc.stderr,
    }


def wait_for_result(
    *,
    config_path: str | Path,
    job_dir: str | Path,
) -> dict[str, Any]:
    config = load_rjob_config(config_path)
    poll_cfg = config["poll"]

    resolved_job_dir = Path(job_dir)
    result_path = resolved_job_dir / "result.json"

    poll_interval = int(poll_cfg.get("interval_sec", 5))
    timeout_sec = int(poll_cfg.get("timeout_sec", 7200))
    start = time.time()

    while time.time() - start < timeout_sec:
        if result_path.exists():
            try:
                result = read_json(result_path)
            except (json.JSONDecodeError, ValueError):
                logger.warning(
                    "result.json exists but is not valid JSON yet, retrying..."
                )
                time.sleep(poll_interval)
                continue
            result["job_dir"] = str(resolved_job_dir)
            result["effective_schedule"] = config.get("schedule", {})
            return result
        time.sleep(poll_interval)

    raise TimeoutError(f"Timed out waiting for result.json, job_dir={resolved_job_dir}")


def submit_job_and_wait(
    *,
    config_path: str | Path,
    task_payload: dict[str, Any],
    image: str,
    job_name_prefix: str = "swebench",
    rjob_override: dict[str, Any] | None = None,
    job_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = load_rjob_config(config_path)
    retry_cfg = config.get("retry") or {}
    max_retries = int(retry_cfg.get("max_retries", 5))
    retry_delay = int(retry_cfg.get("retry_delay", 10))

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            submit_result = submit_job(
                config_path=config_path,
                task_payload=task_payload,
                image=image,
                job_name_prefix=job_name_prefix,
                rjob_override=rjob_override,
                job_dir=job_dir,
            )
            result = wait_for_result(
                config_path=config_path,
                job_dir=submit_result["job_dir"],
            )
            result["job_id"] = submit_result["job_id"]
            return result
        except (RuntimeError, TimeoutError) as e:
            last_error = e
            if attempt < max_retries:
                logger.warning(
                    "rjob attempt %d/%d failed: %s, retrying in %ds...",
                    attempt,
                    max_retries,
                    e,
                    retry_delay,
                )
                time.sleep(retry_delay)
            else:
                logger.error("rjob failed after %d attempts", max_retries)
    raise last_error
