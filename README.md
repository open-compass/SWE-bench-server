# SWE-bench Service for AgentCompass

SWE-bench FastAPI service for integration with AgentCompass service-type benchmarks. It exposes a simple REST API to run the agent on SWE-bench tasks, returning the final answer with evaluation.

## Introduction

- FastAPI app defined in `swebench_service.py`
- Endpoints:
	- `GET /health`: health check
	- `POST /api/tasks`: run a single SWE-bench task and return results (patch, evaluation, trajectory)


## Quick Start

### 1. Configuration

Set environment variables:

- `SWE_BENCH_IMAGES_PATH`: Path containing pre-downloaded SWE-bench Docker images (optional)
- `IMAGE_CACHE_MAX_SIZE`: Max number of cached Docker images (default: 20)
- `THREAD_POOL_MAX_WORKERS`: Number of thread pool workers (default: 1)

#### Image Loading Modes

The service supports two modes for loading SWE-bench evaluation images:

**Mode 1: Local Tar Files**

If `SWE_BENCH_IMAGES_PATH` is set, the service will load pre-downloaded images from local `.tar` files.

**Image Format:**
The required images are `.tar` files exported from Docker (e.g., using `docker save`). Each file should be named according to the SWE-bench naming convention, where all `/` and `:` characters in the image name are replaced with underscores (`_`).

For example, the Docker image:

```
swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest
```
should be saved as:
```
swebench_sweb.eval.x86_64.astropy_1776_astropy-12907_latest.tar
```

These should match the expected image keys for the corresponding SWE-bench tasks and be placed in the `SWE_BENCH_IMAGES_PATH` directory before starting the service.

**Mode 2: Docker Hub**

If `SWE_BENCH_IMAGES_PATH` is not set, the service will automatically pull images from Docker Hub when needed.


### 2. Start the Service

#### Method 1: Run Directly

**Prerequisites:**
- Python 3.12+
- Docker (required for running agent and evaluation)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Start the server:

```bash
python swebench_service.py --host 0.0.0.0 --port 8080
```

#### Method 2: Docker Deployment

**Prerequisites:**
- Docker

Build the image:

```bash
docker build -t swebench-server .
```

Start the service:

```bash
docker run --privileged \
    --name swebench-server \
    -p 8080:8080 \
    -e THREAD_POOL_MAX_WORKERS=4 \
    swebench-server
```

**Optional:** To use Local Tar Files instead of pulling from Docker Hub, mount the image directory and set `SWE_BENCH_IMAGES_PATH`:

Start the service:

```bash
docker run --privileged \
    --name swebench-server \
    -p 8080:8080 \
    -v <host_image_path>:<container_image_path> \
    -e SWE_BENCH_IMAGES_PATH=<container_image_path> \
    -e THREAD_POOL_MAX_WORKERS=4 \
    swebench-server
```
