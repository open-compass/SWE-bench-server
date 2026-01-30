# SWE-bench Service for AgentCompass

SWE-bench FastAPI service for integration with AgentCompass service-type benchmarks. It exposes a simple REST API to run the agent on SWE-bench tasks, returning the final answer with evaluation.

## Introduction

- FastAPI app defined in `swebench_service.py`
- Endpoints:
	- `GET /health`: health check
	- `POST /api/tasks`: run a single SWE-bench task and return results (patch, evaluation, trajectory)


## Quick Start

### 1. Environment setup

**Python:** 3.12+ recommended

Install Python dependencies:

```bash
pip install -r requirements.txt
```

**Docker:** Required for running agent and evaluation in isolated environments.


### 2. Configuration

Set required environment variables (see `.env` or your deployment system):

- `SWE_BENCH_IMAGES_PATH`: Path to store/load SWE-bench Docker images (**required**)
- `IMAGE_CACHE_MAX_SIZE`: Max number of cached Docker images (default: 20)
- `THREAD_POOL_MAX_WORKERS`: Number of thread pool workers (default: 1)

**Important:** SWE-bench evaluation images need to be pre-downloaded and placed in the directory specified by `SWE_BENCH_IMAGES_PATH`. The service will only read images from this path and will not pull or build them automatically.

**Image Format:**
The required images are `.tar` files exported from Docker (e.g., using `docker save`). Each file should be named according to the SWE-bench naming convention, where all / and : characters in the image name are replaced with underscores (_).

For example, the Docker image:

```
swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest
```
should be saved as:
```
swebench_sweb.eval.x86_64.astropy_1776_astropy-12907_latest.tar
```

These should match the expected image keys for the corresponding SWE-bench tasks and be placed in the `SWE_BENCH_IMAGES_PATH` directory before starting the service.


### 3. Start the Service

#### Method 1: Run the API server

```bash
python swebench_service.py --host 0.0.0.0 --port 8080
```

#### Method 2: Docker Deployment

```bash
docker build -t swebench-server .
docker run --privileged \
    --name swebench-server \
    -p 8080:8080 \
    -e SWE_BENCH_IMAGES_PATH=/your/image/path \
    -e THREAD_POOL_MAX_WORKERS=4 \
    swebench-server
```
