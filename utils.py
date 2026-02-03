"""Utility for SWE-bench agent service."""

import logging
import os
import shutil
import threading
import time
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from pathlib import Path

import docker

logger = logging.getLogger(__name__)


class ImageLRUCache:
    """Thread-safe LRU cache for Docker images."""

    def __init__(self, max_size: int | None = None) -> None:
        """Initialize the cache.

        Args:
            max_size: Maximum number of images to retain.
        """
        self.max_size = max_size if max_size is not None else int(os.getenv("IMAGE_CACHE_MAX_SIZE", "20"))
        self.cache = OrderedDict()  # {image_key: timestamp}
        self.usage_count = defaultdict(int)  # {image_key: int} - reference count for images in use
        self.deleting = set()  # {image_key} - images currently being deleted
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

    def cleanup_old_images(self, client: docker.DockerClient) -> None:
        """Remove the oldest images while the cache size exceeds max_size."""
        max_attempts = 50
        attempts = 0

        while attempts < max_attempts:
            # Step 1: Find the candidate image to remove
            candidate_key = None
            with self.lock:
                # Calculate the effective size (current size - images being deleted)
                effective_keys = [k for k in self.cache if k not in self.deleting]

                if len(effective_keys) <= self.max_size:
                    break

                # Find a candidate from the effective keys
                # Since effective_keys preserves order (from OrderedDict), the first one is the oldest
                for key in effective_keys:
                    # Check if the image is in use
                    if self.usage_count[key] > 0:
                        self.cache.move_to_end(key)
                        # Modified the dict, so restart the search to be safe
                        candidate_key = None
                        break

                    # Found a valid candidate
                    candidate_key = key
                    self.deleting.add(candidate_key)
                    break

                if candidate_key is None:
                    # Either moved something to end (LRU update) or didn't find any candidate
                    attempts += 1
                    continue

            # Step 2: Remove image from Docker
            if candidate_key is None:
                break

            removed = False
            try:
                if self._image_exists(client, candidate_key):
                    client.images.remove(candidate_key, force=True)
                    removed = True
                else:
                    # Image doesn't exist in Docker, can remove from the cache
                    removed = True
            except Exception as e:
                logger.warning(f"Failed to remove image {candidate_key}: {e}")
                attempts += 1
                # Continue to clean up the 'deleting' set

            # Step 3: Update cache if removal succeeded and the image is still not in use (in lock)
            with self.lock:
                self.deleting.remove(candidate_key)
                self.condition.notify_all()
                if removed and candidate_key in self.cache and self.usage_count[candidate_key] == 0:
                    del self.cache[candidate_key]
                    attempts = 0  # Successfully removed, reset counter
                else:
                    # Image became in-use during deletion or failed to remove
                    attempts += 1

    @staticmethod
    def _image_exists(client: docker.DockerClient, image_key: str) -> bool:
        """Check if the image exists in Docker."""
        try:
            client.images.get(image_key)
            return True
        except docker.errors.ImageNotFound:
            return False

    @contextmanager
    def use(self, image_key: str):
        """Context manager to safely manage image reference counting.

        Automatically marks image as accessed (updates LRU order) and manages
        reference counting. Increments reference count on entry and decrements
        on exit.

        Args:
            image_key: Image identifier to protect
        """
        with self.lock:
            wait_timeout = float(os.getenv("IMAGE_CACHE_DELETE_WAIT_TIMEOUT", "1200"))
            start_time = time.monotonic()
            while image_key in self.deleting:
                remaining = wait_timeout - (time.monotonic() - start_time)
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out waiting for image deletion lock: {image_key}"
                    )
                self.condition.wait(timeout=remaining)

            if image_key in self.cache:
                self.cache.move_to_end(image_key)
            else:
                self.cache[image_key] = time.time()
            self.usage_count[image_key] += 1

        try:
            yield
        finally:
            with self.lock:
                if image_key in self.usage_count:
                    self.usage_count[image_key] -= 1
                    if self.usage_count[image_key] <= 0:
                        del self.usage_count[image_key]


@contextmanager
def get_docker_client():
    """Context manager for the Docker client to ensure proper cleanup.

    Docker connection can be configured via the DOCKER_HOST environment variable.
    Both Docker Python SDK (used here) and docker CLI (used by DockerEnvironment)
    will respect this environment variable.

    Examples:
    - Local socket: unix:///var/run/docker.sock (default)
    - Remote daemon: tcp://192.168.1.100:2375
    """
    docker_host = os.getenv("DOCKER_HOST", "unix:///var/run/docker.sock")
    client = docker.DockerClient(base_url=docker_host)
    try:
        yield client
    finally:
        try:
            client.close()
        except Exception as e:
            logger.warning(f"Failed to close Docker client: {e}")


def ensure_image_loaded(client: docker.DockerClient, image_key: str) -> None:
    """
    Ensure the Docker image is loaded from the tar file if not already present.

    Args:
        client: Docker client instance
        image_key: Image identifier
    """
    # Check if the image exists
    try:
        client.images.get(image_key)
        logger.debug(f"Image {image_key} already exists, skipping load")
        return  # Image exists, no need to load
    except docker.errors.ImageNotFound:
        pass  # Image doesn't exist, need to load

    # Load image from the tar file or pull from Docker Hub
    images_path_str = os.getenv("SWE_BENCH_IMAGES_PATH")

    if images_path_str:
        # Load from local tar file
        images_path = Path(images_path_str)
        tar_filename = f"{image_key.replace('/', '_').replace(':', '_')}.tar"
        tar_path = images_path / tar_filename

        if not tar_path.exists():
            raise Exception(
                f"Image tar file not found: {tar_path} for remote image {image_key}"
            )
    else:
        # Pull from Docker Hub with retry
        max_retries = 3
        retry_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                logger.info(f"Pulling image from Docker Hub: {image_key} (attempt {attempt + 1}/{max_retries})")
                client.images.pull(image_key)
                logger.info(f"Successfully pulled image {image_key} from Docker Hub")
                return
            except Exception as e:
                # Check if the image now exists (another process might have pulled it)
                try:
                    client.images.get(image_key)
                    logger.info(f"Image {image_key} was pulled by another process")
                    return
                except docker.errors.ImageNotFound:
                    if attempt < max_retries - 1:
                        logger.warning(f"Failed to pull image (attempt {attempt + 1}/{max_retries}): {e}")
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        logger.error(f"Failed to pull image after {max_retries} attempts: {e}")
                        raise

    max_retries = 3
    retry_delay = 1  # seconds

    for attempt in range(max_retries):
        try:
            logger.info(f"Loading image from local tar file: {tar_path} (attempt {attempt + 1}/{max_retries})")
            with tar_path.open("rb") as f:
                client.images.load(f)
            logger.info(f"Successfully loaded image {image_key} from {tar_path}")
            return
        except Exception as e:
            # Check if the image now exists (another thread might have loaded it)
            try:
                client.images.get(image_key)
                logger.info(f"Image {image_key} was loaded by another thread")
                return
            except docker.errors.ImageNotFound:
                # Image still doesn't exist
                if attempt < max_retries - 1:
                    logger.warning(f"Failed to load image (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # The final attempt failed, re-raise the error
                    logger.error(f"Failed to load image from tar file after {max_retries} attempts: {e}")
                    raise


def cleanup_logs(run_id: str) -> None:
    """
    Clean up log files and intermediate results after evaluation.
    Removes the specific run_id directory from logs/run_evaluation/.
    Controlled by SWE_BENCH_CLEANUP_LOGS environment variable (default: true).
    """
    from swebench.harness.constants import RUN_EVALUATION_LOG_DIR

    # Check if cleanup is enabled via environment variable
    cleanup_enabled = os.getenv("SWE_BENCH_CLEANUP_LOGS", "true").lower() in {"true", "1", "yes"}

    if not cleanup_enabled:
        return

    log_dir = Path(RUN_EVALUATION_LOG_DIR) / run_id
    try:
        if log_dir.exists():
            shutil.rmtree(log_dir)
            logger.debug(f"Cleaned up logs for run_id: {run_id}")
    except Exception as e:
        logger.warning(f"Failed to clean up logs for {run_id}: {e}")
