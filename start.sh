#!/bin/sh

# Start Docker daemon using the standard entrypoint (background)
dockerd-entrypoint.sh &

# Wait for Docker daemon socket to appear
echo "Waiting for Docker daemon..."
while [ ! -S /var/run/docker.sock ]; do
    sleep 1
done

echo "Docker daemon is ready!"

# Start FastAPI with uvicorn
exec uvicorn swebench_service:app --host 0.0.0.0 --port 8080 --workers 1
