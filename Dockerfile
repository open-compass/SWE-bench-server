# Stage 1: Build stage - install dependencies
FROM docker:dind AS builder

# Install build dependencies
RUN apk add --no-cache \
    python3 \
    py3-pip \
    build-base \
    git \
    libffi-dev \
    openssl-dev \
    linux-headers

WORKDIR /app

# Copy requirements and install Python dependencies in venv
COPY requirements.txt /app/requirements.txt

RUN python3 -m venv /app/venv \
    && /app/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
    && /app/venv/bin/pip install --no-cache-dir -r /app/requirements.txt \
    && find /app/venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true \
    && find /app/venv -type f -name "*.pyc" -delete

# Stage 2: Final stage - runtime image
FROM docker:dind

# Install only runtime dependencies
RUN apk add --no-cache \
    python3 \
    git \
    curl \
    libffi \
    && rm -rf /var/cache/apk/* /tmp/* /var/tmp/*

WORKDIR /app

# Copy application code
COPY . /app

# Copy virtual environment from builder
COPY --from=builder /app/venv /app/venv

# Make start script executable
RUN chmod +x /app/start.sh

EXPOSE 8080

ENV PATH="/app/venv/bin:$PATH"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["/app/start.sh"]
