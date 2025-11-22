# syntax=docker/dockerfile:1.7

FROM python:3.12-slim

WORKDIR /app

# 1) System deps (rarely change)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libzmq3-dev \
    && rm -rf /var/lib/apt/lists/*

# 2) Python deps layer
#    Only copy reqs.txt here so this layer stays cached unless deps actually change
COPY reqs.txt .

# Use BuildKit cache mount for pip so wheels/downloads are reused across builds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r reqs.txt

# 3) Your code (changes all the time, but now it's after the slow stuff)
COPY . /app

# Default command (overridden in docker-compose)
CMD ["python", "worker.py"]
