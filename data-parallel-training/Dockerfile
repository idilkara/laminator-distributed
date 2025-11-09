FROM python:3.12-slim

WORKDIR /app

# Install system deps (for pandas / numpy / zmq)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libzmq3-dev \
        && rm -rf /var/lib/apt/lists/*

# Install python packages
COPY reqs.txt .
RUN pip install -r reqs.txt

# Copy code
COPY . /app

# Default command (overridden in docker-compose)
CMD ["python", "worker.py"]
