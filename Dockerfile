FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FLEET_BANK_DIR=/models/fleet-bank \
    FLEET_LOAD_IN_4BIT=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python3 -m pip install --break-system-packages ".[server]"

EXPOSE 8000

CMD ["fleet-serve", "--router", "semantic", "--bank", "/models/fleet-bank", "--host", "0.0.0.0", "--port", "8000"]
