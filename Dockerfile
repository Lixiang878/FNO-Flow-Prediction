# syntax=docker/dockerfile:1
FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/Lixiang878/fno-flow-prediction"
LABEL org.opencontainers.image.description="FNO vs UNet surrogate modelling for parametric PDEs (1D Burgers)"

WORKDIR /app

# Copy project (unwanted files excluded via .dockerignore)
COPY . /app

# Offline core install: numpy only. torch/matplotlib are OPTIONAL extras and
# are NOT required to run the solver, baselines, or the offline demo.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e . \
    && pip install --no-cache-dir pytest

# Default: prove the offline (zero-torch) core is green in a clean container.
CMD ["pytest", "-q"]

# Run the offline demo instead of the test suite:
#   docker build -t fno-flow-prediction .
#   docker run --rm fno-flow-prediction python -m fno_flow.cli demo
# For training (needs GPU + torch):
#   pip install -e ".[torch]"   # inside the container
