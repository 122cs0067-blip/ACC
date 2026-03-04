# BASE: PyTorch 2.4.0 with CUDA 12.4 (Compatible with your Driver 580+)
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# METADATA
LABEL description="ACC Student Environment (RTX 2000 Ada / Any4)"
ENV DEBIAN_FRONTEND=noninteractive

# 1. SYSTEM DEPENDENCIES
# 'git' for cloning, 'build-essential' for kernel compilation
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    ninja-build \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. PYTHON DEPENDENCIES
# Install libraries for Any4 calibration (C4 dataset processing)
RUN pip install --no-cache-dir \
    transformers==4.46.3 \
    bitsandbytes>=0.46.1 \
    datasets \
    accelerate \
    sentencepiece \
    protobuf \
    scipy \
    pandas

# 3. NVIDIA ARCHITECTURE FLAG
# Force compilation for Ada Lovelace (RTX 4000/2000 Ada series)
ENV TORCH_CUDA_ARCH_LIST="8.9"

# 4. WORKSPACE SETUP
WORKDIR /app/student
