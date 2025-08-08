# Multi-stage Dockerfile for vLLM with CUDA 12.1 support
# Based on https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile

ARG CUDA_VERSION=12.1.1
ARG PYTHON_VERSION=3.12

# Use official CUDA base images
ARG BUILD_BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04
ARG FINAL_BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

#################### BUILD ARGUMENTS ####################
ARG GET_PIP_URL="https://bootstrap.pypa.io/get-pip.py"
ARG PIP_INDEX_URL
ARG PIP_EXTRA_INDEX_URL
ARG UV_INDEX_URL=${PIP_INDEX_URL}
ARG UV_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}
ARG PYTORCH_CUDA_INDEX_BASE_URL=https://download.pytorch.org/whl
ARG PIP_KEYRING_PROVIDER=disabled
ARG UV_KEYRING_PROVIDER=${PIP_KEYRING_PROVIDER}

#################### BASE BUILD IMAGE ####################
FROM ${BUILD_BASE_IMAGE} AS base
ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG TARGETPLATFORM
ENV DEBIAN_FRONTEND=noninteractive

ARG GET_PIP_URL

# Install Python and basic dependencies with retry mechanism
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y ccache software-properties-common git curl sudo \
    && for i in 1 2 3; do \
        add-apt-repository -y ppa:deadsnakes/ppa && break || \
        { echo "Attempt $i failed, retrying in 5s..."; sleep 5; }; \
    done \
    && for i in 1 2 3; do \
        apt-get update -y && break || sleep 5; \
    done \
    && for i in 1 2 3; do \
        apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv && break || \
        (apt-get update && sleep 10); \
    done \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS ${GET_PIP_URL} | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PYTORCH_CUDA_INDEX_BASE_URL
ARG PIP_KEYRING_PROVIDER UV_KEYRING_PROVIDER

# Install uv for faster pip installs
RUN --mount=type=cache,target=/root/.cache/uv \
    python3 -m pip install uv

# Configure uv environment
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
ENV UV_LINK_MODE=copy

# Upgrade to GCC 10 to avoid compiler issues
RUN apt-get install -y gcc-10 g++-10 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 110 --slave /usr/bin/g++ g++ /usr/bin/g++-10

# Workaround for triton/pytorch issues
RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

WORKDIR /workspace

# Install PyTorch and dependencies compatible with CUDA 12.1
# Note: CUDA 12.1 uses cu121 index
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu121 \
        torch==2.7.1 \
        torchvision==0.22.1 \
        torchaudio==2.7.1 \
    && python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Set CUDA arch list optimized for A100/A800/H20 datacenter GPUs
# 8.0: A100, A800 (Ampere)
# 9.0a: H100, H20 (Hopper)
ARG torch_cuda_arch_list='8.0 9.0a'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}

#################### WHEEL BUILD IMAGE ####################
FROM base AS build
ARG TARGETPLATFORM

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PYTORCH_CUDA_INDEX_BASE_URL

# Clone vLLM source
RUN git clone --depth 1 --branch main https://github.com/vllm-project/vllm.git /workspace/vllm

WORKDIR /workspace/vllm

# Install build dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements/build.txt \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu121 && \
    echo "Build dependencies installed"

# Build configuration - limit resources to prevent OOM
ARG max_jobs=1
ENV MAX_JOBS=${max_jobs}
ARG nvcc_threads=2
ENV NVCC_THREADS=$nvcc_threads

# Build vLLM wheel with reduced memory usage
ENV CCACHE_DIR=/root/.cache/ccache
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/uv \
    echo "Starting vLLM wheel build..." && \
    echo "Available memory:" && free -h && \
    echo "Available disk space:" && df -h && \
    python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38 && \
    echo "Build completed. Final disk usage:" && df -h

# Check wheel size (optional, can be disabled with build arg)
COPY .buildkite/check-wheel-size.py check-wheel-size.py
ARG VLLM_MAX_SIZE_MB=500
ENV VLLM_MAX_SIZE_MB=$VLLM_MAX_SIZE_MB
ARG RUN_WHEEL_CHECK=false
RUN if [ "$RUN_WHEEL_CHECK" = "true" ]; then \
        python3 check-wheel-size.py dist; \
    else \
        echo "Skipping wheel size check"; \
    fi

#################### vLLM INSTALLATION IMAGE ####################
FROM ${FINAL_BASE_IMAGE} AS vllm-base
ARG CUDA_VERSION
ARG PYTHON_VERSION
WORKDIR /vllm-workspace
ENV DEBIAN_FRONTEND=noninteractive

ARG GET_PIP_URL

# Install Python and dependencies with retry mechanism
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y ccache software-properties-common git curl wget sudo vim python3-pip \
    && apt-get install -y ffmpeg libsm6 libxext6 libgl1 \
    && for i in 1 2 3; do \
        add-apt-repository -y ppa:deadsnakes/ppa && break || \
        { echo "Attempt $i failed, retrying in 5s..."; sleep 5; }; \
    done \
    && for i in 1 2 3; do \
        apt-get update -y && break || sleep 5; \
    done \
    && for i in 1 2 3; do \
        apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv libibverbs-dev && break || \
        (apt-get update && sleep 10); \
    done \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS ${GET_PIP_URL} | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PYTORCH_CUDA_INDEX_BASE_URL
ARG PIP_KEYRING_PROVIDER UV_KEYRING_PROVIDER

# Install uv
RUN --mount=type=cache,target=/root/.cache/uv \
    python3 -m pip install uv

# Configure uv
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
ENV UV_LINK_MODE=copy

# Install vLLM from wheel
RUN --mount=type=bind,from=build,src=/workspace/vllm/dist,target=/vllm-workspace/dist \
    --mount=type=cache,target=/root/.cache/uv \
    echo "Installing vLLM wheel..." && \
    uv pip install --system dist/*.whl --verbose \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu121 && \
    echo "Cleaning up after installation..." && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    && python3 -c "import vllm; print(f'vLLM installed successfully')" && \
    echo "Final cleanup..." && \
    # Clean up any remaining build artifacts
    find /usr/local -name "*.pyc" -delete && \
    find /usr/local -name "__pycache__" -type d -exec rm -rf {} + || true && \
    # Clean up Python caches
    find /root -name "*.pyc" -delete || true && \
    find /root -name "__pycache__" -type d -exec rm -rf {} + || true

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
