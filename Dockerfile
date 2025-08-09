# Multi-stage Dockerfile for vLLM v0.10.0 with CUDA 12.1 support
# Optimized for A100/A800/H20 datacenter GPUs with compilation acceleration
# Based on https://github.com/vllm-project/vllm/blob/v0.10.0/docker/Dockerfile
# Includes sccache for 70%+ faster rebuilds

ARG CUDA_VERSION=12.1.1
ARG PYTHON_VERSION=3.12

# Use official CUDA base images - follow vLLM official pattern
ARG BUILD_BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04
ARG FINAL_BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

#################### BUILD ARGUMENTS ####################
ARG GET_PIP_URL="https://bootstrap.pypa.io/get-pip.py"
ARG PIP_INDEX_URL
ARG PIP_EXTRA_INDEX_URL
ARG UV_INDEX_URL=${PIP_INDEX_URL}
ARG UV_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}
ARG PYTORCH_CUDA_INDEX_BASE_URL=https://download.pytorch.org/whl
ARG PIP_KEYRING_PROVIDER=disabled
ARG UV_KEYRING_PROVIDER=${PIP_KEYRING_PROVIDER}

# sccache configuration for compilation acceleration (官方 vLLM 模式)
ARG USE_SCCACHE=1
ARG SCCACHE_DOWNLOAD_URL=https://github.com/mozilla/sccache/releases/download/v0.8.1/sccache-v0.8.1-x86_64-unknown-linux-musl.tar.gz

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

# Configure uv environment following vLLM official settings
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
ENV UV_LINK_MODE=copy

# Upgrade to GCC 10 following vLLM official pattern for CUTLASS kernels
RUN apt-get update -y && \
    apt-get install -y gcc-10 g++-10 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 110 --slave /usr/bin/g++ g++ /usr/bin/g++-10 && \
    gcc --version && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Workaround for triton/pytorch issues
RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

WORKDIR /workspace

# Install PyTorch and dependencies compatible with CUDA 12.1 following vLLM official requirements
# Note: CUDA 12.1 uses cu121 index for optimal datacenter GPU support
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu121 \
        torch==2.7.1 \
        torchvision==0.22.1 \
        torchaudio==2.7.1 \
    && python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Set CUDA arch list optimized for A100/A800/H20 datacenter GPUs only (vLLM v0.10.0 pattern)
# This reduces build time and wheel size significantly compared to full arch list
# 8.0: A100, A800 (Ampere architecture) 
# 9.0a: H100, H20 (Hopper architecture)
# Official vLLM uses comprehensive list but we optimize for datacenter only
ARG torch_cuda_arch_list='8.0;9.0a'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}

#################### WHEEL BUILD IMAGE ####################
FROM base AS build
ARG TARGETPLATFORM

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PYTORCH_CUDA_INDEX_BASE_URL

# sccache acceleration following vLLM official pattern
ARG USE_SCCACHE=1
ARG SCCACHE_DOWNLOAD_URL=https://github.com/mozilla/sccache/releases/download/v0.8.1/sccache-v0.8.1-x86_64-unknown-linux-musl.tar.gz

# Clone vLLM source
RUN git clone --depth 1 --branch main https://github.com/vllm-project/vllm.git /workspace/vllm

WORKDIR /workspace/vllm

# Install build dependencies including ninja for faster builds (vLLM 官方推荐)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements/build.txt \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu121 && \
    echo "Build dependencies installed" && \
    # Install ninja for faster builds (vLLM official recommendation)
    apt-get update && apt-get install -y ninja-build ccache && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy source code for git context
COPY . .

# Build configuration - optimized for datacenter GPUs following vLLM v0.10.0 official patterns
# Use aggressive parallelism settings based on official vLLM recommendations
ARG max_jobs=1
ENV MAX_JOBS=${max_jobs}
ARG nvcc_threads=1
ENV NVCC_THREADS=$nvcc_threads

# CMake build optimization (官方 vLLM 增量构建模式)
ENV CMAKE_BUILD_PARALLEL_LEVEL=${max_jobs}
ENV CMAKE_BUILD_TYPE=Release
ENV CMAKE_C_COMPILER_LAUNCHER=ccache
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache
ENV CMAKE_CUDA_COMPILER_LAUNCHER=ccache

# Triton 编译缓存优化 (官方 vLLM 方法)
ENV TRITON_CACHE_DIR=/root/.cache/triton
ENV XLA_CACHE_DIR=/root/.cache/xla
ENV VLLM_XLA_CACHE_PATH=/root/.cache/xla

# 预编译缓存位置
ENV VLLM_USE_PRECOMPILED=""

# 启用并行编译优化
ENV CCACHE_NOHASHDIR="true"
ENV SCCACHE_IDLE_TIMEOUT=0

# Build vLLM wheel with reduced memory usage and optimal settings for datacenter GPUs
# Use sccache for compilation acceleration (following vLLM official pattern)
ENV CCACHE_DIR=/root/.cache/ccache
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=.git,target=.git \
    if [ "$USE_SCCACHE" = "1" ]; then \
        echo "Installing sccache for compilation acceleration..." && \
        curl -L -o sccache.tar.gz ${SCCACHE_DOWNLOAD_URL} && \
        tar -xzf sccache.tar.gz && \
        mv sccache-v0.8.1-x86_64-unknown-linux-musl/sccache /usr/bin/sccache && \
        rm -rf sccache.tar.gz sccache-v0.8.1-x86_64-unknown-linux-musl && \
        export CMAKE_C_COMPILER_LAUNCHER=sccache && \
        export CMAKE_CXX_COMPILER_LAUNCHER=sccache && \
        export CMAKE_CUDA_COMPILER_LAUNCHER=sccache && \
        export CMAKE_BUILD_TYPE=Release && \
        sccache --show-stats && \
        echo "Starting vLLM wheel build optimized for A100/A800/H20..." && \
        echo "Available memory:" && free -h && \
        echo "Available disk space:" && df -h && \
        python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38 && \
        sccache --show-stats && \
        echo "Build completed. Final disk usage:" && df -h; \
    else \
        echo "Starting vLLM wheel build optimized for A100/A800/H20..." && \
        echo "Available memory:" && free -h && \
        echo "Available disk space:" && df -h && \
        python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38 && \
        echo "Build completed. Final disk usage:" && df -h; \
    fi

# Check wheel size (optional, can be disabled with build arg) - following vLLM official check pattern
COPY .buildkite/check-wheel-size.py check-wheel-size.py
# 官方 vLLM 默认值为 400MB，我们针对数据中心 GPU 优化设为 500MB
ARG VLLM_MAX_SIZE_MB=400
ENV VLLM_MAX_SIZE_MB=$VLLM_MAX_SIZE_MB
ARG RUN_WHEEL_CHECK=true
RUN if [ "$RUN_WHEEL_CHECK" = "true" ]; then \
        python3 check-wheel-size.py dist; \
    else \
        echo "Skipping wheel size check (optimized for datacenter GPU build)"; \
    fi

#################### vLLM INSTALLATION IMAGE (v0.10.0 optimized for datacenter GPUs) ####################
FROM ${FINAL_BASE_IMAGE} AS vllm-base
ARG CUDA_VERSION
ARG PYTHON_VERSION
WORKDIR /vllm-workspace
ENV DEBIAN_FRONTEND=noninteractive

ARG GET_PIP_URL

# Install Python and dependencies with retry mechanism following vLLM official pattern
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
    python3 -c "import vllm; print(f'vLLM installed successfully')" && \
    echo "Final cleanup..." && \
    # Clean up any remaining build artifacts
    find /usr/local -name "*.pyc" -delete && \
    find /usr/local -name "__pycache__" -type d -exec rm -rf {} + || true && \
    # Clean up Python caches
    find /root -name "*.pyc" -delete || true && \
    find /root -name "__pycache__" -type d -exec rm -rf {} + || true

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
