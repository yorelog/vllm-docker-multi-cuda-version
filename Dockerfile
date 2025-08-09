# Multi-stage Dockerfile for vLLM v0.10.0 with CUDA 12.1 support
# Optimized for A100/A800/H20 datacenter GPUs with compilation acceleration
# Following official vLLM Dockerfile patterns more closely
# Based on https://github.com/vllm-project/vllm/blob/v0.10.0/docker/Dockerfile

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

#################### BASE BUILD IMAGE ####################
FROM ${BUILD_BASE_IMAGE} AS base
ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG TARGETPLATFORM
ENV DEBIAN_FRONTEND=noninteractive

ARG GET_PIP_URL

# Install Python and basic dependencies following vLLM official pattern
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

# Install uv for faster pip installs following vLLM official pattern
RUN --mount=type=cache,target=/root/.cache/uv \
    python3 -m pip install uv

# Set UV environment variables following vLLM official pattern
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
ENV UV_LINK_MODE=copy

# Upgrade to GCC 10 following vLLM official pattern
RUN apt-get update -y && \
    apt-get install -y gcc-10 g++-10 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 110 --slave /usr/bin/g++ g++ /usr/bin/g++-10 && \
    gcc --version && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ldconfig workaround following vLLM official pattern
RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

WORKDIR /workspace

# Install PyTorch first following vLLM official pattern
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \
        "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1"

# Clone vLLM source from official repository
RUN git clone --depth 1 --branch v0.10.0 https://github.com/vllm-project/vllm.git .

# Set CUDA arch list optimized for datacenter GPUs (A100/A800/H20)
# Following vLLM v0.10.0 official patterns
ARG torch_cuda_arch_list='8.0 9.0a'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}

#################### WHEEL BUILD IMAGE ####################
FROM base AS build
ARG TARGETPLATFORM

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PYTORCH_CUDA_INDEX_BASE_URL

# Install build dependencies following vLLM official pattern
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements/build.txt \
    --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')

# Git repository check (following vLLM official pattern)
ARG GIT_REPO_CHECK=0
RUN if [ "$GIT_REPO_CHECK" != "0" ]; then bash tools/check_repo.sh ; fi

# max jobs used by Ninja to build extensions
ARG max_jobs=2
ENV MAX_JOBS=${max_jobs}
# number of threads used by nvcc
ARG nvcc_threads=8
ENV NVCC_THREADS=$nvcc_threads

# sccache configuration for compilation acceleration (following vLLM official pattern)
ARG USE_SCCACHE=1
ARG SCCACHE_DOWNLOAD_URL=https://github.com/mozilla/sccache/releases/download/v0.8.1/sccache-v0.8.1-x86_64-unknown-linux-musl.tar.gz
ARG SCCACHE_ENDPOINT
ARG SCCACHE_BUCKET_NAME=vllm-build-sccache
ARG SCCACHE_REGION_NAME=us-west-2
ARG SCCACHE_S3_NO_CREDENTIALS=0

# Flag to control whether to use pre-built vLLM wheels
ARG VLLM_USE_PRECOMPILED
ENV VLLM_USE_PRECOMPILED=""
RUN if [ "${VLLM_USE_PRECOMPILED}" = "1" ]; then \
        export VLLM_USE_PRECOMPILED=1 && \
        echo "Using precompiled wheels"; \
    else \
        unset VLLM_USE_PRECOMPILED && \
        echo "Leaving VLLM_USE_PRECOMPILED unset to build wheels from source"; \
    fi

# Build vLLM wheel with sccache acceleration following vLLM official pattern
ENV CCACHE_DIR=/root/.cache/ccache
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/uv \
    if [ "$USE_SCCACHE" = "1" ]; then \
        echo "Installing sccache..." && \
        curl -L -o sccache.tar.gz ${SCCACHE_DOWNLOAD_URL} && \
        tar -xzf sccache.tar.gz && \
        sudo mv sccache-v0.8.1-x86_64-unknown-linux-musl/sccache /usr/bin/sccache && \
        rm -rf sccache.tar.gz sccache-v0.8.1-x86_64-unknown-linux-musl && \
        export SCCACHE_IDLE_TIMEOUT=0 && \
        export CMAKE_BUILD_TYPE=Release && \
        sccache --show-stats && \
        python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38 && \
        sccache --show-stats; \
    else \
        rm -rf .deps && \
        mkdir -p .deps && \
        python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38; \
    fi

# Check the size of the wheel following vLLM official pattern
ARG VLLM_MAX_SIZE_MB=400
ENV VLLM_MAX_SIZE_MB=$VLLM_MAX_SIZE_MB
ARG RUN_WHEEL_CHECK=false
RUN if [ "$RUN_WHEEL_CHECK" = "true" ]; then \
        echo "Wheel size check enabled but script not available in minimal setup"; \
    else \
        echo "Skipping wheel size check."; \
    fi

#################### vLLM INSTALLATION IMAGE ####################
FROM ${FINAL_BASE_IMAGE} AS vllm-base
ARG CUDA_VERSION
ARG PYTHON_VERSION
WORKDIR /vllm-workspace
ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

ARG GET_PIP_URL

# Install Python and other dependencies following vLLM official pattern
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y ccache software-properties-common git curl wget sudo vim python3-pip \
    && apt-get install -y ffmpeg libsm6 libxext6 libgl1 \
    && for i in 1 2 3; do \
        add-apt-repository -y ppa:deadsnakes/ppa && break || \
        { echo "Attempt $i failed, retrying in 5s..."; sleep 5; }; \
    done \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv libibverbs-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS ${GET_PIP_URL} | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version

# Install uv for faster pip installs
RUN --mount=type=cache,target=/root/.cache/uv \
    python3 -m pip install uv

# Set UV environment variables following vLLM official pattern
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
ENV UV_LINK_MODE=copy

# Workaround for Triton issues following vLLM official pattern
RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

# Install vLLM wheel from build stage
RUN --mount=type=bind,from=build,src=/workspace/dist,target=/vllm-workspace/dist \
    --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system dist/*.whl --verbose \
        --extra-index-url https://download.pytorch.org/whl/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.')

# Environment optimization for datacenter GPUs
ENV VLLM_USAGE_SOURCE=production-docker-image

#################### OPENAI API SERVER ####################
FROM vllm-base AS vllm-openai

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
