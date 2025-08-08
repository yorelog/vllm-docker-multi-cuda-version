# The vLLM Dockerfile based on vllm-project/vllm, optimized for CUDA 12.1

# Use official vLLM base image
FROM vllm/vllm-openai:latest

LABEL maintainer="yorelog"
LABEL description="vLLM Docker image optimized for CUDA 12.1"
LABEL version="cuda12.1"

ENV VLLM_USAGE_SOURCE=production-docker-cuda12.1

# The official vLLM image already has all necessary dependencies
# This image is ready to use with CUDA 12.1 drivers
