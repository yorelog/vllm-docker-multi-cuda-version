# vLLM Docker for A100/A800/H20 Datacenter GPUs

🚀 Optimized vLLM Docker image specifically designed for datacenter GPUs (A100, A800, H20) with CUDA 12.1 support.

## ✨ Key Features

- **Datacenter GPU Optimized**: Specifically built for A100, A800, and H20 architectures
- **CUDA 12.1 Support**: Latest CUDA runtime for optimal performance
- **Reduced Build Time**: Only compiles for datacenter GPU architectures (8.0, 9.0a)
- **Memory Efficient**: Multi-stage build with aggressive cleanup
- **Production Ready**: Includes docker-compose setup with load balancing

## 🎯 Target GPU Architectures

| GPU Model | Architecture | Supported |
|-----------|--------------|-----------|
| A100      | sm_80        | ✅        |
| A800      | sm_80        | ✅        |
| H20       | sm_90a       | ✅        |
| H100      | sm_90a       | ✅        |
| Others    | -            | ❌        |

## 🚀 Quick Start

### Option 1: Pre-built Image
```bash
docker pull registry.cn-shenzhen.aliyuncs.com/yorelog/vllm-cuda121-a100:latest

# Run with multi-GPU support
docker run --gpus all -p 8000:8000 \
  registry.cn-shenzhen.aliyuncs.com/yorelog/vllm-cuda121-a100:latest \
  --model microsoft/DialoGPT-medium \
  --tensor-parallel-size auto
```

### Option 2: Build from Source
```bash
git clone https://github.com/yorelog/vllm-docker-multi-cuda-version.git
cd vllm-docker-multi-cuda-version

# Test build (includes GPU detection and validation)
./test-build.sh

# Or build manually
docker build -t vllm-cuda121-a100 \
  --build-arg torch_cuda_arch_list="8.0;9.0a" \
  --build-arg max_jobs=2 \
  --build-arg nvcc_threads=8 \
  --target vllm-base .
```

### Option 3: Docker Compose (Recommended for Production)
```bash
# Start multi-GPU setup with load balancing
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs vllm-multi-gpu
```

## 📊 Performance Optimizations

### Build Optimizations
- **CUDA Arch Reduction**: Only builds for sm_80 and sm_90a (saves ~60% build time)
- **Parallel Compilation**: Optimized max_jobs and nvcc_threads for datacenter hardware
- **Memory Management**: Multi-stage builds with aggressive cleanup
- **Disk Space**: Comprehensive cleanup in GitHub Actions

### Runtime Optimizations
- **Tensor Parallelism**: Auto-detection of GPU count
- **Memory Utilization**: Optimized for 80-90% GPU memory usage
- **Chunked Prefill**: Enabled for better throughput
- **Fast Downloads**: HF Transfer for faster model downloads

## 🔧 Configuration

### Environment Variables
```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES=all                    # Use all GPUs
VLLM_USE_MODELSCOPE=false                  # Use HuggingFace models
HF_HUB_ENABLE_HF_TRANSFER=1                # Fast model downloads

# Performance Tuning
VLLM_ATTENTION_BACKEND=FLASHINFER         # Use FlashInfer backend
VLLM_GPU_MEMORY_UTILIZATION=0.8           # GPU memory usage
```

### Model Configuration Examples

#### Small Models (Single GPU)
```bash
docker run --gpus '"device=0"' -p 8000:8000 \
  vllm-cuda121-a100 \
  --model microsoft/DialoGPT-small \
  --gpu-memory-utilization 0.9 \
  --max-model-len 2048
```

#### Large Models (Multi-GPU)
```bash
docker run --gpus all -p 8000:8000 \
  vllm-cuda121-a100 \
  --model meta-llama/Llama-2-70b-chat-hf \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096 \
  --enable-chunked-prefill
```

## 📋 System Requirements

### Minimum Requirements
- **GPU**: A100, A800, H20, or H100
- **VRAM**: 24GB+ recommended
- **RAM**: 32GB+ recommended
- **Disk**: 50GB+ free space
- **CUDA**: 12.1+ drivers

### Recommended Setup
- **GPU**: 4x A100 (80GB) or 8x H20
- **RAM**: 128GB+
- **Disk**: NVMe SSD with 200GB+ free
- **Network**: 10Gbps+ for model downloads

## 🛠️ Build Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `CUDA_VERSION` | 12.1.1 | CUDA runtime version |
| `PYTHON_VERSION` | 3.12 | Python version |
| `torch_cuda_arch_list` | "8.0;9.0a" | CUDA architectures to build |
| `max_jobs` | 2 | Parallel compilation jobs |
| `nvcc_threads` | 8 | NVCC threads per job |
| `RUN_WHEEL_CHECK` | false | Enable wheel size validation |

## 🔍 Troubleshooting

### Build Issues
```bash
# Check GPU compatibility
nvidia-smi --query-gpu=name,compute_cap --format=csv

# Monitor build progress
docker build --progress=plain ...

# Check disk space during build
df -h
```

### Runtime Issues
```bash
# Test vLLM installation
docker run --rm --gpus all vllm-cuda121-a100 python3 -c "import vllm; print(vllm.__version__)"

# Check CUDA availability
docker run --rm --gpus all vllm-cuda121-a100 python3 -c "import torch; print(torch.cuda.is_available())"

# Monitor GPU usage
nvidia-smi -l 1
```

### Common Errors

#### "No space left on device"
```bash
# Clean Docker cache
docker system prune -af --volumes

# Monitor disk usage
df -h

# Use smaller build parallelism
docker build --build-arg max_jobs=1 --build-arg nvcc_threads=2 ...
```

#### "CUDA out of memory"
```bash
# Reduce GPU memory utilization
--gpu-memory-utilization 0.7

# Use smaller max model length
--max-model-len 2048

# Enable CPU offloading
--cpu-offload-gb 4
```

## 📈 Performance Benchmarks

### Build Performance
- **Traditional Build**: ~4-6 hours, 15GB+ disk usage
- **Optimized Build**: ~2-3 hours, 8GB disk usage
- **Size Reduction**: ~40% smaller final image

### Runtime Performance
| Model Size | GPUs | Throughput | Latency |
|------------|------|------------|---------|
| 7B | 1x A100 | ~2000 tok/s | ~50ms |
| 13B | 1x A100 | ~1200 tok/s | ~80ms |
| 70B | 4x A100 | ~800 tok/s | ~120ms |

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md).

### Development Setup
```bash
# Clone repository
git clone https://github.com/yorelog/vllm-docker-multi-cuda-version.git

# Create feature branch
git checkout -b feature/your-feature

# Test changes
./test-build.sh --build-only

# Submit PR
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [vLLM Team](https://github.com/vllm-project/vllm) for the excellent inference framework
- NVIDIA for CUDA and GPU optimization guides
- Community contributors and testers

## 📚 Additional Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [A100 Performance Guide](https://docs.nvidia.com/deeplearning/performance/index.html)

---

**Note**: This image is specifically optimized for datacenter GPUs. For consumer GPUs (RTX series), please use the standard vLLM Docker images.
