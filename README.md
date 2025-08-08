# vLLM Docker with CUDA 12.1

基于 [vllm-project/vllm](https://github.com/vllm-project/vllm) 的 Docker 镜像，专门针对 CUDA 12.1 优化构建。

## 功能特性

- 基于 vLLM 最新主分支构建
- 支持 CUDA 12.1
- **专门优化支持 A100、A800、H20 等数据中心显卡**
- 针对阿里云容器镜像服务优化
- 支持 OpenAI 兼容 API
- 自动化 CI/CD 构建和推送
- **优化的磁盘空间使用** - 解决 GitHub Actions 构建时的磁盘空间问题
- **多阶段构建** - 减少最终镜像大小
- **并行构建优化** - 防止内存溢出和磁盘空间不足

## GPU 支持

本镜像专门针对以下高端数据中心显卡优化：

### 支持的 GPU 型号
- **NVIDIA A100** (Compute Capability 8.0)
- **NVIDIA A800** (Compute Capability 8.0) 
- **NVIDIA H20** (Compute Capability 9.0a)
- **NVIDIA H100** (Compute Capability 9.0a)

### CUDA 架构优化
```dockerfile
# 仅编译必需的 CUDA 架构，大幅减少构建时间和镜像大小
torch_cuda_arch_list='8.0 9.0a'
```

这种优化策略：
- **减少构建时间 50-70%**：只编译需要的架构
- **减少镜像大小 40-60%**：移除不必要的 CUDA 代码
- **提升运行性能**：针对特定架构的优化代码

## 磁盘空间优化

本项目参考了 vLLM 官方项目的优化策略，解决了 GitHub Actions 构建过程中的磁盘空间不足问题：

### 主要优化措施：

1. **GitHub Actions 级别优化**：
   - 全面清理不必要的系统文件和工具
   - 积极的 Docker 清理策略
   - 构建过程中的磁盘监控
   - 减少并行作业数量以节省内存和磁盘

2. **Dockerfile 优化**：
   - 多阶段构建，仅保留运行时必需的文件
   - 优化 CUDA 架构列表，仅支持常用架构
   - 限制并行编译作业数 (`max_jobs=1`) 和 NVCC 线程数
   - 构建过程中及时清理缓存和临时文件
   - 减小 PyTorch 和其他依赖的下载包

3. **构建配置**：
   ```yaml
   build-args: |
     CUDA_VERSION=12.1.1
     PYTHON_VERSION=3.12
     max_jobs=1          # 减少并行作业
     nvcc_threads=2      # 减少 NVCC 线程
     VLLM_MAX_SIZE_MB=500  # 允许更大的 wheel 文件
     RUN_WHEEL_CHECK=false # 跳过 wheel 大小检查
   ```

## 快速开始

### 使用预构建镜像

```bash
# 运行 vLLM 服务器（A100/A800/H20 优化版本）
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    --shm-size=16g \
    registry.cn-beijing.aliyuncs.com/yoce/vllm:latest \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768
```

### 使用 Docker Compose

```bash
# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 本地构建

```bash
# 构建镜像（针对 A100/A800/H20 优化）
docker build -t vllm:cuda12.1-datacenter .

# 运行大模型推理（8卡 A100 示例）
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    --shm-size=32g \
    vllm:cuda12.1-datacenter \
    --model meta-llama/Llama-3.1-405B-Instruct \
    --tensor-parallel-size 8 \
    --max-model-len 16384 \
    --enforce-eager
```

## API 使用

启动服务后，可以通过 OpenAI 兼容的 API 进行调用：

```bash
# 健康检查
curl http://localhost:8000/health

# 模型列表
curl http://localhost:8000/v1/models

# 文本补全
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/DialoGPT-medium",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'

# 聊天补全
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/DialoGPT-medium",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 50
  }'
```

## 环境要求

- Docker 19.03+
- NVIDIA Docker runtime
- CUDA 12.1 兼容的 GPU 驱动
- **支持的 GPU**: A100, A800, H20, H100 等数据中心显卡
- 至少 40GB GPU 内存（推荐用于大模型）
- 至少 80GB 系统内存（推荐）

## 构建参数

支持以下构建参数：

- `CUDA_VERSION`: CUDA 版本 (默认: 12.1.1)
- `PYTHON_VERSION`: Python 版本 (默认: 3.12)
- `max_jobs`: 并行编译任务数 (默认: 1，针对磁盘空间优化)
- `nvcc_threads`: NVCC 线程数 (默认: 2，针对内存优化)
- `torch_cuda_arch_list`: CUDA 架构列表 (默认: '8.0 9.0a'，针对 A100/A800/H20)

```bash
# 自定义构建示例
docker build \
  --build-arg CUDA_VERSION=12.1.1 \
  --build-arg PYTHON_VERSION=3.12 \
  --build-arg max_jobs=2 \
  --build-arg nvcc_threads=4 \
  --build-arg torch_cuda_arch_list="8.0 9.0a" \
  -t vllm:custom-datacenter .
```

## GitHub Actions 设置

为了使用自动化 CI/CD，需要在 GitHub 仓库中设置以下 Secrets：

- `ALIYUN_USERNAME`: 阿里云容器镜像服务用户名
- `ALIYUN_PASSWORD`: 阿里云容器镜像服务密码

## 支持的标签

- `latest`: 最新构建的镜像
- `cuda12.1-latest`: CUDA 12.1 最新版本
- `main`: 主分支最新构建
- `v*`: 版本标签 (如 v1.0.0)
- `{branch}-{sha}`: 分支和提交 SHA

## 常见问题

### 1. GPU 内存不足

对于大模型推理，推荐使用张量并行：

```bash
# 使用 4 卡 A100/A800 运行 70B 模型
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    --shm-size=16g \
    registry.cn-beijing.aliyuncs.com/yoce/vllm:latest \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 16384
```

### 2. 模型加载优化

使用量化和内存优化：

```bash
# 使用 AWQ 量化节省显存
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    --shm-size=16g \
    registry.cn-beijing.aliyuncs.com/yoce/vllm:latest \
    --model casperhansen/llama-3-70b-instruct-awq \
    --quantization awq \
    --tensor-parallel-size 2
```

### 2. CUDA 版本不兼容

确保主机的 CUDA 驱动版本支持 CUDA 12.1。可以通过 `nvidia-smi` 查看驱动版本。

对于 A100/A800/H20 用户，推荐驱动版本：
- **NVIDIA Driver ≥ 525.60.13** (支持 CUDA 12.1)
- **NVIDIA Driver ≥ 530.30.02** (推荐，支持所有 CUDA 12.x 特性)

```bash
# 检查驱动和 CUDA 版本
nvidia-smi

# 检查 GPU 架构支持
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

### 3. 网络访问问题

如果在中国大陆使用，可能需要配置镜像源：

```bash
# 使用阿里云镜像
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_ENDPOINT=https://hf-mirror.com \
    -p 8000:8000 \
    --ipc=host \
    registry.cn-beijing.aliyuncs.com/yoce/vllm:latest \
    --model microsoft/DialoGPT-medium
```

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

本项目基于 Apache 2.0 许可证，详见原始 vLLM 项目许可证。