# vLLM Docker with CUDA 12.1

基于 [vllm-project/vllm](https://github.com/vllm-project/vllm) 的 Docker 镜像，专门针对 CUDA 12.1 优化构建。

## 功能特性

- 基于 vLLM 最新主分支构建
- 支持 CUDA 12.1
- 针对阿里云容器镜像服务优化
- 支持 OpenAI 兼容 API
- 自动化 CI/CD 构建和推送

## 快速开始

### 使用预构建镜像

```bash
# 拉取镜像
docker pull registry.cn-beijing.aliyuncs.com/yoce/vllm:latest

# 运行 vLLM 服务器
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    registry.cn-beijing.aliyuncs.com/yoce/vllm:latest \
    --model microsoft/DialoGPT-medium
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
# 构建镜像
docker build -t vllm:cuda12.1-local .

# 运行容器
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm:cuda12.1-local \
    --model microsoft/DialoGPT-medium
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
- CUDA 12.1 兼容的 GPU
- 至少 8GB GPU 内存（根据模型大小而定）

## 构建参数

支持以下构建参数：

- `CUDA_VERSION`: CUDA 版本 (默认: 12.1.1)
- `PYTHON_VERSION`: Python 版本 (默认: 3.12)
- `max_jobs`: 并行编译任务数 (默认: 2)
- `nvcc_threads`: NVCC 线程数 (默认: 8)

```bash
docker build \
  --build-arg CUDA_VERSION=12.1.1 \
  --build-arg PYTHON_VERSION=3.12 \
  --build-arg max_jobs=4 \
  --build-arg nvcc_threads=4 \
  -t vllm:custom .
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

减少模型大小或使用量化模型：

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    registry.cn-beijing.aliyuncs.com/yoce/vllm:latest \
    --model microsoft/DialoGPT-small \
    --quantization awq
```

### 2. CUDA 版本不兼容

确保主机的 CUDA 驱动版本支持 CUDA 12.1。可以通过 `nvidia-smi` 查看驱动版本。

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