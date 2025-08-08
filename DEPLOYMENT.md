# 部署指南

本文档提供了详细的部署指南，帮助你在不同环境中部署 vLLM Docker 镜像。

## 前置要求

### 硬件要求

- **GPU**: NVIDIA GPU，计算能力 7.0 或更高（如 V100, T4, RTX20xx, A100, L4, H100 等）
- **显存**: 至少 4GB（根据模型大小调整）
- **CPU**: 多核 CPU 推荐
- **内存**: 至少 16GB RAM
- **存储**: 至少 50GB 可用空间

### 软件要求

- **操作系统**: Linux（推荐 Ubuntu 20.04+）
- **Docker**: 19.03 或更高版本
- **NVIDIA Docker**: nvidia-container-toolkit
- **CUDA 驱动**: 支持 CUDA 12.1 的驱动版本（≥ 530.30.02）

## 环境配置

### 1. 安装 Docker

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 添加用户到 docker 组
sudo usermod -aG docker $USER
newgrp docker
```

### 2. 安装 NVIDIA Container Toolkit

```bash
# 添加 NVIDIA 软件源
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 安装
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 配置 Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 3. 验证安装

```bash
# 验证 GPU 可访问性
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi
```

## 部署方式

### 方式一：使用预构建镜像（推荐）

#### 快速启动

```bash
# 拉取镜像
docker pull registry.cn-beijing.aliyuncs.com/yoce/vllm:latest

# 运行服务
docker run -d \
    --name vllm-server \
    --runtime nvidia \
    --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    --restart unless-stopped \
    registry.cn-beijing.aliyuncs.com/yoce/vllm:latest \
    --model microsoft/DialoGPT-medium \
    --host 0.0.0.0 \
    --port 8000
```

#### 使用环境变量配置

```bash
docker run -d \
    --name vllm-server \
    --runtime nvidia \
    --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v /path/to/models:/models \
    -p 8000:8000 \
    --ipc=host \
    --restart unless-stopped \
    -e HUGGING_FACE_HUB_TOKEN=your_hf_token \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    registry.cn-beijing.aliyuncs.com/yoce/vllm:latest \
    --model /models/your-model \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8000
```

### 方式二：使用 Docker Compose

创建 `docker-compose.production.yml` 文件：

```yaml
version: '3.8'

services:
  vllm:
    image: registry.cn-beijing.aliyuncs.com/yoce/vllm:latest
    container_name: vllm-production
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0,1  # 指定使用的GPU
      - VLLM_WORKER_MULTIPROC_METHOD=spawn
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
      - /data/models:/models
      - ./logs:/logs
    command: >
      --model /models/your-model
      --tensor-parallel-size 2
      --host 0.0.0.0
      --port 8000
      --trust-remote-code
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

启动服务：

```bash
# 设置环境变量
export HF_TOKEN=your_huggingface_token

# 启动服务
docker-compose -f docker-compose.production.yml up -d

# 查看日志
docker-compose -f docker-compose.production.yml logs -f

# 停止服务
docker-compose -f docker-compose.production.yml down
```

### 方式三：使用 Kubernetes

创建 `vllm-deployment.yaml`：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-deployment
  namespace: ai-services
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      containers:
      - name: vllm
        image: registry.cn-beijing.aliyuncs.com/yoce/vllm:latest
        ports:
        - containerPort: 8000
        args:
          - "--model"
          - "microsoft/DialoGPT-medium"
          - "--host"
          - "0.0.0.0"
          - "--port"
          - "8000"
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "2"
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
        volumeMounts:
        - name: hf-cache
          mountPath: /root/.cache/huggingface
        - name: model-storage
          mountPath: /models
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: hf-cache
        persistentVolumeClaim:
          claimName: hf-cache-pvc
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-storage-pvc
      nodeSelector:
        accelerator: nvidia-tesla-v100  # 根据实际GPU类型调整

---
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
  namespace: ai-services
spec:
  selector:
    app: vllm
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

部署到 Kubernetes：

```bash
# 创建命名空间
kubectl create namespace ai-services

# 应用配置
kubectl apply -f vllm-deployment.yaml

# 查看状态
kubectl get pods -n ai-services
kubectl get services -n ai-services

# 查看日志
kubectl logs -f deployment/vllm-deployment -n ai-services
```

## 模型配置

### 支持的模型类型

- **文本生成**: GPT, LLaMA, Mistral, CodeLlama
- **对话模型**: ChatGLM, Baichuan, Qwen
- **代码生成**: CodeGeeX, StarCoder
- **多模态**: LLaVA, BLIP-2

### 模型下载和挂载

#### 预下载模型

```bash
# 创建模型目录
mkdir -p /data/models

# 使用 huggingface-cli 下载
pip install huggingface_hub
huggingface-cli download microsoft/DialoGPT-medium --local-dir /data/models/DialoGPT-medium

# 或使用 git lfs
git lfs clone https://huggingface.co/microsoft/DialoGPT-medium /data/models/DialoGPT-medium
```

#### 挂载本地模型

```bash
docker run -d \
    --name vllm-server \
    --runtime nvidia \
    --gpus all \
    -v /data/models:/models \
    -p 8000:8000 \
    --ipc=host \
    registry.cn-beijing.aliyuncs.com/yoce/vllm:latest \
    --model /models/DialoGPT-medium
```

## 性能优化

### GPU 内存优化

```bash
# 启用 KV 缓存量化
docker run -d \
    --name vllm-server \
    --runtime nvidia \
    --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    registry.cn-beijing.aliyuncs.com/yoce/vllm:latest \
    --model microsoft/DialoGPT-medium \
    --kv-cache-dtype fp8 \
    --quantization awq \
    --max-model-len 4096
```

### 多GPU 支持

```bash
# 张量并行
docker run -d \
    --name vllm-server \
    --runtime nvidia \
    --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    registry.cn-beijing.aliyuncs.com/yoce/vllm:latest \
    --model microsoft/DialoGPT-medium \
    --tensor-parallel-size 2
```

### 批处理优化

```bash
# 调整批处理大小
docker run -d \
    --name vllm-server \
    --runtime nvidia \
    --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    registry.cn-beijing.aliyuncs.com/yoce/vllm:latest \
    --model microsoft/DialoGPT-medium \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 256
```

## 监控和日志

### 健康检查

```bash
# 基本健康检查
curl http://localhost:8000/health

# 详细状态
curl http://localhost:8000/v1/models
```

### 日志收集

```bash
# 查看容器日志
docker logs -f vllm-server

# 使用日志驱动
docker run -d \
    --name vllm-server \
    --log-driver=json-file \
    --log-opt max-size=100m \
    --log-opt max-file=5 \
    ...
```

### Prometheus 监控

vLLM 内置了 Prometheus 指标支持：

```bash
# 启用指标
docker run -d \
    --name vllm-server \
    --runtime nvidia \
    --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    -p 9090:9090 \
    --ipc=host \
    registry.cn-beijing.aliyuncs.com/yoce/vllm:latest \
    --model microsoft/DialoGPT-medium \
    --enable-metrics \
    --metrics-port 9090
```

访问指标：`http://localhost:9090/metrics`

## 安全配置

### 网络安全

```bash
# 限制访问来源
docker run -d \
    --name vllm-server \
    --runtime nvidia \
    --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 127.0.0.1:8000:8000 \  # 只绑定本地接口
    --ipc=host \
    registry.cn-beijing.aliyuncs.com/yoce/vllm:latest \
    --model microsoft/DialoGPT-medium
```

### 资源限制

```bash
# 设置资源限制
docker run -d \
    --name vllm-server \
    --runtime nvidia \
    --gpus all \
    --memory=16g \
    --cpus=4.0 \
    --ulimit nofile=65536:65536 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    registry.cn-beijing.aliyuncs.com/yoce/vllm:latest \
    --model microsoft/DialoGPT-medium
```

## 故障排除

### 常见问题

1. **GPU 内存不足**
   ```bash
   # 查看 GPU 使用情况
   nvidia-smi
   
   # 减少模型大小或使用量化
   --quantization awq
   --kv-cache-dtype fp8
   ```

2. **模型下载失败**
   ```bash
   # 使用镜像站
   -e HF_ENDPOINT=https://hf-mirror.com
   ```

3. **容器启动失败**
   ```bash
   # 查看详细日志
   docker logs vllm-server
   
   # 检查 GPU 可用性
   docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.1-base nvidia-smi
   ```

### 性能调优

1. **内存优化**
   - 调整 `--max-model-len`
   - 使用 `--kv-cache-dtype fp8`
   - 启用 `--enable-chunked-prefill`

2. **吞吐量优化**
   - 调整 `--max-num-batched-tokens`
   - 增加 `--max-num-seqs`
   - 使用 `--tensor-parallel-size`

3. **延迟优化**
   - 减少 `--max-num-batched-tokens`
   - 使用 `--disable-log-requests`
   - 启用 `--enable-prefix-caching`

## 生产环境注意事项

1. **高可用性**
   - 使用负载均衡器
   - 部署多个实例
   - 实施健康检查

2. **数据持久化**
   - 挂载持久化存储
   - 备份模型文件
   - 配置日志轮转

3. **安全性**
   - 限制网络访问
   - 使用 HTTPS
   - 实施访问控制

4. **监控**
   - CPU、内存、GPU 使用率
   - 请求延迟和吞吐量
   - 错误率和可用性
