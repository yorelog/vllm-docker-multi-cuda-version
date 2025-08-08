#!/bin/bash

# vLLM Docker 构建和测试脚本
# 使用方法: ./test-build.sh [选项]

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 默认参数
IMAGE_NAME="vllm:cuda12.1-test"
CUDA_VERSION="12.1.1"
PYTHON_VERSION="3.12"
MAX_JOBS="2"
NVCC_THREADS="4"
TEST_MODEL="microsoft/DialoGPT-small"  # 使用小模型进行测试

print_usage() {
    echo "使用方法: $0 [选项]"
    echo "选项:"
    echo "  -h, --help           显示帮助信息"
    echo "  -b, --build-only     仅构建镜像，不运行测试"
    echo "  -t, --test-only      仅运行测试（假设镜像已存在）"
    echo "  -j, --jobs NUM       设置并行编译任务数 (默认: $MAX_JOBS)"
    echo "  -n, --nvcc-threads NUM 设置NVCC线程数 (默认: $NVCC_THREADS)"
    echo "  -m, --model MODEL    设置测试模型 (默认: $TEST_MODEL)"
    echo "  --clean              清理构建缓存"
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "检查系统要求..."
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装"
        exit 1
    fi
    
    # 检查NVIDIA Docker运行时
    if ! docker info 2>/dev/null | grep -q nvidia; then
        log_warn "NVIDIA Docker运行时可能未正确配置"
    fi
    
    # 检查GPU
    if ! command -v nvidia-smi &> /dev/null; then
        log_warn "nvidia-smi 未找到，可能没有NVIDIA GPU"
    else
        log_info "GPU 信息:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
    fi
}

build_image() {
    log_info "开始构建 Docker 镜像..."
    log_info "镜像名称: $IMAGE_NAME"
    log_info "CUDA 版本: $CUDA_VERSION"
    log_info "Python 版本: $PYTHON_VERSION"
    log_info "并行任务数: $MAX_JOBS"
    log_info "NVCC 线程数: $NVCC_THREADS"
    
    # 记录构建开始时间
    BUILD_START=$(date +%s)
    
    docker build \
        --build-arg CUDA_VERSION="$CUDA_VERSION" \
        --build-arg PYTHON_VERSION="$PYTHON_VERSION" \
        --build-arg max_jobs="$MAX_JOBS" \
        --build-arg nvcc_threads="$NVCC_THREADS" \
        -t "$IMAGE_NAME" \
        . || {
        log_error "镜像构建失败"
        exit 1
    }
    
    # 计算构建时间
    BUILD_END=$(date +%s)
    BUILD_TIME=$((BUILD_END - BUILD_START))
    
    log_info "镜像构建完成，耗时: ${BUILD_TIME}s"
    
    # 显示镜像信息
    log_info "镜像信息:"
    docker images "$IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
}

test_image() {
    log_info "开始测试镜像..."
    
    # 检查镜像是否存在
    if ! docker images "$IMAGE_NAME" --format "{{.Repository}}:{{.Tag}}" | grep -q "^${IMAGE_NAME}$"; then
        log_error "镜像 $IMAGE_NAME 不存在，请先构建镜像"
        exit 1
    fi
    
    # 测试容器启动
    log_info "测试容器启动..."
    CONTAINER_NAME="vllm-test-$(date +%s)"
    
    # 启动容器
    docker run -d \
        --name "$CONTAINER_NAME" \
        --runtime nvidia \
        --gpus all \
        -p 8001:8000 \
        --ipc=host \
        "$IMAGE_NAME" \
        --model "$TEST_MODEL" \
        --host 0.0.0.0 \
        --port 8000 || {
        log_error "容器启动失败"
        exit 1
    }
    
    log_info "容器已启动，名称: $CONTAINER_NAME"
    
    # 等待服务启动
    log_info "等待服务启动..."
    TIMEOUT=300  # 5分钟超时
    START_TIME=$(date +%s)
    
    while true; do
        CURRENT_TIME=$(date +%s)
        if [ $((CURRENT_TIME - START_TIME)) -gt $TIMEOUT ]; then
            log_error "服务启动超时"
            docker logs "$CONTAINER_NAME"
            docker stop "$CONTAINER_NAME" 2>/dev/null || true
            docker rm "$CONTAINER_NAME" 2>/dev/null || true
            exit 1
        fi
        
        if curl -s http://localhost:8001/health >/dev/null 2>&1; then
            log_info "服务启动成功"
            break
        fi
        
        echo -n "."
        sleep 5
    done
    echo
    
    # 测试API
    log_info "测试API接口..."
    
    # 测试健康检查
    if curl -s http://localhost:8001/health | grep -q "ok"; then
        log_info "✓ 健康检查通过"
    else
        log_error "✗ 健康检查失败"
    fi
    
    # 测试模型列表
    if curl -s http://localhost:8001/v1/models | grep -q "$TEST_MODEL"; then
        log_info "✓ 模型列表获取成功"
    else
        log_error "✗ 模型列表获取失败"
    fi
    
    # 测试文本生成
    log_info "测试文本生成..."
    RESPONSE=$(curl -s http://localhost:8001/v1/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$TEST_MODEL\", \"prompt\": \"Hello\", \"max_tokens\": 10}")
    
    if echo "$RESPONSE" | grep -q "choices"; then
        log_info "✓ 文本生成测试通过"
    else
        log_error "✗ 文本生成测试失败"
        echo "响应: $RESPONSE"
    fi
    
    # 清理容器
    log_info "清理测试容器..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
    
    log_info "测试完成"
}

clean_cache() {
    log_info "清理 Docker 构建缓存..."
    docker builder prune -f
    docker system prune -f
    log_info "缓存清理完成"
}

# 解析命令行参数
BUILD_ONLY=false
TEST_ONLY=false
CLEAN_CACHE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_usage
            exit 0
            ;;
        -b|--build-only)
            BUILD_ONLY=true
            shift
            ;;
        -t|--test-only)
            TEST_ONLY=true
            shift
            ;;
        -j|--jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        -n|--nvcc-threads)
            NVCC_THREADS="$2"
            shift 2
            ;;
        -m|--model)
            TEST_MODEL="$2"
            shift 2
            ;;
        --clean)
            CLEAN_CACHE=true
            shift
            ;;
        *)
            log_error "未知选项: $1"
            print_usage
            exit 1
            ;;
    esac
done

# 主执行流程
main() {
    log_info "vLLM Docker 构建和测试脚本"
    log_info "================================"
    
    if [ "$CLEAN_CACHE" = true ]; then
        clean_cache
        exit 0
    fi
    
    check_requirements
    
    if [ "$TEST_ONLY" = false ]; then
        build_image
    fi
    
    if [ "$BUILD_ONLY" = false ]; then
        test_image
    fi
    
    log_info "所有操作完成!"
}

# 运行主函数
main
