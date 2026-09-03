#!/bin/bash
set -e

source train-roma/con_to_web.sh


# ==========================================
# 1. 引擎版本（Critical）
# ==========================================
export VLLM_USE_V1=1

# ==========================================
# 2. vLLM-Ascend 特定优化
# ==========================================
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# ==========================================
# 3. 内存优化（解决碎片）
# ==========================================
export ACL_MEM_ALLOW_REUSE=1
export ASCEND_PYTORCH_ACL_ALLOCATOR_CONF=enable_single_stream_pool:True
export TASK_QUEUE_ENABLE=1

# ==========================================
# 4. 通信优化（解决超时）
# ==========================================
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=3600
export HCCL_ENABLE_FAULT_DETECTION=1
export HCCL_DETERMINISTIC=false

# ==========================================
# 5. 算子编译优化
# ==========================================
export TE_PARALLEL_COMPILER=8

## 可调整
export ROOT_PATH=$(pwd)
export MOXING_PATH="${ROOT_PATH}/../../data/moxing"

## MODEL_PATH
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="/home/ma-user/work/Qwen3-4B-Instruct-2507"
    echo "Set model path: ${MODEL_PATH}"
fi

## ======== 启动 ASCEND 环境 ==============
source /usr/local/Ascend/driver/bin/setenv.bash
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH

source /usr/local/Ascend/cann/ascend-toolkit/set_env.sh
source /usr/local/Ascend/cann/nnal/atb/set_env.sh

export MY_IP=$(hostname -I | awk '{print $1}')
echo "MyIP: ${MY_IP}"
export no_proxy="${MY_IP},${no_proxy}"
echo "no_proxy=${no_proxy}"

## =======  执行资源 ==============
export PATH=$PATH:/root/miniconda3/envs/agent_flow/bin

# ## 模型迁移（所有节点都需要访问模型）
# if [ -e "/cache/model" ]; then
#     echo "/cache/model文件已存在"
# else
#     echo "/cache/model文件不存在，需要迁移"
#     cd ${MOXING_PATH}
#     python mox_copy.py --src-dir ${MODEL_PATH} --target-dir /cache/model
# fi

# model_name=$(basename "${MODEL_PATH}")
# model_path="${ROOT_PATH}/${model_name}"
# echo "fetch ${model_path} from /cache/model" 
# ln -sf /cache/model "${model_path}"
# echo "ln -sf from [/cache/model] to [${model_path}]"

## 安装agentflow（所有节点）
echo "正在安装 agentflow..."
cd "${ROOT_PATH}/agentflow"
python -m pip install --user --no-deps -e . --no-build-isolation
cd ..

# ==========================================
# 多节点分布式配置（ModelArts环境变量）
# ==========================================
NODE_RANK=${VC_TASK_INDEX:-0}

# ==========================================
# 将云道运行变量显式导出给 config.yaml 使用
# ==========================================
export NNODES=${MA_NUM_HOSTS:-1}
export N_GPUS=${MA_NUM_GPUS:-8}


# 从VC_WORKER_HOSTS获取主节点地址（第一个域名即为Head节点）
MASTER_ADDR=$(echo "${VC_WORKER_HOSTS}" | cut -d',' -f1)

echo "======================================"
echo "多节点配置:"
echo "  NNODES: ${NNODES}"
echo "  NODE_RANK: ${NODE_RANK}"
echo "  NPUS_PER_NODE: ${N_GPUS}"
echo "  MASTER_ADDR: ${MASTER_ADDR}"
echo "======================================"

# ==========================================
# 启动Ray集群（所有节点）
# ==========================================
export RAY_ulimit_nofile=65536
ulimit -n 65536

if [ "${NODE_RANK}" -eq 0 ]; then
    echo "【Head节点】启动Ray Head..."
    ray stop --force --grace-period 60 2>/dev/null || true
    sleep 3
    
    ray start --head \
        --node-ip-address="${MY_IP}" \
        --port=6379 \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265 \
        --resources='{"NPU": '${N_GPUS}'}'
    
    echo "Ray Head已启动，等待Worker节点加入..."
    
    # 等待所有节点加入（通过ray status检查Alive节点数）
    EXPECTED_NODES=${NNODES}
    for i in $(seq 1 120); do
        sleep 5
        echo "引入Python获取节点"
        which python
        
        ALIVE_NODES=$(python -c "
import ray, sys
try:
    ray.init(address='${MY_IP}:6379', ignore_reinit_error=True)
    alive = sum(1 for n in ray.nodes() if n.get('Alive', False))
    print(alive)
    ray.shutdown()
except Exception as e:
    print(e)
")
        echo "结束Python获取流程"

        if [ -z "$ALIVE_NODES" ]; then
            ALIVE_NODES=0
        fi
        
        
        echo "[$i/120] 等待Ray集群就绪... 当前节点数: ${ALIVE_NODES}/${EXPECTED_NODES}"
        
        if [ "${ALIVE_NODES}" -ge "${EXPECTED_NODES}" ] 2>/dev/null; then
            echo "Ray集群已就绪！所有 ${EXPECTED_NODES} 个节点已加入。"
            ray status
            break
        fi
        
        if [ "$i" -eq 120 ]; then
            echo "警告：Ray集群未完全就绪，但继续执行..."
            ray status || true
        fi
    done
    
    # ==========================================
    # Head节点：启动所有服务
    # ==========================================
    export TIME_STR=$(date +"%y%m%d-%H%M%S")
    export LOG_DIR_PREFIX="/home/ma-user/modelarts/log/${TIME_STR}_RL"
    mkdir -p ${LOG_DIR_PREFIX}
    echo "Create: ${LOG_DIR_PREFIX}"
    
    # 导出Ray地址供子脚本使用
    export RAY_ADDRESS="${MY_IP}:6379"
    
    echo "Run all scripts on Head node!"
    bash train-roma/con_to_llm.sh > "${LOG_DIR_PREFIX}/con_to_llm.log" 2>&1 &
    bash train-roma/serve_with_logs.sh > "${LOG_DIR_PREFIX}/serve_with_logs.log" 2>&1 &
    bash train-roma/train_with_logs.sh > "${LOG_DIR_PREFIX}/train_with_logs.log" 2>&1 &
    
    echo "已启动服务，PID: $(jobs -p)"
    echo "直接 kill 此脚本即可终止所有服务"
    
    wait
    
else
    echo "【Worker节点】加入Ray集群..."
    ray stop --force --grace-period 60 2>/dev/null || true
    sleep 3
    
    # Worker节点循环尝试连接，直到成功
    CONNECTED=false
    for i in $(seq 1 120); do
        echo "[$i/120] 尝试连接Ray Head: ${MASTER_ADDR}:6379"
        if ray start \
            --address="${MASTER_ADDR}:6379" \
            --node-ip-address="${MY_IP}" \
            --resources='{"NPU": '${N_GPUS}'}' 2>/dev/null; then
            
            echo "成功加入Ray集群！"
            ray status || true
            CONNECTED=true
            break
        fi
        sleep 10
    done
    
    if [ "$CONNECTED" = false ]; then
        echo "错误：无法连接到Ray集群，退出..."
        exit 1
    fi
    
    # Worker节点保持存活（等待训练结束，ModelArts终止时会发信号触发trap）
    echo "Worker节点已就绪，保持运行..."
    tail -f /dev/null
fi