#!/bin/bash

ROOT_DIR="/home/ma-user/work/code-rl/new/AgentFlow-distributed-GRPO-z"
PID_FILE="$ROOT_DIR/train_main.pid"
LOG_FILE="$ROOT_DIR/nohup_train.log"

cd "$ROOT_DIR" || exit 1


start_train() {
    echo "=============================="
    echo "启动训练"
    echo "=============================="

    nohup bash train-roma/run_distribute_train.sh \
        > "$LOG_FILE" 2>&1 < /dev/null &

    echo $! > "$PID_FILE"

    echo "训练主进程 PID: $!"
    echo "日志: $LOG_FILE"
    echo
    echo "查看日志："
    echo "tail -f $LOG_FILE"
}


stop_train() {
    echo "=============================="
    echo "停止训练"
    echo "=============================="

    # ==============================
    # 1. 优雅停止训练入口
    # ==============================

    pkill -TERM -f 'train-roma/run_distribute_train.sh' 2>/dev/null || true
    pkill -TERM -f 'train-roma/train_with_logs.sh' 2>/dev/null || true
    pkill -TERM -f 'train-roma/serve_with_logs.sh' 2>/dev/null || true
    pkill -TERM -f 'train-roma/rollout.py' 2>/dev/null || true
    pkill -TERM -f 'agentflow.verl' 2>/dev/null || true

    sleep 5


    # ==============================
    # 2. 强制清训练进程
    # ==============================

    pkill -9 -f 'train-roma/run_distribute_train.sh' 2>/dev/null || true
    pkill -9 -f 'train-roma/train_with_logs.sh' 2>/dev/null || true
    pkill -9 -f 'train-roma/serve_with_logs.sh' 2>/dev/null || true
    pkill -9 -f 'train-roma/rollout.py' 2>/dev/null || true

    pkill -9 -f 'agentflow.verl' 2>/dev/null || true
    pkill -9 -f 'agentflow.runner' 2>/dev/null || true
    pkill -9 -f 'TaskRunner' 2>/dev/null || true
    pkill -9 -f 'WorkerDict' 2>/dev/null || true


    # ==============================
    # 3. 清 Ray
    # ==============================

    ray stop --force 2>/dev/null || true

    pkill -9 -f 'raylet' 2>/dev/null || true
    pkill -9 -f 'gcs_server' 2>/dev/null || true
    pkill -9 -f 'ray/dashboard' 2>/dev/null || true
    pkill -9 -f 'runtime_env/agent' 2>/dev/null || true


    # ==============================
    # 4. 清 LLM SSH tunnel
    # ==============================

    pkill -9 -f 'train-roma/con_to_llm.sh' 2>/dev/null || true
    pkill -9 -f 'ssh.*19996.*19997.*19998' 2>/dev/null || true


    # ==============================
    # 5. 清 PID 文件
    # ==============================

    rm -f "$PID_FILE"

    sleep 2

    echo
    echo "=============================="
    echo "检查残留进程"
    echo "=============================="

    ps -ef | grep -Ei \
    'run_distribute_train|train_with_logs|serve_with_logs|rollout.py|agentflow.verl|agentflow.runner|TaskRunner|WorkerDict|raylet|gcs_server' \
    | grep -v grep || true

    echo
    echo "=============================="
    echo "检查 19996/19997/19998"
    echo "=============================="

    netstat -lntp 2>/dev/null | grep -E '19996|19997|19998' || true

    echo
    echo "停止完成。"
}


status_train() {
    echo "=============================="
    echo "训练相关进程"
    echo "=============================="

    ps -ef | grep -Ei \
    'run_distribute_train|train_with_logs|serve_with_logs|rollout.py|agentflow.verl|agentflow.runner|TaskRunner|WorkerDict|raylet|gcs_server' \
    | grep -v grep || true

    echo
    echo "=============================="
    echo "19996 / 19997 / 19998"
    echo "=============================="

    netstat -lntp 2>/dev/null | grep -E '19996|19997|19998' || true
}


case "$1" in
    start)
        start_train
        ;;
    stop)
        stop_train
        ;;
    restart)
        stop_train
        sleep 2
        start_train
        ;;
    status)
        status_train
        ;;
    log)
        tail -f "$LOG_FILE"
        ;;
    *)
        echo "Usage:"
        echo "  bash train_control.sh start"
        echo "  bash train_control.sh stop"
        echo "  bash train_control.sh restart"
        echo "  bash train_control.sh status"
        echo "  bash train_control.sh log"
        exit 1
        ;;
esac