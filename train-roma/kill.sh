#!/bin/bash

# ============================================================
# AgentFlow 训练：彻底停止所有相关进程
# ============================================================

set +e

REPO="/home/ma-user/work/code-rl/new/AgentFlow-distributed-GRPO-z"
LOG_ROOT="/home/ma-user/modelarts/log"

echo "=================================================="
echo "开始彻底清理 AgentFlow 训练进程"
echo "=================================================="

# 当前这个 stop shell 自己的进程组，绝对不能杀
SELF_PID=$$
SELF_PGID=$(ps -o pgid= -p $$ | tr -d ' ')

echo "[INFO] stop PID=$SELF_PID PGID=$SELF_PGID"


# ============================================================
# 1. 收集所有 AgentFlow / 训练相关 PID
# ============================================================

collect_pids() {
    {
        pgrep -f 'train-roma/run_distribute_train.sh'
        pgrep -f 'train-roma/run_train.sh'
        pgrep -f 'train-roma/train_with_logs.sh'
        pgrep -f 'train-roma/serve_with_logs.sh'
        pgrep -f 'train-roma/rollout.py'
        pgrep -f 'train-roma/train_agent.py'

        pgrep -f 'agentflow.verl'
        pgrep -f 'agentflow.runner'

        pgrep -f 'TaskRunner'
        pgrep -f 'WorkerDict'

        # 孤儿 split：之前你的 111290 就是这种
        pgrep -f 'split.*modelarts/log/.*_RL'

        # Ray
        pgrep -f 'raylet'
        pgrep -f 'gcs_server'
        pgrep -f 'ray/dashboard'
        pgrep -f 'runtime_env/agent'

        # LLM SSH tunnel
        pgrep -f 'train-roma/con_to_llm.sh'
        pgrep -f 'ssh.*19996.*19997.*19998'
    } 2>/dev/null | sort -nu
}


# ============================================================
# 2. 根据 PID 找到对应 PGID
#    不只是杀 PID，而是整个进程组一起杀
# ============================================================

collect_pgids() {
    for pid in $(collect_pids); do

        [ "$pid" = "$SELF_PID" ] && continue
        [ "$pid" = "1" ] && continue

        pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')

        [ -z "$pgid" ] && continue
        [ "$pgid" = "$SELF_PGID" ] && continue
        [ "$pgid" = "1" ] && continue

        echo "$pgid"
    done | sort -nu
}


echo
echo "===== 当前匹配到的相关进程 ====="

ps -eo pid,ppid,pgid,sid,lstart,cmd \
| grep -Ei \
'run_distribute_train|run_train.sh|train_with_logs|serve_with_logs|rollout.py|train_agent.py|agentflow.verl|agentflow.runner|TaskRunner|WorkerDict|raylet|gcs_server|split.*modelarts/log/.*_RL|ssh.*19996.*19997.*19998' \
| grep -v grep


# ============================================================
# 3. 第一轮：TERM 整个进程组
# ============================================================

PGIDS=$(collect_pgids)

echo
echo "===== TERM 进程组 ====="

for pgid in $PGIDS; do
    echo "kill -TERM -$pgid"
    kill -TERM -- "-$pgid" 2>/dev/null
done

sleep 5


# ============================================================
# 4. 第二轮：KILL 仍然存在的整个进程组
# ============================================================

PGIDS=$(collect_pgids)

echo
echo "===== KILL 残留进程组 ====="

for pgid in $PGIDS; do
    echo "kill -KILL -$pgid"
    kill -KILL -- "-$pgid" 2>/dev/null
done

sleep 2


# ============================================================
# 5. 单独补杀漏网 PID
#    防止孤儿进程已经换了 PGID
# ============================================================

echo
echo "===== 补杀漏网 PID ====="

for pid in $(collect_pids); do

    [ "$pid" = "$SELF_PID" ] && continue
    [ "$pid" = "1" ] && continue

    echo "kill -9 $pid"
    kill -9 "$pid" 2>/dev/null
done


# ============================================================
# 6. Ray 官方清理
# ============================================================

echo
echo "===== 清理 Ray ====="

ray stop --force 2>/dev/null || true

pkill -9 -f '[r]aylet' 2>/dev/null || true
pkill -9 -f '[g]cs_server' 2>/dev/null || true
pkill -9 -f '[r]ay/dashboard' 2>/dev/null || true
pkill -9 -f '[r]untime_env/agent' 2>/dev/null || true


# ============================================================
# 7. 专门清孤儿 split
# ============================================================

echo
echo "===== 清理孤儿日志 split ====="

ps -eo pid,ppid,cmd \
| awk '/split .*\/home\/ma-user\/modelarts\/log\/.*_RL/ && !/awk/ {print $1}' \
| while read pid; do
    echo "kill orphan split PID=$pid"
    kill -9 "$pid" 2>/dev/null
done


# ============================================================
# 8. 清理 SSH tunnel
# ============================================================

echo
echo "===== 清理 19996/19997/19998 tunnel ====="

pkill -9 -f '[s]sh.*19996.*19997.*19998' 2>/dev/null || true


# ============================================================
# 9. 删除 PID 文件
# ============================================================

rm -f "$REPO/train_main.pid"


# ============================================================
# 10. 最终检查
# ============================================================

sleep 2

echo
echo "=================================================="
echo "最终残留检查"
echo "=================================================="

REMAINING=$(ps -eo pid,ppid,pgid,sid,cmd \
| grep -Ei \
'run_distribute_train|run_train.sh|train_with_logs|serve_with_logs|rollout.py|train_agent.py|agentflow.verl|agentflow.runner|TaskRunner|WorkerDict|raylet|gcs_server|split.*modelarts/log/.*_RL' \
| grep -v grep)

if [ -z "$REMAINING" ]; then
    echo "[OK] 没有发现训练相关残留进程"
else
    echo "[WARNING] 仍有残留："
    echo "$REMAINING"
fi

echo
echo "===== Ray ====="

ray status 2>/dev/null || echo "[OK] Ray 已停止"

echo
echo "===== 19996 / 19997 / 19998 ====="

PORTS=$(netstat -lntp 2>/dev/null | grep -E '19996|19997|19998')

if [ -z "$PORTS" ]; then
    echo "[OK] 19996/19997/19998 无监听"
else
    echo "[WARNING] 仍有监听："
    echo "$PORTS"
fi

echo
echo "=================================================="
echo "清理完成"
echo "=================================================="