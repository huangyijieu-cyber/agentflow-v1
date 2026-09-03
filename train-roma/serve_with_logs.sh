#!/bin/bash
# set -x
export AGENTOPS_AUTO_INIT=false

# Pub IP
PUBLIC_IP=127.0.0.1
echo "curl ip: ${PUBLIC_IP}"

# 1. 检查是否为空
if [ -z "$PUBLIC_IP" ]; then
  echo "No valid IP ($PUBLIC_IP) get!"
  exit 1
fi

# 2. 检查是否包含 "Failed" 字样（不区分大小写）
if echo "$PUBLIC_IP" | grep -qi "failed"; then
  echo "Failed to get public IP!"
  exit 1
fi

LAST_TWO_OCTETS=$(echo "$PUBLIC_IP" | awk -F'.' '{print $3"."$4}')

# --- Configuration Section ---
# 1. Define the log directory（加入节点标识）
NODE_ID=${VC_TASK_INDEX:-0}
LOG_DIR="${LOG_DIR_PREFIX}/task_logs/${PUBLIC_IP}_rank${NODE_ID}/serve_log"

# 2. Define the prefix for output files
LOG_PREFIX="serving_output_"

# 3. Define the maximum size of a single log file (1MB)
LOG_SIZE='1M'

# 4. Define the maximum number of log files to keep
MAX_LOG_FILES=5000

# 5. Read ASCEND_RT_VISIBLE_DEVICES from config.yaml if available
if [ -f "train-roma/config.yaml" ]; then
    ## NPU setting
    ASCEND_DEVICES=$(grep -m1 "ASCEND_RT_VISIBLE_DEVICES:" train-roma/config.yaml | sed "s/.*ASCEND_RT_VISIBLE_DEVICES: *['\"]\\([^'\"]*\\)['\"].*/\\1/")
    if [ -n "$ASCEND_DEVICES" ]; then
        export ASCEND_RT_VISIBLE_DEVICES="$ASCEND_DEVICES"
        echo "Setting ASCEND_RT_VISIBLE_DEVICES=$ASCEND_DEVICES from config.yaml"
    fi 
fi

# 6. The Python command you want to run
PYTHON_COMMAND="python train-roma/rollout.py"

# --- Function: calculate digit length of a number ---
suffix_length() {
    echo "${#1}"
}

# Remove and recreate the log directory
rm -rf $LOG_DIR
mkdir -p $LOG_DIR

# Calculate suffix digit length
SUFFIX_DIGITS=$(suffix_length $((MAX_LOG_FILES - 1)))

echo "Starting the task... Log files will use $SUFFIX_DIGITS-digit suffixes"

# 检查Ray集群（vLLM多机后端可能需要Ray）
echo "检查Ray集群状态..."
ray status || echo "警告：Ray集群状态异常"

# Use split with dynamic suffix length
PYTHONUNBUFFERED=1 $PYTHON_COMMAND 2>&1 | \
    split -b "$LOG_SIZE" -d -a "$SUFFIX_DIGITS" - "$LOG_DIR/$LOG_PREFIX"

SPLIT_EXIT_CODE=${PIPESTATUS[1]}

if [ $SPLIT_EXIT_CODE -eq 0 ]; then
    echo "Task completed successfully."
else
    echo "Error: The task or log splitting failed with exit code $SPLIT_EXIT_CODE."
    exit $SPLIT_EXIT_CODE
fi

# Clean up: keep only the newest MAX_LOG_FILES files
echo "Cleaning up old log files, keeping the latest $MAX_LOG_FILES..."
ls -1t "$LOG_DIR"/"$LOG_PREFIX"* 2>/dev/null | \
    tail -n +$((MAX_LOG_FILES + 1)) | \
    xargs rm -f

echo "Log files are saved in: $LOG_DIR"