#!/usr/bin/env bash
# Run the existing service launcher with timing enabled for the first 100 val samples.
# Example: bash train-roma/profile_val_timing.sh bash train-roma/run_distribute_train.sh
set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "Usage: bash train-roma/profile_val_timing.sh <existing launch command...>" >&2
    exit 2
fi

export AGENTFLOW_PROFILE_VAL_TIMING=1
export AGENTFLOW_PROFILE_VAL_ONLY=1
export AGENTFLOW_PROFILE_VAL_LIMIT="${AGENTFLOW_PROFILE_VAL_LIMIT:-100}"
export AGENTFLOW_PROFILE_VAL_TIMEOUT_S="${AGENTFLOW_PROFILE_VAL_TIMEOUT_S:-3600}"
export AGENTFLOW_PROFILE_OUTPUT_DIR="${AGENTFLOW_PROFILE_OUTPUT_DIR:-rollout_data/val_timing}"

exec "$@"
