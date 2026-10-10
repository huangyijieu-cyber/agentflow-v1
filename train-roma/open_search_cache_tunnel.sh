#!/usr/bin/env bash
# Foreground tunnel using the same automatic SSH preparation as training.
set -euo pipefail
_search_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${SEARCH_CACHE_PYTHON:-python3}" \
    "${_search_repo_dir}/train-roma/search_cache_bootstrap.py" --manual-tunnel
