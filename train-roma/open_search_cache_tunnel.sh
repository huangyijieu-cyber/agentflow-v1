#!/usr/bin/env bash
# Foreground SSH forward, one on every training node. Needs an authorized SSH key.
set -euo pipefail
_search_local_port="${SEARCH_CACHE_LOCAL_PORT:-8091}"
_search_remote_port="${SEARCH_SERVICE_PORT:-8091}"
exec ssh -N -T \
    -o BatchMode=yes \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -p "${SEARCH_CACHE_SSH_PORT:-31753}" \
    -L "127.0.0.1:${_search_local_port}:127.0.0.1:${_search_remote_port}" \
    "${SEARCH_CACHE_SSH_TARGET:-ma-user@7.150.11.99}"
