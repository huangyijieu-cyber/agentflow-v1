#!/bin/bash

## HD2 Resource: Note-VLLM-Qwen

## 打通本地 19996/19997/19998 到远程 7.194.184.142 对应端口
pem_path="pem/t00620714-hd2-ffn.pem"

ssh -i ${pem_path} \
    -p 32356 \
    -o StrictHostKeyChecking=no \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o UserKnownHostsFile=/dev/null \
    -N \
    -L 0.0.0.0:19996:127.0.0.1:19996 \
    -L 0.0.0.0:19997:127.0.0.1:19997 \
    -L 0.0.0.0:19998:127.0.0.1:19998 \
    ma-user@7.150.11.99
