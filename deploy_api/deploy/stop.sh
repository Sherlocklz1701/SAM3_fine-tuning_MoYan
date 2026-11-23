#!/usr/bin/env bash
# 停止服务：置停止标志 + 结束 serve 与 run.sh 循环。
set -uo pipefail
ROOT=/root/weights_bundle
touch "$ROOT/deploy_api/.stop"
pkill -f 'deploy_api.serve' 2>/dev/null || true
pkill -f 'deploy/run.sh' 2>/dev/null || true
echo "stop requested"
