#!/usr/bin/env bash
# 常驻启动脚本（无 systemd 的容器环境）：前台循环，用 nohup 调用本脚本以脱离 SSH 会话。
# 启动：nohup bash /root/weights_bundle/deploy_api/deploy/run.sh > /root/weights_bundle/deploy_api/serve.log 2>&1 &
set -uo pipefail

ROOT=/root/weights_bundle
STOP_FLAG="$ROOT/deploy_api/.stop"
cd "$ROOT"

rm -f "$STOP_FLAG"

if [ -f "$ROOT/deploy_api/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT/deploy_api/.env"
  set +a
fi

while true; do
  /root/miniconda3/envs/uly/bin/python -m deploy_api.serve --host 0.0.0.0 --port 8000
  rc=$?
  echo "[$(date '+%F %T')] serve exited rc=$rc" >> "$ROOT/deploy_api/serve.log"
  if [ -f "$STOP_FLAG" ]; then
    echo "[$(date '+%F %T')] stop flag set, exiting" >> "$ROOT/deploy_api/serve.log"
    break
  fi
  echo "[$(date '+%F %T')] restart in 5s" >> "$ROOT/deploy_api/serve.log"
  sleep 5
done
