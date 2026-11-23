# 部署步骤（4090）

> 代码目录 `deploy_api/`，推理环境 `python=/root/miniconda3/envs/uly/bin/python`。
>
> **权重位置**（当前未解包 tar，直接指向真实源路径）：
> - base：`/root/models/sam3/sam3.pt`
> - λ 混合：`/root/runs/sam3ft-muntrungca_lora_r6/final/current_r6_ep14/model_only_r6_ep14.pth`
>
> 也可解包 `/root/weights_bundle/sam3ft_weights_r6_ep14.tar` 后用 `--base/--model-only` 指向 `/root/weights_bundle/`。

## 1. 上传代码

```bash
# 本地
scp -r deploy_api 4090:/root/weights_bundle/
```

## 2. 安装 HTTP 依赖（一次性）

```bash
ssh 4090 '/root/miniconda3/envs/uly/bin/pip install fastapi "uvicorn[standard]" python-multipart'
```

## 3. 配置密钥

```bash
ssh 4090 'cat > /root/weights_bundle/deploy_api/.env <<EOF
DEPLOY_API_KEY=<换成真实密钥>
EOF
chmod 600 /root/weights_bundle/deploy_api/.env'
```

`.env` 已 `.gitignore` 忽略，绝不提交真实密钥。

## 4. 冒烟（先手动跑，验证模型能加载）

```bash
ssh -t 4090 'cd /root/weights_bundle && DEPLOY_API_KEY=smoke /root/miniconda3/envs/uly/bin/python -m deploy_api.serve --host 127.0.0.1 --port 8000'
```

另开终端：

```bash
ssh 4090 'curl -s http://127.0.0.1:8000/healthz'
ssh 4090 'curl -s -H "X-API-Key: smoke" -F "prompt=blackheads" -F "file=@/path/to/skin.jpg" http://127.0.0.1:8000/segment'
ssh 4090 'curl -s -H "X-API-Key: smoke" -F "prompt=acne" -F "file=@/path/to/skin.jpg" http://127.0.0.1:8000/segment'   # λ=0.1 分支
```

## 5. 常驻（容器无 systemd → nohup + 自动重启循环）

这台 4090 是 **Docker 容器**（PID 1 为 `docker-init`，无 systemd/supervisord），用 `nohup` 脱离 SSH 会话 + `run.sh` 崩溃自动重启。

```bash
# 启动（会先等模型加载 ~40s，之后 /readyz 才就绪）
ssh 4090 'nohup bash /root/weights_bundle/deploy_api/deploy/run.sh > /root/weights_bundle/deploy_api/serve.log 2>&1 &
sleep 1; cat /root/weights_bundle/deploy_api/serve.pid 2>/dev/null'

# 停止
ssh 4090 'bash /root/weights_bundle/deploy_api/deploy/stop.sh'

# 查看日志 / 状态
ssh 4090 'tail -f /root/weights_bundle/deploy_api/serve.log'
```

> ⚠️ 限制：进程在 SSH 断开后仍存活、崩溃后自动重启；但**容器整体重启时不会自启**（容器启动命令固定为 `tail -f /dev/null`）。容器重启后需重跑上面启动命令，或把 `run.sh` 加进容器的启动脚本。

## 6. 对外暴露 + HTTPS（宿主机侧，需 docker/host 权限）

> 完整可复用模板见 [`host/`](host/)：`docker-compose.yml` + `nginx.conf` + `README.md`（API 容器化 + nginx 自动 HTTPS + 健康检查 + 崩溃自启）。

服务绑定 `0.0.0.0:8000`（容器内所有接口）。客户端要够到它，需在**宿主机**发布端口并反代（API key 不能明文过网，务必上 HTTPS）：

```bash
# 宿主机：发布端口（重启容器时加 -p 8000:8000，或改 docker compose）
docker run ... -p 8000:8000 ...

# 宿主机：nginx 反代，终结 HTTPS + 保留 X-API-Key 头
```

```nginx
location /segment/ {
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header X-API-Key $http_x_api_key;
    proxy_read_timeout 120s;
}
```
