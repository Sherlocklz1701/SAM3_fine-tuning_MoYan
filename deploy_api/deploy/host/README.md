# 宿主机侧部署模板（docker-compose + nginx HTTPS）

把分割 API 容器化常驻，nginx 终结 HTTPS，客户端走 `https://<host>/segment`。
在**宿主机**上执行（把 `deploy_api/` 目录拷到宿主机后，进入 `deploy_api/deploy/host/`）。

## 前置：三样东西

### 1) 权重放到宿主机

从现有 4090 容器 `docker cp` 到本目录 `weights/`：

```bash
mkdir -p weights certs
docker cp 4090:/root/models/sam3/sam3.pt ./weights/
docker cp 4090:/root/runs/sam3ft-muntrungca_lora_r6/final/current_r6_ep14/model_only_r6_ep14.pth ./weights/
docker cp 4090:/root/sam3-main/sam3/assets/bpe_simple_vocab_16e6.txt.gz ./weights/
```

### 2) 得到含 sam3+torch(cuda)+fastapi 的镜像

最简单：`docker commit` 现有容器（体积大，但环境齐全；权重走挂载、不进镜像）：

```bash
docker commit 4090 sam3ft-api:latest
```

### 3) TLS 证书

把 `fullchain.pem`、`privkey.pem` 放进 `certs/`。Let's Encrypt 示例：

```bash
certbot certonly --standalone -d api.example.com \
  --deploy-hook "cp /etc/letsencrypt/live/api.example.com/{fullchain.pem,privkey.pem} certs/"
# 或自签：
openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
  -keyout certs/privkey.pem -out certs/fullchain.pem
```

把 `nginx.conf` 里的 `server_name` 改成你的域名。

## 配置密钥

```bash
cp ../../.env.example .env
# 编辑 .env：DEPLOY_API_KEY=<真实密钥>、SAM3_IMAGE=sam3ft-api:latest
```

## 启动与验证

```bash
docker compose up -d
docker compose ps
curl -s https://api.example.com/healthz -H "X-API-Key: $DEPLOY_API_KEY"
curl -s https://api.example.com/segment \
  -H "X-API-Key: $DEPLOY_API_KEY" \
  -F "prompt=blackheads" -F "file=@/path/to/skin.jpg"
```

## 说明

- `segment-api` 不对外暴露 8000，仅 nginx 反代可达（更安全）。
- `restart: unless-stopped` + docker 重启策略：进程/容器崩溃自动重启，**也解决了容器内 nohup 无法在容器重启后自启的问题**。
- API key 经 HTTPS 传输；`X-API-Key` 由 nginx 显式透传给后端。
- 备选：不想用 nginx 可换 Caddy（`caddy:2` 自动 HTTPS），把反代换成 Caddyfile 的 `reverse_proxy segment-api:8000`。
