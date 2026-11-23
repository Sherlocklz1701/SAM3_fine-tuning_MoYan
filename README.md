# SAM3FT-001 · 黑头分割模型与部署 API

> SAM3 基础模型 + **r6–ep14 LoRA 微调主权重**（黑头 / 脓疱 / 白头），及其**常驻 HTTP 分割 API**。
> 本仓库包含：**部署代码（`deploy_api/`）+ BPE 词表 + 权重元数据（sha256 / λ 策略）**。

---

## ⚠️ 权重文件不在此仓库

权重体积过大（约 20 GB），**不放入 Git**。实际权重位于：

- 服务器 `/root/weights_bundle/`（`sam3.pt`、`model_only_r6_ep14.pth`、`detector_prefix_r6_ep14.pth`、`lora_adapter_r6_ep14.pth`）
- 或打包文件 `sam3ft_weights_r6_ep14.tar`（10.22 GB）

部署时把这三个文件放到 `deploy_api/` 的**上一级目录**（本仓库根目录）即可：

```
repo/
├── deploy_api/                  ← 本仓库已含
├── bpe_simple_vocab_16e6.txt.gz ← 本仓库已含（SAM3 词表）
├── sam3.pt                      ← 自行放置（base）
└── model_only_r6_ep14.pth       ← 自行放置（λ 混合）
```

---

## 权重模型说明

| 模型 | 大小 | 是什么 | 用在哪 |
|---|---|---|---|
| `sam3.pt` | 3.45 GB | SAM3 基础权重（未微调） | 模型主干 + λ 混合的「底座」；单独加载 = 无 LoRA |
| `model_only_r6_ep14.pth` | 3.36 GB | r6-ep14 合并后主权重（**标准键**） | **API 默认**：λ 混合计算 `base + λ·(model_only − base)` |
| `detector_prefix_r6_ep14.pth` | 3.36 GB | 同权重、`detector.` 前缀 | λ=1.0 直接推理（黑头/脓疱/白头） |
| `lora_adapter_r6_ep14.pth` | 43 MB | LoRA 增量（A/B 矩阵，rank16/alpha32） | 从 adapter 自己重合并/重混合 |
| `bpe_simple_vocab_16e6.txt.gz` | 1.36 MB | SAM3 BPE 词表 | 推理加载（本仓库已含） |

> `detector_prefix` 与 `model_only` 是**同一份模型两种键格式**，两者取其一即可（各 3.36 GB）。

### λ 策略（哪个模型干什么）

- **黑头 / 脓疱 / 白头**（域内）→ **λ=1.0**（满 LoRA，最强项）
- **其它内容**（acne、dark circles、皱纹、灰斑…）→ **λ=0.1**（保守近 base，防弱迁移干扰）
- 只想要通用 base → `sam3.pt` 单独（λ=0，黑头 AP50 ≈ .077）

> 一句话：**域内三类用满 LoRA（λ=1），其它一切用 λ=0.1**。

### LoRA 混合公式

```
W_λ = base + λ · (model_only − base)
```

- λ=1.0 → 直接用 `model_only`（或 `detector_prefix`）
- λ=0.0 → 直接用 `sam3.pt`
- λ=0.1（或任意 0~1）→ 按上式混合一次再加载（API 已内置）

---

## 部署（deploy_api）

`deploy_api/` 是常驻 HTTP 分割服务：客户端上传图片 + 文本提示，返回结构化 JSON + overlay 图。

### 环境依赖

- Python 3.12，`torch`(cuda) + `sam3` + `PIL` + `numpy`（推理机已装，如 `/root/miniconda3/envs/uly/bin/python`）
- HTTP：`pip install fastapi uvicorn python-multipart`

### 目录要求

`serve.py` 默认从 `deploy_api/` 的**上一级目录**找权重与词表，故把 `deploy_api/`、`sam3.pt`、`model_only_r6_ep14.pth`、`bpe_simple_vocab_16e6.txt.gz` 放同一目录即可。也可用 `--base` / `--model-only` / `--bpe` 显式覆盖。

### 启动

```bash
export DEPLOY_API_KEY=<你的密钥>     # 客户端 X-API-Key 用同一值

cd <仓库根目录>
python -m deploy_api.serve --host 0.0.0.0 --port 8000   # 模型加载约 40s 后才 ready
```

### 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/segment` | `multipart`：`file`=图片、`prompt`=文本、可选 `conf`；返回 JSON |
| GET | `/healthz` | 存活探针 |
| GET | `/readyz` | 模型已加载（可接请求） |

请求示例：

```bash
curl -H "X-API-Key: $DEPLOY_API_KEY" \
     -F "prompt=blackheads" -F "file=@skin.jpg" \
     http://127.0.0.1:8000/segment
```

响应（单图扁平 JSON）：`prompt / lambda / lambda_reason / conf / mask_quality / caveat / name / size / n_ge_conf / boxes[] / scores[] / overlay_b64`。

### 常驻 + 对外暴露

- **容器常驻**（无 systemd）：`deploy_api/deploy/run.sh`（nohup + 自动重启）/ `stop.sh`。
- **宿主机 HTTPS**：`deploy_api/deploy/host/`（docker-compose + nginx，发布端口 + TLS）。
- 详见 `deploy_api/deploy/DEPLOY.md`。

---

## 完整性校验

```bash
shasum -a 256 sam3.pt model_only_r6_ep14.pth detector_prefix_r6_ep14.pth lora_adapter_r6_ep14.pth
# 期望（前 16 位，详见 MANIFEST.json）：
#   sam3.pt                        9999e2341ceef5e1
#   model_only_r6_ep14.pth         a124132ddce01ce1
#   detector_prefix_r6_ep14.pth    195ea6301d4c7c8c
#   lora_adapter_r6_ep14.pth       702852387e53bd64
```

---

## 已知行为与边界（务必先读）

- **掩码参考级**：训练为**纯框**（`enable_segmentation=False`），掩码由 base 的**未训练分割头**产出，无像素 GT ⇒ `mask_quality=insufficient_evidence`。**不要**拿掩码做像素级计量（不能算 IoU/Dice/面积）。
- **域外弱迁移**：acne 仅低置信（0.30–0.42）、60/73 张无候选；**dark circles 任意 λ 都 0 激活**（当前模型族不可分割）。λ=0.1 只是保守默认，不代表域外能可靠分割。
- **域内阈值**：黑头 0.55 / 脓疱 0.65 / 白头 0.50（默认 `conf=0.3` 只作定性）。
- 无字库环境请用英文标注（容器无中文字体）。

---

## 溯源

- 训练：SAM3FT-001；主权重 = R6 **checkpoint_14**（CT=False、lr_lora=1.4e-4、warmup=60、cosine total=1056、24ep；LoRA rank16/alpha32/scaling2.0，仅 LoRA 可训练 1.266%）。
- 域内基准（测试集 101，v2 阈值）：overall F1 `.553`、AP50 `.515`；黑头 P/R/F1 `.782/.760/.771`、AP50 `.804`；每图 FP `3.74`。对照 r4（balanced-overall-F1 `.573`）留作基线。
