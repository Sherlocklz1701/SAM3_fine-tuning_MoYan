"""SAM3FT-001 分割服务 FastAPI 应用工厂。

seam：HTTP 边界 + 可注入的 `segment(image, prompt, conf) -> dict`。
测试注入 fake segmenter；生产注入 Sam3Segmenter.segment。
"""
import asyncio
import base64
import io
from typing import Callable, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from PIL import Image, ImageDraw
from starlette.concurrency import run_in_threadpool

from .lam_policy import lambda_for_prompt, lambda_reason

DEFAULT_MAX_BYTES = 20 * 1024 * 1024
DEFAULT_OVERLAY_MAX_EDGE = 1024
DEFAULT_OVERLAY_QUALITY = 85
OVERLAY_COLOR = (255, 90, 90)
# 掩码由 base 未训练分割头产出，无像素 GT（对齐 AGENTS.md 的 mask_quality_status 口径）
MASK_QUALITY = "insufficient_evidence"

_BASE_CAVEAT = ("masks are reference-level (base un-fine-tuned segmentation head; "
                "no pixel GT) — do not use for pixel-level metric / IoU / area")


def _open_rgb(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}")


def _render_overlay(img, union_mask, boxes, max_edge=DEFAULT_OVERLAY_MAX_EDGE,
                    quality=DEFAULT_OVERLAY_QUALITY) -> str:
    base = np.array(img).astype(np.uint8).copy()
    if union_mask is not None and int(union_mask.sum()) > 0:
        col = np.array(OVERLAY_COLOR, dtype=np.int16)
        blend = base.astype(np.int16)
        base[union_mask > 0] = (blend[union_mask > 0] * 0.55 + col * 0.45).astype(np.uint8)
    ov = Image.fromarray(base)
    d = ImageDraw.Draw(ov)
    for box in boxes:
        x0, y0, x1, y1 = [int(v) for v in box]
        d.rectangle([x0, y0, x1, y1], outline=(255, 255, 255), width=2)
    w, h = ov.size
    if max(w, h) > max_edge:
        scale = max_edge / max(w, h)
        ov = ov.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    ov.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _caveat(prompt: str) -> str:
    caveat = _BASE_CAVEAT
    if lambda_for_prompt(prompt) < 1.0:
        caveat += ("; out-of-domain prompt -> lambda=0.1 conservative default "
                   "(weak / low-confidence transfer; dark circles may yield zero)")
    return caveat


def create_app(segment_fn: Callable, *, api_key: Optional[str] = None,
               max_bytes: int = DEFAULT_MAX_BYTES,
               overlay_max_edge: int = DEFAULT_OVERLAY_MAX_EDGE,
               overlay_quality: int = DEFAULT_OVERLAY_QUALITY) -> FastAPI:
    app = FastAPI(title="SAM3FT-001 segmentation API")
    sem = asyncio.Semaphore(1)  # 单 GPU worker：同一时刻只跑一张图

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz():
        # 模型在 serve.py 启动阶段已加载完毕，服务才开始对外；故恒为 ready。
        return {"status": "ready"}

    @app.post("/segment")
    async def segment(request: Request,
                      file: Optional[UploadFile] = File(None),
                      prompt: str = Form(""),
                      conf: float = Form(0.3)):
        # 鉴权
        if api_key:
            provided = request.headers.get("x-api-key")
            if not provided or provided != api_key:
                raise HTTPException(status_code=401,
                                    detail="invalid or missing X-API-Key")
        # 校验
        if file is None:
            raise HTTPException(status_code=400, detail="file is required")
        prompt = (prompt or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")

        # 流式限额读取：只读 max_bytes+1 字节，超限即拒，避免超大文件先吃满内存。
        data = await file.read(max_bytes + 1) if max_bytes else await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="empty file")
        if max_bytes and len(data) > max_bytes:
            raise HTTPException(status_code=413,
                                detail=f"file exceeds {max_bytes} bytes")
        img = _open_rgb(data)

        # 串行化（单 worker；忙时最多等 50ms，否则 503，不无限排队）
        try:
            await asyncio.wait_for(sem.acquire(), timeout=0.05)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=503, detail="server busy, retry later")
        try:
            # conf 同时传给 segment_fn（Sam3Processor 的 confidence_threshold）
            # 并在下方再做一次 >=conf 过滤（processor 阈值 + 显式过滤，双保险）。
            result = await run_in_threadpool(segment_fn, img, prompt, float(conf))
        finally:
            sem.release()

        boxes = np.asarray(result["boxes"], dtype=np.float32)
        scores = np.asarray(result["scores"], dtype=np.float32)
        masks = np.asarray(result["masks"], dtype=np.uint8)

        sel = [i for i in range(len(scores)) if scores[i] >= conf]
        sel_boxes = [boxes[i].tolist() for i in sel]
        sel_scores = [round(float(scores[i]), 4) for i in sel]

        union = None
        if sel and masks.ndim == 3 and masks.shape[0] == len(scores):
            union = masks[sel].max(axis=0) if len(sel) else None
            if union is not None:
                union = (union > 0).astype(np.uint8)

        w, h = img.size
        overlay_b64 = _render_overlay(
            img, union, sel_boxes,
            max_edge=overlay_max_edge, quality=overlay_quality)

        return {
            "prompt": prompt,
            "lambda": lambda_for_prompt(prompt),
            "lambda_reason": lambda_reason(prompt),
            "conf": conf,
            "mask_quality": MASK_QUALITY,
            "caveat": _caveat(prompt),
            "name": file.filename or "image",
            "size": [w, h],
            "n_ge_conf": len(sel),
            "boxes": sel_boxes,
            "scores": sel_scores,
            "overlay_b64": overlay_b64,
        }

    return app
