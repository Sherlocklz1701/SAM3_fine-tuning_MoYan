"""HTTP 边界测试：用 fake segmenter 验证 /segment 契约（无 GPU）。"""
import asyncio
import base64
import io
import threading

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from deploy_api.app import create_app

def make_png_bytes(w=100, h=80, color=(200, 200, 200)):
    img = Image.new("RGB", (w, h), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


PNG_BYTES = make_png_bytes()


def fake_segment(image, prompt, conf):
    w, h = image.size
    masks = np.zeros((2, h, w), dtype=np.uint8)
    masks[0, 10:40, 10:40] = 1
    masks[1, 50:80, 50:80] = 1
    return {
        "boxes": np.array([[10, 10, 40, 40], [50, 50, 80, 80]], dtype=np.float32),
        "scores": np.array([0.9, 0.5], dtype=np.float32),
        "masks": masks,
    }


def make_client(segment_fn=fake_segment, **kwargs):
    app = create_app(segment_fn, **kwargs)
    return TestClient(app)


def _post(client, prompt="blackheads", png=PNG_BYTES, conf=None, api_key=None):
    headers = {"X-API-Key": api_key} if api_key else {}
    data = {"prompt": prompt}
    if conf is not None:
        data["conf"] = str(conf)
    return client.post(
        "/segment",
        files={"file": ("skin.png", png, "image/png")},
        data=data,
        headers=headers,
    )


def test_healthz():
    assert make_client().get("/healthz").json() == {"status": "ok"}


def test_readyz():
    assert make_client().get("/readyz").json() == {"status": "ready"}


def test_segment_success_shape():
    r = _post(make_client(), prompt="blackheads")
    assert r.status_code == 200
    body = r.json()
    assert body["prompt"] == "blackheads"
    assert body["lambda"] == 1.0
    assert body["mask_quality"] == "insufficient_evidence"
    assert "reference-level" in body["caveat"]
    assert body["size"] == [100, 80]
    assert body["n_ge_conf"] == 2
    assert body["boxes"] == [[10.0, 10.0, 40.0, 40.0], [50.0, 50.0, 80.0, 80.0]]
    assert body["scores"] == [0.9, 0.5]


def test_overlay_is_valid_jpeg():
    body = _post(make_client()).json()
    raw = base64.b64decode(body["overlay_b64"])
    img = Image.open(io.BytesIO(raw))
    assert img.format == "JPEG"


def test_lambda_out_of_domain():
    body = _post(make_client(), prompt="acne").json()
    assert body["lambda"] == 0.1
    assert "conservative" in body["lambda_reason"]
    assert "out-of-domain" in body["caveat"]


def test_conf_filters_candidates():
    body = _post(make_client(), conf=0.7).json()
    assert body["n_ge_conf"] == 1
    assert body["scores"] == [0.9]


def test_auth_required():
    c = make_client(api_key="secret")
    assert _post(c).status_code == 401
    assert _post(c, api_key="wrong").status_code == 401
    assert _post(c, api_key="secret").status_code == 200


def test_auth_disabled_when_no_key():
    c = make_client(api_key=None)
    assert _post(c).status_code == 200


def test_missing_file():
    c = make_client()
    r = c.post("/segment", data={"prompt": "blackheads"})
    assert r.status_code == 400


def test_missing_prompt():
    c = make_client()
    r = c.post("/segment", files={"file": ("skin.png", PNG_BYTES, "image/png")})
    assert r.status_code == 400


def test_size_limit():
    c = make_client(max_bytes=10)
    r = _post(c)
    assert r.status_code == 413


def test_invalid_image():
    c = make_client()
    r = c.post(
        "/segment",
        files={"file": ("bad.png", b"not-an-image", "image/png")},
        data={"prompt": "blackheads"},
    )
    assert r.status_code == 400


def test_busy_503():
    started = threading.Event()
    release = threading.Event()

    def blocking_segment(image, prompt, conf):
        started.set()
        release.wait(timeout=5)
        return fake_segment(image, prompt, conf)

    app = create_app(blocking_segment)

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            async def post():
                return await c.post(
                    "/segment",
                    files={"file": ("skin.png", PNG_BYTES, "image/png")},
                    data={"prompt": "blackheads"},
                )

            t1 = asyncio.create_task(post())
            for _ in range(500):
                if started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert started.is_set()
            r2 = await post()  # 第一个请求占住 worker → 应 503
            release.set()
            r1 = await t1
            return r1.status_code, r2.status_code

    s1, s2 = asyncio.run(run())
    assert s1 == 200
    assert s2 == 503
