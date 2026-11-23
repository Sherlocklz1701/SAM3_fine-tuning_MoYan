"""生产入口：加载模型 → 创建 app → 单 worker 跑 uvicorn。

环境变量：
  DEPLOY_API_KEY           客户端 X-API-Key（为空则关闭鉴权，生产必填）
  DEPLOY_MAX_BYTES         图片体积上限（默认 20971520）
  DEPLOY_OVERLAY_MAX_EDGE  overlay 最长边（默认 1024）
  DEPLOY_OVERLAY_QUALITY   overlay JPEG 质量（默认 85）
"""
import argparse
import os

from .app import DEFAULT_MAX_BYTES, DEFAULT_OVERLAY_MAX_EDGE, DEFAULT_OVERLAY_QUALITY, create_app
from .segmenter import Sam3Segmenter

# 权重与词表默认放在 deploy_api 包的上一级目录（即 weights_bundle/），
# 使 bundle 自包含：模型 + bpe + deploy_api 同放一个目录即可跑。
_BUNDLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description="SAM3FT-001 分割 API 服务")
    ap.add_argument("--base", default=os.path.join(_BUNDLE_DIR, "sam3.pt"))
    ap.add_argument("--model-only",
                    default=os.path.join(_BUNDLE_DIR, "model_only_r6_ep14.pth"))
    ap.add_argument("--bpe",
                    default=os.path.join(_BUNDLE_DIR, "bpe_simple_vocab_16e6.txt.gz"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resolution", type=int, default=1008)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    api_key = os.environ.get("DEPLOY_API_KEY") or None
    max_bytes = int(os.environ.get("DEPLOY_MAX_BYTES", DEFAULT_MAX_BYTES))
    overlay_max_edge = int(os.environ.get("DEPLOY_OVERLAY_MAX_EDGE", DEFAULT_OVERLAY_MAX_EDGE))
    overlay_quality = int(os.environ.get("DEPLOY_OVERLAY_QUALITY", DEFAULT_OVERLAY_QUALITY))

    # 启动时加载模型（阻塞；加载完才对外服务）
    seg = Sam3Segmenter(base_path=args.base, model_only_path=args.model_only,
                        bpe_path=args.bpe, device=args.device,
                        resolution=args.resolution)
    app = create_app(seg.segment, api_key=api_key, max_bytes=max_bytes,
                     overlay_max_edge=overlay_max_edge,
                     overlay_quality=overlay_quality)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
