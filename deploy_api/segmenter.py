"""真实 SAM3 分割器（常驻推理核心）。

自包含的推理逻辑：构建 base 模型 → 按 λ 混合权重 →
单图文本提示分割 → 返回 boxes / scores / masks。

只在装有 sam3 + torch(cuda) 的推理机（4090）上导入本模块。
"""
import numpy as np

from .lam_policy import lambda_for_prompt

DETECTOR_PREFIX = "detector."
FULL_LORA_EPS = 0.999  # λ 视作“满 LoRA”的阈值


class Sam3Segmenter:
    """加载一次模型，按请求做单图分割。

    seam contract（与 app.py 约定）：
        segment(image, prompt, conf) -> dict
            "boxes":  np.ndarray (N,4) float32   # xyxy，全部候选
            "scores": np.ndarray (N,)  float32   # 全部候选分数
            "masks":  np.ndarray (N,H,W) uint8   # 每候选二值掩码（H,W=原图尺寸）
    """

    def __init__(self, base_path, model_only_path, bpe_path,
                 device="cuda", resolution=1008):
        import torch
        from sam3 import build_sam3_image_model

        self._torch = torch
        self.device = device
        self.resolution = resolution

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.model = build_sam3_image_model(
            bpe_path=bpe_path, device=device, eval_mode=True,
            checkpoint_path=base_path, enable_segmentation=True,
            enable_inst_interactivity=False)
        self.base_sd = {k: v.detach().to("cpu", dtype=torch.float32)
                        for k, v in self.model.state_dict().items()}
        self.mo_sd = self._load_state(model_only_path)
        self.current_lam = None

    @staticmethod
    def _load_state(path):
        import torch
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        return {k[len(DETECTOR_PREFIX):] if k.startswith(DETECTOR_PREFIX) else k: v.float()
                for k, v in sd.items()}

    def _apply_lam(self, lam):
        if lam == self.current_lam:
            return
        if lam >= FULL_LORA_EPS:
            W = self.mo_sd
        else:
            W = {}
            for k, v in self.mo_sd.items():
                if k in self.base_sd and tuple(v.shape) == tuple(self.base_sd[k].shape):
                    W[k] = self.base_sd[k] + lam * (v - self.base_sd[k])
        self.model.load_state_dict(W, strict=False)
        self.current_lam = lam

    def segment(self, image, prompt, conf):
        from sam3.model.sam3_image_processor import Sam3Processor

        lam = lambda_for_prompt(prompt)
        self._apply_lam(lam)

        with self._torch.inference_mode(), \
                self._torch.autocast(self.device, dtype=self._torch.bfloat16):
            processor = Sam3Processor(self.model, resolution=self.resolution,
                                      device=self.device,
                                      confidence_threshold=conf)
            state = processor.set_image(image)
            state = processor.set_text_prompt(prompt, state)
            boxes = state["boxes"].float().cpu().numpy()
            scores = state["scores"].float().cpu().numpy()
            masks = state["masks"].cpu().numpy()

        if masks.ndim == 4:
            masks = masks[:, 0]
        masks = (masks > 0).astype(np.uint8)
        return {"boxes": boxes, "scores": scores, "masks": masks}
