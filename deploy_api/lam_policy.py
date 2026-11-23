"""λ 策略（纯函数，无第三方依赖）。

公式：W = base + λ·(LoRA合并 − base)
- prompt ∈ {blackheads, pustules, whiteheads} → λ=1.0（满 LoRA）
- 其余任何内容 → λ=0.1（保守近 base）
"""
IN_DOMAIN = {"blackheads", "pustules", "whiteheads"}
LAM_IN_DOMAIN = 1.0
LAM_ELSE = 0.1


def _is_in_domain(prompt: str) -> bool:
    return (prompt or "").strip() in IN_DOMAIN


def lambda_for_prompt(prompt: str) -> float:
    """按文本提示返回 λ 混合强度。"""
    return LAM_IN_DOMAIN if _is_in_domain(prompt) else LAM_ELSE


def lambda_reason(prompt: str) -> str:
    """返回 λ 判定原因（用于响应与审计）。"""
    if _is_in_domain(prompt):
        return "in-domain (blackheads/pustules/whiteheads) -> full LoRA"
    return "other class -> conservative 0.1"
