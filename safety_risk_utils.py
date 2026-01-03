# safety_risk_utils.py

from typing import Optional

import torch


def squash_safety_logits(
    reward_logits: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """
    把 Q 模型的 reward logit 压到 (0, 1)，作为安全评分 s(v)。
    这里假设 logit 越大越安全（和你 eval_reward >= 0 判安全的逻辑一致）。
    """
    return torch.sigmoid(kappa * reward_logits)


def compute_step_risk(
    probs: torch.Tensor,
    reward_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    kappa: float = 1.0,
    lambda_c: float = 0.0,
    C0: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    单步风险 r_t^{(j)} 计算：

    - probs: shape [B, K]，policy 在 step t 对每个 path 的候选 token 概率
    - reward_logits: shape [B, K]，Q 模型对应候选 token 的 reward logit
    - mask: shape [B, K]，top_p 之类的 mask，为 False 的 token 不参与风险计算
    - kappa: 安全 logit -> 安全概率 的尺度参数
    - lambda_c, C0: 分布集中度修正项的超参数（先设 lambda_c=0 即可）

    返回：
    - step_risk: shape [B]，每个 path 在该 step 的单步风险 r_t^{(j)}
    """
    # probs, reward_logits, mask 都是 [B, K]
    if mask is not None:
        probs = probs * mask

    # 安全分 s(v) ∈ (0,1)
    s = squash_safety_logits(reward_logits, kappa=kappa)

    if mask is not None:
        s = s * mask

    # 绝对不安全质量 U_t^{(j)} = Σ p(v) * (1 - s(v))
    unsafe_mass = probs * (1.0 - s)
    U = unsafe_mass.sum(dim=-1)  # [B]

    # 如果 U==0，则说明所有 token 都非常安全，此时风险为 0
    U_safe = U.clamp_min(eps)

    # 在“不安全质量”的内部归一化，得到 q(v)
    q = unsafe_mass / U_safe.unsqueeze(-1)
    # 对 U==0 的行，把 q 置为 0，防止 Nan
    q = torch.where(
        (U_safe > 0).unsqueeze(-1),
        q,
        torch.zeros_like(q),
    )

    # 集中度 C_t^{(j)} = ||q||_2
    C = torch.sqrt((q ** 2).sum(dim=-1))  # [B]

    # 总风险 r_t^{(j)} = U_t^{(j)} * (1 + lambda_c * (C - C0))
    # 默认 lambda_c=0，则退化为 r_t^{(j)} = U_t^{(j)}
    step_risk = U * (1.0 + lambda_c * (C - C0))

    return step_risk


def update_sequence_risk(
    prev_R: torch.Tensor,
    step_risk: torch.Tensor,
    rho: float = 0.8,
) -> torch.Tensor:
    """
    序列级风险累计：
    R_t^{(j)} = rho * R_{t-1}^{(j)} + (1 - rho) * r_t^{(j)}

    - prev_R: [B]，上一 step 的 R_{t-1}
    - step_risk: [B]，当前 step 的 r_t
    """
    return rho * prev_R + (1.0 - rho) * step_risk


def compute_safe_scores(
    R: torch.Tensor,
    gamma_abs: float = 1.0,
    gamma_rel: float = 0.0,
    w_abs: float = 1.0,
    w_rel: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    把序列级风险 R_t^{(j)} 映射成安全评分 S_t^{(j)} ∈ (0,1)，越大越安全。

    - 绝对项：S_abs = exp(-gamma_abs * R)
    - 相对项：S_rel = exp(-gamma_rel * z)，z 为跨路径 z-score
    - 最终：S = w_abs * S_abs + w_rel * S_rel

    默认只用绝对项 (w_abs=1, w_rel=0)，避免过多超参，后面你想开相对项可以调。
    """
    # 绝对安全评分
    S_abs = torch.exp(-gamma_abs * R)

    # 相对安全评分（可选）
    if gamma_rel != 0.0 and w_rel != 0.0:
        mean = R.mean()
        std = R.std(unbiased=False).clamp_min(eps)
        z = (R - mean) / std
        S_rel = torch.exp(-gamma_rel * z)
    else:
        S_rel = torch.zeros_like(R)

    S = w_abs * S_abs + w_rel * S_rel
    return S
