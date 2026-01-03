from typing import Optional

import torch


def squash_safety_logits(
    reward_logits: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """
    The reward logit of the Q-model is compressed to the range (0, 1) and used as the safety score s(v).
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
Single-step risk r_t^{(j)} calculation:

- probs: shape [B, K], the probability of candidate tokens for each path at step t according to the policy
- reward_logits: shape [B, K], the reward logits of the corresponding candidate tokens from the Q model
- mask: shape [B, K], a mask similar to top_p; tokens with a value of False do not participate in risk calculation
- kappa: scaling parameter for safety logit -> safety probability
- lambda_c, C0: hyperparameters for the distribution concentration correction term (initially set lambda_c=0)

Returns:
- step_risk: shape [B], the single-step risk r_t^{(j)} for each path at this step
    """

    if mask is not None:
        probs = probs * mask


    s = squash_safety_logits(reward_logits, kappa=kappa)

    if mask is not None:
        s = s * mask


    unsafe_mass = probs * (1.0 - s)
    U = unsafe_mass.sum(dim=-1)  # [B]

    U_safe = U.clamp_min(eps)

    q = unsafe_mass / U_safe.unsqueeze(-1)

    q = torch.where(
        (U_safe > 0).unsqueeze(-1),
        q,
        torch.zeros_like(q),
    )


    C = torch.sqrt((q ** 2).sum(dim=-1))  # [B]



    step_risk = U * (1.0 + lambda_c * (C - C0))

    return step_risk


def update_sequence_risk(
    prev_R: torch.Tensor,
    step_risk: torch.Tensor,
    rho: float = 0.8,
) -> torch.Tensor:

    return rho * prev_R + (1.0 - rho) * step_risk


def compute_safe_scores(
    R: torch.Tensor,
    gamma_abs: float = 1.0,
    gamma_rel: float = 0.0,
    w_abs: float = 1.0,
    w_rel: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:


    S_abs = torch.exp(-gamma_abs * R)


    if gamma_rel != 0.0 and w_rel != 0.0:
        mean = R.mean()
        std = R.std(unbiased=False).clamp_min(eps)
        z = (R - mean) / std
        S_rel = torch.exp(-gamma_rel * z)
    else:
        S_rel = torch.zeros_like(R)

    S = w_abs * S_abs + w_rel * S_rel
    return S

