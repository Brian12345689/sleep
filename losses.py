import torch
import torch.nn as nn


def env_quality_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    risk_targets: torch.Tensor | None = None,
    comfort_targets: torch.Tensor | None = None,
    class_weight: float = 1.0,
    risk_weight: float = 1.0,
    comfort_weight: float = 1.0,
) -> torch.Tensor:
    """计算多任务环境质量损失。
    """
    class_loss = nn.CrossEntropyLoss()(outputs["class_logits"], labels)
    loss = class_weight * class_loss

    if risk_targets is not None and "risk_logits" in outputs:
        risk_loss = nn.BCEWithLogitsLoss()(outputs["risk_logits"], risk_targets)
        loss = loss + risk_weight * risk_loss

    if comfort_targets is not None and "comfort_score" in outputs:
        comfort_loss = nn.MSELoss()(outputs["comfort_score"].unsqueeze(-1), comfort_targets)
        loss = loss + comfort_weight * comfort_loss

    return loss


def sleep_impact_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    output_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """计算睡眠影响预测的回归损失。
    """
    loss = nn.SmoothL1Loss(reduction="none")(outputs, targets)
    if output_weights is not None:
        if output_weights.dim() == 1:
            output_weights = output_weights.unsqueeze(0)
        loss = loss * output_weights
    return loss.mean()


def control_policy_losses(
    eval_out: dict[str, torch.Tensor],
    action_discrete: torch.Tensor,
    action_cont: torch.Tensor,
    discrete_weight: float = 1.0,
    continuous_weight: float = 1.0,
    entropy_coef: float = 0.0,
) -> dict[str, torch.Tensor]:
    """计算用于控制策略训练的监督式模仿损失。
    """
    discrete_loss = nn.CrossEntropyLoss()(eval_out["discrete_logits"], action_discrete)
    continuous_loss = nn.MSELoss()(eval_out["continuous_mean"], action_cont)
    entropy = None
    loss = discrete_weight * discrete_loss + continuous_weight * continuous_loss

    if entropy_coef != 0.0 and "continuous_log_std" in eval_out:
        discrete_dist = torch.distributions.Categorical(logits=eval_out["discrete_logits"])
        entropy = discrete_dist.entropy().mean()
        loss = loss - entropy_coef * entropy

    return {
        "loss": loss,
        "discrete_loss": discrete_loss,
        "continuous_loss": continuous_loss,
        "entropy": entropy,
    }
