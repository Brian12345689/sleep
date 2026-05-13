#!/usr/bin/env python3
"""
预测脚本：使用训练好的模型进行推理
支持单样本预测和批量评估
"""

import argparse
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sleep_model import (
    EnvQualityClassifier,
    SleepImpactPredictor,
    ControlPolicyModel,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model(model_type: str, checkpoint_path: str, device: str):
    """加载指定类型的模型及权重"""
    if model_type == "env_quality":
        model = EnvQualityClassifier(
            numeric_dim=6,
            odor_vocab_size=5,
            odor_emb_dim=4,
            hidden_dim=64,
            risk_dim=3,
            num_classes=4,
        )
    elif model_type == "sleep_impact":
        model = SleepImpactPredictor(
            env_seq_dim=4,
            static_dim=11,
            hist_dim=5,
            lstm_hidden_dim=64,
            lstm_layers=2,
            fusion_hidden_dim=128,
            output_dim=6,
            dropout=0.0,
        )
    elif model_type == "control_policy":
        model = ControlPolicyModel(
            state_dim=5,
            discrete_action_dim=3,
            continuous_action_dim=2,
            hidden_dim=128,
            rnn_layers=2,
            rnn_type="GRU",
        )
    else:
        raise ValueError(f"未知模型类型: {model_type}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)  # 直接是state_dict
    model.to(device)
    model.eval()
    logger.info(f"模型 {model_type} 已加载权重: {checkpoint_path}")
    return model


@torch.no_grad()
def predict_env_quality(model, numeric: np.ndarray, odor: np.ndarray) -> Dict[str, np.ndarray]:
    """
    预测环境质量
    Args:
        numeric: shape (batch, 6) 或 (6,)
        odor: shape (batch,) 或标量
    Returns:
        comfort (0~1), risk_probs (sigmoid), class_probs (softmax)
    """
    device = next(model.parameters()).device
    numeric = torch.FloatTensor(numeric).to(device)
    odor = torch.LongTensor(odor).to(device)
    if numeric.dim() == 1:
        numeric = numeric.unsqueeze(0)
    # Embedding 接受 (batch,) 或标量，不需要 unsqueeze

    out = model(numeric, odor)
    comfort = out["comfort_score"].cpu().numpy()
    risk_probs = torch.sigmoid(out["risk_logits"]).cpu().numpy()
    class_probs = torch.softmax(out["class_logits"], dim=-1).cpu().numpy()
    class_pred = np.argmax(class_probs, axis=-1)

    return {
        "comfort": comfort,
        "risk_probs": risk_probs,
        "class_probs": class_probs,
        "class_pred": class_pred,
    }


@torch.no_grad()
def predict_sleep_impact(
    model,
    env_seq: np.ndarray,
    static: np.ndarray,
    history: np.ndarray,
    seq_lengths: np.ndarray = None,
) -> np.ndarray:
    """
    预测睡眠指标
    Args:
        env_seq: (batch, T, 4)
        static: (batch, 11)
        history: (batch, 5)
        seq_lengths: (batch,) 或 None
    Returns:
        预测值数组 (batch, 6)，顺序:
        [睡眠效率, 入睡潜伏期, 深睡时长, 觉醒次数, 呼吸暂停指数, 主观睡眠质量]
    """
    device = next(model.parameters()).device
    env_seq = torch.FloatTensor(env_seq).to(device)
    static = torch.FloatTensor(static).to(device)
    history = torch.FloatTensor(history).to(device)
    if env_seq.dim() == 2:
        env_seq = env_seq.unsqueeze(0)
        static = static.unsqueeze(0)
        history = history.unsqueeze(0)
    if seq_lengths is not None:
        seq_lengths = torch.LongTensor(seq_lengths).to(device)
    else:
        seq_lengths = None

    outputs = model(env_seq, static, history, seq_lengths)  # (batch, 6)
    return outputs.cpu().numpy()


@torch.no_grad()
def predict_control(
    model,
    state_seq: np.ndarray,
    seq_lengths: np.ndarray = None,
    deterministic: bool = False,
) -> Dict[str, np.ndarray]:
    """
    控制策略预测（生成动作）
    Args:
        state_seq: (batch, T, 5) 或 (T, 5)
        seq_lengths: (batch,) 或 None
        deterministic: True 时使用均值（连续动作），离散动作取 argmax
    Returns:
        discrete_action, continuous_action, state_value
    """
    device = next(model.parameters()).device
    state_seq = torch.FloatTensor(state_seq).to(device)
    if state_seq.dim() == 2:
        state_seq = state_seq.unsqueeze(0)
    if seq_lengths is not None:
        seq_lengths = torch.LongTensor(seq_lengths).to(device)
    else:
        seq_lengths = None

    if deterministic:
        out = model.forward(state_seq, seq_lengths)
        disc_action = torch.argmax(out["discrete_logits"], dim=-1)
        cont_action = out["continuous_mean"]
        state_value = out["state_value"]
        log_prob = None
    else:
        act_out = model.act(state_seq, seq_lengths)
        disc_action = act_out["discrete_action"]
        cont_action = act_out["continuous_action"]
        state_value = act_out["state_value"]
        log_prob = act_out["log_prob"].cpu().numpy()

    result = {
        "discrete_action": disc_action.cpu().numpy(),
        "continuous_action": cont_action.cpu().numpy(),
        "state_value": state_value.cpu().numpy(),
    }
    if log_prob is not None:
        result["log_prob"] = log_prob
    return result


def main():
    parser = argparse.ArgumentParser(description="模型预测脚本")
    parser.add_argument("--model", type=str, required=True,
                        choices=["env_quality", "sleep_impact", "control_policy"],
                        help="模型类型")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型权重路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--input", type=str, required=True,
                        help="输入数据文件（.npz 格式）")
    parser.add_argument("--output", type=str, default=None,
                        help="结果保存路径（.npy 或 .npz），若不指定则打印")
    parser.add_argument("--deterministic", action="store_true",
                        help="控制策略是否使用确定性动作（默认随机采样）")
    args = parser.parse_args()

    # 加载模型
    model = load_model(args.model, args.checkpoint, args.device)

    # 读取输入数据
    data = dict(np.load(args.input, allow_pickle=True))

    # 根据模型类型进行推理
    if args.model == "env_quality":
        numeric = data["numeric"]  # (N, 6) 或 (6,)
        odor = data["odor"]        # (N,) 或标量
        results = predict_env_quality(model, numeric, odor)
    elif args.model == "sleep_impact":
        env_seq = data["env_seq"]
        static = data["static"]
        history = data["history"]
        seq_lengths = data.get("seq_lengths", None)
        results = predict_sleep_impact(model, env_seq, static, history, seq_lengths)
    else:  # control_policy
        state_seq = data["state_seq"]
        seq_lengths = data.get("seq_lengths", None)
        results = predict_control(model, state_seq, seq_lengths, args.deterministic)

    # 输出
    if args.output:
        if isinstance(results, dict):
            np.savez(args.output, **results)
        else:
            np.save(args.output, results)
        logger.info(f"结果已保存至 {args.output}")
    else:
        logger.info("预测结果：")
        if isinstance(results, dict):
            for k, v in results.items():
                print(f"{k}: {v}")
        else:
            print(results)


if __name__ == "__main__":
    main()