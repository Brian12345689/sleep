
"""
快速测试脚本：验证模型初始化和前向传播
不需要完整数据即可运行
"""

import torch
from sleep_model import build_models

def test_models():
    """测试模型初始化和前向传播"""
    print("构建模型...")
    models = build_models()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 测试 EnvQualityClassifier
    print("\n测试 EnvQualityClassifier...")
    model = models["env_quality_classifier"].to(device)
    batch_size = 4
    numeric_feat = torch.randn(batch_size, 6).to(device)  # 6个数值特征
    odor_idx = torch.randint(0, 5, (batch_size,)).to(device)  # 5种气味类型

    with torch.no_grad():
        output = model(numeric_feat, odor_idx)
        print(f"输入形状: numeric_feat={numeric_feat.shape}, odor_idx={odor_idx.shape}")
        print(f"comfort_score shape: {output['comfort_score'].shape}")
        print(f"risk_logits shape: {output['risk_logits'].shape}")
        print(f"class_logits shape: {output['class_logits'].shape}")

    # 测试 SleepImpactPredictor
    print("\n测试 SleepImpactPredictor...")
    model = models["sleep_impact_predictor"].to(device)
    seq_len = 8
    env_seq = torch.randn(batch_size, seq_len, 4).to(device)  # 4个环境特征序列
    static_feat = torch.randn(batch_size, 11).to(device)  # 11个静态特征
    hist_feat = torch.randn(batch_size, 5).to(device)  # 5个历史睡眠特征（不含psqi_score）

    with torch.no_grad():
        output = model(env_seq, static_feat, hist_feat)
        print(f"输入形状: env_seq={env_seq.shape}, static_feat={static_feat.shape}, hist_feat={hist_feat.shape}")
        print(f"输出形状: {output.shape} (constrained predictions)")
        print(f"输出范围检查: sleep_efficiency ∈ [0,1]: {output[:, 0].min().item():.3f} - {output[:, 0].max().item():.3f}")
        print(f"其他指标 ≥ 0: min values: {output[:, 1:].min(dim=0).values}")

    # 测试 ControlPolicyModel (Actor-Critic)
    print("\n测试 ControlPolicyModel (Actor-Critic)...")
    model = models["control_policy_model"].to(device)
    state_seq = torch.randn(batch_size, seq_len, 5).to(device)  # 5个状态特征序列

    with torch.no_grad():
        output = model(state_seq)
        sampled = model.act(state_seq)
        eval_out = model.evaluate_actions(
            state_seq,
            sampled["discrete_action"],
            sampled["continuous_action"],
        )
        print(f"输入形状: state_seq={state_seq.shape}")
        print(f"离散策略形状: {output['discrete_logits'].shape} (logits for 3 actions)")
        print(f"连续动作均值形状: {output['continuous_mean'].shape} (2 continuous actions)")
        print(f"价值输出形状: {output['state_value'].shape}")
        print(f"采样离散动作形状: {sampled['discrete_action'].shape}")
        print(f"采样连续动作形状: {sampled['continuous_action'].shape}")
        print(f"动作评估 log_prob 形状: {eval_out['log_prob'].shape}")

    print("\n所有模型测试通过！")

if __name__ == "__main__":
    test_models()