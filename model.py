"""Transformer-DQN 网络结构：Dueling + 分布强化学习 + NoisyNet。

核心改动（相比原始版本）：
  1. Dueling 架构 — V(s, τ) + A(s, a, τ) − mean_a A
  2. 分布输出 — 每个动作输出 N=51 个分位数，而非单一标量 Q 值
  3. NoisyNet — 输出层用 FactorizedNoisyLinear，权重噪声驱动探索

网络流程：
  棋盘 4×4 → log2 → 展平 (16,) → 值嵌入 + 2D 位置编码
  → Transformer Encoder (2层, 2头) → GAP
  → Q-Trunk (Linear+LN+ReLU) → [Value Head | Advantage Head] → 分布 Q
  → SSL Head → 下一状态预测
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
#  FactorizedNoisyLinear — NoisyNet 的核心组件
# ═══════════════════════════════════════════════════════════════

class FactorizedNoisyLinear(nn.Module):
    """分解高斯噪声线性层。

    权重 W = μ_W + σ_W ⊙ ε_W，其中：
      μ_W — 可学习的权重均值（与标准 Linear 相同）
      σ_W — 可学习的噪声幅度
      ε_W = f(ε_in) ⊗ f(ε_out) — 分解噪声（外积），f(x) = sign(x)·√|x|

    训练时每步重采样 ε，评估时使用 μ（等价于关闭噪声）。

    为什么分解噪声？
      独立噪声需要 O(in×out) 个随机数，分解噪声只需 O(in+out) 个，
      同时保持类似独立噪声的表达能力。
    """

    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.reset_parameters(sigma_init)

    def reset_parameters(self, sigma_init):
        """μ 用均匀初始化（同 Linear），σ 用 σ₀ / √d 初始化。"""
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(x):
        """分解噪声中使用的缩放变换 f(x) = sign(x)·√|x|。"""
        return x.sign() * x.abs().sqrt()

    def forward(self, x):
        if self.training:
            # 每次前向传播重采样噪声（仅在训练时）
            eps_in = self._scale_noise(torch.randn(self.in_features, device=x.device))
            eps_out = self._scale_noise(torch.randn(self.out_features, device=x.device))
            weight = self.weight_mu + self.weight_sigma * eps_out.outer(eps_in)
            bias = self.bias_mu + self.bias_sigma * eps_out
        else:
            # 评估模式：使用期望权重（噪声归零）
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


# ═══════════════════════════════════════════════════════════════
#  TransformerDQN — 主网络
# ═══════════════════════════════════════════════════════════════

class TransformerDQN(nn.Module):
    """Transformer 编码器 + Dueling 分布 Q 头 + SSL 辅助头。

    Args:
        action_dim: 动作空间大小（2048 为 4: 上下左右）
        board_size: 棋盘格子数（4×4=16）
        d_model: Transformer 的隐藏维度
        nhead: 多头注意力头数
        num_layers: Transformer Encoder 层数
        num_quant: 分布 RL 的分位数数量
    """

    def __init__(self, action_dim=4, board_size=16, d_model=64, nhead=2,
                 num_layers=2, num_quant=51):
        super(TransformerDQN, self).__init__()

        self.action_dim = action_dim
        self.board_size = board_size
        self.num_quant = num_quant

        # ── 1. 输入编码 ───────────────────────────────────────
        # log2 后的值范围 [0, 15]（2¹ 到 2¹⁵），用 16 类嵌入
        self.val_embedding = nn.Embedding(num_embeddings=16, embedding_dim=d_model)

        # 2D 位置编码：行和列独立编码后拼接
        # 让 Transformer 知道每个值在棋盘上的 (row, col) 位置
        self.row_embed = nn.Embedding(4, d_model // 2)
        self.col_embed = nn.Embedding(4, d_model // 2)

        # ── 2. Transformer 主干 ───────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,    # FFN 内部维度
            batch_first=True,               # 输入格式 (B, Seq, D)
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        pool_dim = d_model  # GAP 后的特征维度 = Transformer 的 d_model

        # ── 3. Dueling 分布 Q 头 ──────────────────────────────
        # 共享 trunk → V(s, τ) 价值流 + A(s, a, τ) 优势流 → 合并
        self.q_trunk = nn.Sequential(
            nn.Linear(pool_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        # 价值流: 输出 N 个分位数（与动作无关）
        self.value_head = FactorizedNoisyLinear(256, num_quant)
        # 优势流: 输出 action_dim × N 个分位数（reshape 为 (action_dim, N)）
        self.advantage_head = FactorizedNoisyLinear(256, action_dim * num_quant)

        # ── 4. SSL 自监督预测头 ───────────────────────────────
        # 预测每个动作执行后的下一棋盘状态（4 × 16 = 64 维输出）
        self.ssl_head = nn.Sequential(
            nn.Linear(pool_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            FactorizedNoisyLinear(256, action_dim * board_size),
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, 16) — log2 处理后的 16 格棋盘展平向量

        Returns:
            q_quantiles: (batch_size, 4, num_quant) — 每动作的分布 Q 值
            ssl_preds:  (batch_size, 4, 16) — 每动作预测的下一状态
        """
        x = x.long()
        batch_size = x.size(0)
        device = x.device

        # ── 输入编码 ──────────────────────────────────────────
        val_emb = self.val_embedding(x)                               # (B, 16, d_model)

        # 2D 位置索引：行索引 [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3]
        #                列索引 [0,1,2,3, 0,1,2,3, 0,1,2,3, 0,1,2,3]
        row_idx = torch.arange(4, device=device).repeat_interleave(4)
        row_idx = row_idx.unsqueeze(0).expand(batch_size, -1)        # (B, 16)
        col_idx = torch.arange(4, device=device).repeat(4)
        col_idx = col_idx.unsqueeze(0).expand(batch_size, -1)        # (B, 16)

        r_emb = self.row_embed(row_idx)                               # (B, 16, d/2)
        c_emb = self.col_embed(col_idx)                               # (B, 16, d/2)
        pos_emb = torch.cat([r_emb, c_emb], dim=-1)                   # (B, 16, d)

        seq = val_emb + pos_emb                                       # 值嵌入 + 位置编码

        # ── Transformer 编码 ──────────────────────────────────
        out = self.transformer(seq)                                   # (B, 16, d)

        # 全局平均池化 → 固定长度特征向量
        features = out.mean(dim=1)                                    # (B, d)

        # ── Dueling + 分布 Q ─────────────────────────────────
        # Q(s, a, τ) = V(s, τ) + A(s, a, τ) − (1/4) Σ_{a'} A(s, a', τ)
        q_hidden = self.q_trunk(features)                             # (B, 256)
        v = self.value_head(q_hidden)                                 # (B, N)
        a = self.advantage_head(q_hidden)
        a = a.reshape(-1, self.action_dim, self.num_quant)            # (B, 4, N)
        q_quantiles = v.unsqueeze(1) + a - a.mean(dim=1, keepdim=True)

        # ── SSL 预测 ──────────────────────────────────────────
        ssl_preds = self.ssl_head(features)
        ssl_preds = ssl_preds.reshape(-1, self.action_dim, self.board_size)

        return q_quantiles, ssl_preds


if __name__ == "__main__":
    print("=== TransformerDQN 编译测试 ===")

    model = TransformerDQN(num_quant=51)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总量: {total_params:,}")

    dummy_x = torch.tensor([
        [0, 0, 1, 0, 3, 2, 0, 0, 4, 0, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
    ], dtype=torch.float32)

    q_vals, ssl_states = model(dummy_x)

    print(f"\n输入: {dummy_x.shape}")
    print(f"分布 Q 值: {q_vals.shape}  (batch=2, 动作=4, 分位数={model.num_quant})")
    print(f"SSL 预测:  {ssl_states.shape}  (batch=2, 动作=4, 棋盘=16)")
    print(f"期望 Q (取分位数均值): {q_vals.mean(dim=-1).shape}  (batch=2, 动作=4)")
