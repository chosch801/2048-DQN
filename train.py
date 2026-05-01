"""2048 分布 Dueling DQN 训练主程序。

六项核心改进：
  1. 多步 Double-Q 目标 (n=3) — TD 信号跨多步传播，缓解延迟奖励的信用分配问题
  2. 简化奖励函数 — 仅保留原生合并得分（sqrt 缩放）+ 弱空格引导，其余手工项全部移除
  3. 优先经验回放 (PER) — Sum-Tree 按 TD 误差优先采样 + 重要性采样权重纠正分布偏移
  4. NoisyNet 探索 — 权重注入分解高斯噪声，替代 epsilon-greedy 的盲目随机探索
  5. Dueling 架构 — Q = V(s) + A(s,a) − mean(A)，解耦状态价值与动作优势
  6. 分布强化学习 — 51 个分位数替代标量 Q 值 + Quantile Huber Loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

from env import Env2048
from model import TransformerDQN
from replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self):
        # ── 超参数 ───────────────────────────────────────────────
        self.gamma = 0.99               # 折扣因子
        self.lr = 1e-4                  # Adam 学习率
        self.batch_size = 128           # 小批量大小
        self.target_update_freq = 1000  # 目标网络每 1000 步硬更新一次
        self.ssl_weight = 0.1           # SSL 辅助任务在总损失中的权重
        self.update_freq = 4            # 每 4 步做一次梯度更新，减少冗余计算
        self.num_quant = 51             # 分位数数量，控制回报分布的精度
        self.kappa = 1.0                # Quantile Huber Loss 的 κ 阈值

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"训练设备: {self.device}")

        # ── 环境、经验池、网络 ──────────────────────────────────
        self.env = Env2048()
        self.buffer = ReplayBuffer(capacity=1_000_000)  # 100 万容量，保留多样化的经验

        # 行为网络：参与梯度更新
        self.policy_net = TransformerDQN(num_quant=self.num_quant).to(self.device)
        self.policy_net.train()

        # 目标网络：仅用于计算稳定的 TD 目标，定期从 policy_net 拷贝
        self.target_net = TransformerDQN(num_quant=self.num_quant).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 分位水平 τ_i = (i − 0.5) / N，共 N=51 个
        self.tau = torch.arange(0.5, self.num_quant, 1.0, device=self.device) / self.num_quant

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss(reduction='none')  # reduction='none' 供 PER IS 加权

        self.total_env_steps = 0

        # ── 持久化路径 ──────────────────────────────────────────
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(base_dir, "models")
        os.makedirs(self.save_dir, exist_ok=True)

        self.step_count = 0
        self.start_episode = 0
        self.buffer_path = os.path.join(self.save_dir, "replay_buffer.npz")

        # ── 指标记录（用于画图） ──────────────────────────────────
        self.history = {
            'episodes': [],
            'scores': [],
            'max_tiles': [],
            'rewards': [],
            'dqn_losses': [],
            'ssl_losses': [],
            'mean_qs': [],
            'episode_lengths': [],
        }

        # 尝试加载断点继续训练
        self._load_checkpoint()

    # ─────────────────────────────────────────────────────────────
    #  断点续训
    # ─────────────────────────────────────────────────────────────
    def _load_checkpoint(self):
        checkpoints = [f for f in os.listdir(self.save_dir)
                       if f.startswith("dqn_ep") and f.endswith(".pth")]
        if not checkpoints:
            return

        latest_ckpt = max(checkpoints,
                         key=lambda x: int(x.split('ep')[1].split('.pth')[0]))
        ckpt_path = os.path.join(self.save_dir, latest_ckpt)

        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.policy_net.load_state_dict(checkpoint["model_state_dict"])
                if "optimizer_state_dict" in checkpoint:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.start_episode = int(latest_ckpt.split('ep')[1].split('.pth')[0])
                self.total_env_steps = checkpoint["total_env_steps"]
                if "step_count" in checkpoint:
                    self.step_count = checkpoint["step_count"]
                if "update_freq" in checkpoint:
                    self.update_freq = checkpoint["update_freq"]
                if "history" in checkpoint:
                    self.history = checkpoint["history"]
            else:
                # 兼容仅保存权重快照的旧格式
                self.policy_net.load_state_dict(checkpoint)
                self.start_episode = int(latest_ckpt.split('ep')[1].split('.pth')[0])
                self.total_env_steps = self.start_episode * 150

            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"\n>>> [Checkpoint] 成功读取快照：{latest_ckpt}！")

            if os.path.exists(self.buffer_path):
                self.buffer.load(self.buffer_path)
            else:
                print(">>> [Warning] 未发现可用的 ReplayBuffer，经验池将从零填充。")
            print(f">>> 从第 {self.start_episode + 1} Episode 继续 (NoisyNet 探索) <<<\n")

        except Exception as e:
            print(f"读取存档 {latest_ckpt} 异常: {e}, 自动降级从头开始训练。")

    # ─────────────────────────────────────────────────────────────
    #  纯贪婪评估（关闭 NoisyNet 噪声）
    # ─────────────────────────────────────────────────────────────
    def evaluate(self, num_eval_episodes=5):
        self.policy_net.eval()  # model.eval() 使 FactorizedNoisyLinear 关闭噪声
        eval_scores = []
        eval_max_tiles = []

        with torch.no_grad():
            for _ in range(num_eval_episodes):
                state, info = self.env.reset()
                valid_actions = info["valid_actions"]
                while True:
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_quantiles, _ = self.policy_net(state_t)        # (1, 4, 51)
                    q_expected = q_quantiles.mean(dim=-1).cpu().numpy().flatten()
                    q_expected[~valid_actions] = -np.inf
                    action = int(np.argmax(q_expected))

                    next_state, reward, done, _, next_info = self.env.step(action)
                    valid_actions = next_info["valid_actions"]
                    state = next_state
                    if done:
                        eval_scores.append(next_info["score"])
                        eval_max_tiles.append(next_info["max_tile"])
                        break

        self.policy_net.train()  # 恢复训练模式
        avg_score = np.mean(eval_scores)
        avg_tile = np.mean(eval_max_tiles)
        print(f"\n=======================================================")
        print(f"=== [评估] ({num_eval_episodes} 局, 零探索) ===")
        print(f"Avg Score: {avg_score:.0f} | Avg Max Tile: {avg_tile:.0f}")
        print(f"=======================================================\n")

    # ─────────────────────────────────────────────────────────────
    #  动作选择：NoisyNet 探索 + 分布取均值
    # ─────────────────────────────────────────────────────────────
    def select_action(self, state, valid_actions):
        """始终贪婪选择，探索由网络权重的随机噪声驱动。"""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_quantiles, _ = self.policy_net(state_t)             # (1, 4, N)
            q_expected = q_quantiles.mean(dim=-1).cpu().numpy().flatten()
            q_expected[~valid_actions] = -np.inf
            return int(np.argmax(q_expected))

    # ─────────────────────────────────────────────────────────────
    #  模型更新：分布 Double DQN + 多步 TD + PER
    # ─────────────────────────────────────────────────────────────
    def update_model(self):
        if len(self.buffer) < self.batch_size:
            return

        n_step = 3
        gamma_n = self.gamma ** n_step

        # 优先采样，同时获取 1-step 与 n-step 数据
        (b_s, b_a, _b_r, b_ns, _b_d, b_mask, b_next_mask,
         b_nr, b_nns, b_nd, b_nnm,                       # n-step 数据
         tree_indices, is_weights) = self.buffer.sample(  # PER 数据
            self.batch_size, n_step=n_step, gamma=self.gamma)

        # ═══════════════════════════════════════════════════════
        #  分支 A: 分布 Double DQN + 多步 TD + PER
        # ═══════════════════════════════════════════════════════

        # 当前状态 → 51 个分位数的 Q 值
        q_quantiles, ssl_pred = self.policy_net(b_s)           # (B, 4, N), (B, 4, 16)
        b_a_expand = b_a.unsqueeze(2).expand(-1, -1, self.num_quant)
        q_current = q_quantiles.gather(1, b_a_expand).squeeze(1)  # (B, N)

        with torch.no_grad():
            # --- Double DQN 解耦：policy 选动作, target 估值 ---
            q_next_pol, _ = self.policy_net(b_nns)             # (B, 4, N)
            q_next_tgt, _ = self.target_net(b_nns)             # (B, 4, N)

            # 取分位数均值 → 期望 Q → 选最优动作
            q_next_pol_mean = q_next_pol.mean(dim=-1)          # (B, 4)
            q_next_pol_mean[~b_nnm] = -float('inf')            # 掩码无效动作
            best_actions = q_next_pol_mean.argmax(dim=1, keepdim=True)

            # 取出最优动作对应的目标分位数
            best_exp = best_actions.unsqueeze(2).expand(-1, -1, self.num_quant)
            q_next_target = q_next_tgt.gather(1, best_exp).squeeze(1)  # (B, N)

            # 终止状态 Q 值置零
            q_next_target[b_nd.bool().squeeze(1)] = 0.0

            # 分布 TD 目标: y_j = Σγ^k r_k + γ^n · θ_j(s_{t+n}, a*)
            target_quantiles = b_nr + gamma_n * q_next_target * (1 - b_nd.float())

        # --- Quantile Huber Loss ──────────────────────────
        # δ_{ij} = y_j − θ_i   (B, N, N)，N=51 个分位数两两比较
        td_error = target_quantiles.unsqueeze(1) - q_current.unsqueeze(2)

        # Huber: L_κ(δ) = 0.5δ² (|δ|≤κ), κ(|δ|−0.5κ) (|δ|>κ)
        abs_td = td_error.abs()
        huber = torch.where(
            abs_td <= self.kappa,
            0.5 * td_error ** 2,
            self.kappa * (abs_td - 0.5 * self.kappa),
        )

        # 分位权重: ρ_τ(δ) = |τ − 1_{δ<0}| · L_κ(δ)
        tau_view = self.tau.view(1, self.num_quant, 1)              # (1, N, 1)
        quantile_weight = (tau_view - (td_error.detach() < 0).float()).abs()

        # 逐样本损失 → IS 加权 → 批均值
        per_sample_loss = (quantile_weight * huber).sum(dim=2).sum(dim=1) / self.num_quant
        loss_dqn = (is_weights * per_sample_loss).mean()

        # PER 优先级：基于分位数期望值的 TD 误差
        td_errors_per = (q_current.mean(dim=1) - target_quantiles.mean(dim=1)).abs()

        # ═══════════════════════════════════════════════════════
        #  分支 B: SSL 自监督预测（仍为 1-step）
        # ═══════════════════════════════════════════════════════
        action_indices = b_a.unsqueeze(-1).expand(-1, -1, 16)
        predicted_next_states = ssl_pred.gather(1, action_indices).squeeze(1)
        loss_ssl = self.loss_fn(predicted_next_states, b_ns).mean()

        # ── 总损失 & 反向传播 ───────────────────────────
        total_loss = loss_dqn + self.ssl_weight * loss_ssl

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # PER 优先级更新
        self.buffer.update_priorities(
            tree_indices.cpu().numpy(),
            td_errors_per.detach().cpu().numpy(),
        )

        # 目标网络定期同步（硬更新）
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return (
            q_current.detach().mean().item(),
            loss_dqn.item(),
            loss_ssl.item(),
        )

    # ─────────────────────────────────────────────────────────────
    #  训练主循环
    # ─────────────────────────────────────────────────────────────
    def train(self, target_episodes):
        """训练至 target_episodes 局（含断点续训）。
        例如 target_episodes=9000，已训 5000 局 → 续训 5001~9000。"""
        if self.start_episode >= target_episodes:
            print(f"已到达目标 {target_episodes} 局，无需继续训练。")
            return
        print(f"====== 2048 分布 Dueling DQN 训练: {self.start_episode + 1} → {target_episodes} ======")
        best_score = 0

        for episode in range(self.start_episode + 1, target_episodes + 1):
            state, info = self.env.reset()
            valid_actions = info["valid_actions"]
            episode_reward = 0
            ep_steps = 0
            ep_q_vals = []
            ep_dqn_losses = []
            ep_ssl_losses = []

            while True:
                action = self.select_action(state, valid_actions)
                next_state, reward, done, _, next_info = self.env.step(action)
                next_valid_actions = next_info["valid_actions"]

                self.buffer.push(state, action, reward, next_state, done,
                                 valid_actions, next_valid_actions)

                # 每 update_freq 步更新一次，减少连续数据的冗余梯度
                if self.total_env_steps % self.update_freq == 0:
                    metrics = self.update_model()
                    if metrics is not None:
                        ep_q_vals.append(metrics[0])
                        ep_dqn_losses.append(metrics[1])
                        ep_ssl_losses.append(metrics[2])

                self.total_env_steps += 1
                ep_steps += 1
                state = next_state
                valid_actions = next_valid_actions
                episode_reward += reward

                if done:
                    break

            final_score = next_info["score"]
            max_tile = next_info["max_tile"]
            mean_ep_q = np.mean(ep_q_vals) if ep_q_vals else 0.0
            mean_dqn_loss = np.mean(ep_dqn_losses) if ep_dqn_losses else 0.0
            mean_ssl_loss = np.mean(ep_ssl_losses) if ep_ssl_losses else 0.0

            if final_score > best_score:
                best_score = final_score

            # 记录指标
            self.history['episodes'].append(episode)
            self.history['scores'].append(final_score)
            self.history['max_tiles'].append(max_tile)
            self.history['rewards'].append(episode_reward)
            self.history['dqn_losses'].append(mean_dqn_loss)
            self.history['ssl_losses'].append(mean_ssl_loss)
            self.history['mean_qs'].append(mean_ep_q)
            self.history['episode_lengths'].append(ep_steps)

            if episode % 10 == 0:
                print(f"Epi: {episode:5d} | Score: {final_score:5.0f} | "
                      f"Max Tile: {max_tile:4d} | Rwd: {episode_reward:5.1f} | "
                      f"Len: {ep_steps:4d} | DQN_L: {mean_dqn_loss:.4f} | "
                      f"SSL_L: {mean_ssl_loss:.4f} | Mean Q: {mean_ep_q:.4f}")

            # 每 100 局：保存快照、画图、纯贪心评估
            if episode % 100 == 0:
                # 清理旧快照
                for ckpt in glob.glob(os.path.join(self.save_dir, "dqn_ep*.pth")):
                    try:
                        os.remove(ckpt)
                    except OSError:
                        pass

                save_path = os.path.join(self.save_dir, f"dqn_ep{episode}.pth")
                torch.save({
                    "model_state_dict": self.policy_net.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "total_env_steps": self.total_env_steps,
                    "step_count": self.step_count,
                    "update_freq": self.update_freq,
                    "history": self.history,
                }, save_path)

                self.buffer.save(self.buffer_path)
                print(f"--> [存档] 已保存至 {save_path}")

                self.plot_metrics()
                self.evaluate(num_eval_episodes=5)

    # ─────────────────────────────────────────────────────────────
    #  训练指标可视化面板
    # ─────────────────────────────────────────────────────────────
    def plot_metrics(self):
        if len(self.history['episodes']) == 0:
            return

        fig, axs = plt.subplots(3, 2, figsize=(20, 15))
        fig.suptitle('2048 Distributional Dueling DQN Training Dashboard', fontsize=16)
        eps = self.history['episodes']

        # [0, 0] 每局得分
        axs[0, 0].plot(eps, self.history['scores'], alpha=0.3, linewidth=0.5)
        if len(eps) >= 100:
            avg = np.convolve(self.history['scores'], np.ones(100) / 100, mode='valid')
            axs[0, 0].plot(eps[99:], avg, color='blue', linewidth=2, label='Avg(100)')
        axs[0, 0].set_title('Episode Score', fontsize=14)
        axs[0, 0].legend()

        # [0, 1] 最高方块（对数纵轴）
        axs[0, 1].plot(eps, self.history['max_tiles'], color='orange',
                       alpha=0.3, marker='.', markersize=1, linestyle='none')
        axs[0, 1].set_title('Max Tile Reached', fontsize=14)
        axs[0, 1].set_yscale('log', base=2)
        axs[0, 1].grid(axis='y', linestyle='--', alpha=0.7)

        # [1, 0] 累计奖励
        axs[1, 0].plot(eps, self.history['rewards'], color='green', alpha=0.3, linewidth=0.5)
        axs[1, 0].set_title('Episode Cumulative Reward', fontsize=14)

        # [1, 1] 每局存活步数（替代原来的 epsilon 图）
        if self.history['episode_lengths']:
            axs[1, 1].plot(eps, self.history['episode_lengths'],
                           color='red', alpha=0.3, linewidth=0.5)
            if len(eps) >= 100:
                avg_len = np.convolve(self.history['episode_lengths'],
                                      np.ones(100) / 100, mode='valid')
                axs[1, 1].plot(eps[99:], avg_len, color='darkred', linewidth=2, label='Avg(100)')
        axs[1, 1].set_title('Episode Length (steps survived)', fontsize=14)
        axs[1, 1].legend()

        # [2, 0] DQN + SSL 损失
        axs[2, 0].plot(eps, self.history['dqn_losses'], alpha=0.8, linewidth=0.5, label='DQN Loss')
        axs[2, 0].plot(eps, self.history['ssl_losses'], alpha=0.8, linewidth=0.5, label='SSL Loss')
        axs[2, 0].set_title('Training Losses', fontsize=14)
        axs[2, 0].set_yscale('log')
        axs[2, 0].legend()

        # [2, 1] 期望 Q 值均值
        axs[2, 1].plot(eps, self.history['mean_qs'], color='purple', linewidth=1.5)
        axs[2, 1].set_title('Expected Mean Q Value', fontsize=14)

        plt.tight_layout()
        img_path = os.path.join(self.save_dir, "training_dashboard.png")
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"--> [面板] 可视化走势图已写入 {img_path}")


if __name__ == "__main__":
    agent = DQNAgent()

    # 目标总局数，可根据需要调整
    # 无论中断多少次，每次启动都会自动续训到 10,000
    agent.train(target_episodes=10000)
