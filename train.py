import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import math
import glob
import matplotlib.pyplot as plt
from collections import deque

from env import Env2048
from model import TransformerDQN
from replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self):
        # 1. 初始化超参数
        self.gamma = 0.99           # 折扣因子
        self.lr = 1e-4              # 学习率
        self.batch_size = 128       # 设定批大小为 128 进行小批量训练
        self.target_update_freq = 1000 # 目标网络硬更新步长频率
        self.ssl_weight = 0.1       # 自监督辅助任务(SSL)损失的比重系数

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"训练设备: {self.device}")

        # 2. 初始化环境、网络、经验池
        self.env = Env2048()
        self.buffer = ReplayBuffer(capacity=100000) # 初始化大规模回放经验池

        
        # 行为网络 (不断被直接由优化器更新)
        self.policy_net = TransformerDQN().to(self.device)
        self.policy_net.train() # 确保主网络置于训练模式
        # 目标网络 (提供稳定的目标 Q 值，定期从 policy_net 拷贝参数)
        self.target_net = TransformerDQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # 目标网络不需要计算梯度

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        # 采用 SmoothL1Loss 处理异常值梯度，防范价值高估
        self.loss_fn = nn.SmoothL1Loss()

        # 3. 探索率 (Epsilon-Greedy - 指数衰减)
        self.epsilon = 1.0          
        self.epsilon_min = 0.1      # 设定探索率最小值保底界限
        self.epsilon_decay = 1200000 # 设定探索率指数衰减时间常数（步数）
        self.total_env_steps = 0  # 记录与环境交互的总物理步数

        # SSL的辅助任务比重
        self.ssl_weight = 0.1

        # 模型自动保存机制 (确保保存在 2048 项目的子文件夹下)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(base_dir, "models")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.step_count = 0
        self.start_episode = 0

        # 经验池存档路径
        self.buffer_path = os.path.join(self.save_dir, "replay_buffer.npz")
        
        # === 新增：保存所有训练指标的历史记录 ===
        self.history = {
            'episodes': [],
            'scores': [],
            'max_tiles': [],
            'epsilons': [],
            'rewards': [],
            'dqn_losses': [],
            'ssl_losses': [],
            'mean_qs': [],
            'episode_lengths': []  # 收录每局的存活步数
        }
        
        # 加载历史模型进行断点续训
        self._load_checkpoint()

    def _load_checkpoint(self):
        """检查本地模型快照并尝试导入"""
        checkpoints = [f for f in os.listdir(self.save_dir) if f.startswith("dqn_ep") and f.endswith(".pth")]
        if not checkpoints:
            return
            
        # 根据后缀数字定位最新的 pth 文件
        latest_ckpt = max(checkpoints, key=lambda x: int(x.split('ep')[1].split('.pth')[0]))
        ckpt_path = os.path.join(self.save_dir, latest_ckpt)
        
        try:
            # 读取存档权重格式
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # 新版本存档：包含模型、优化器状态及精确步数
                self.policy_net.load_state_dict(checkpoint["model_state_dict"])
                if "optimizer_state_dict" in checkpoint:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.start_episode = int(latest_ckpt.split('ep')[1].split('.pth')[0])
                self.total_env_steps = checkpoint["total_env_steps"]
                if "step_count" in checkpoint:
                    self.step_count = checkpoint["step_count"]
                
                # 恢复历史指标，用于继续画图
                if "history" in checkpoint:
                    self.history = checkpoint["history"]
            else:
                # 兼容旧版仅保存了网络参数快照的文件
                self.policy_net.load_state_dict(checkpoint)
                self.start_episode = int(latest_ckpt.split('ep')[1].split('.pth')[0])
                self.total_env_steps = self.start_episode * 150  
                
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.update_epsilon()  # 同步当前计算引擎的探索率系数
            
            print(f"\\n>>> [Checkpoint] 成功读取快照： {latest_ckpt}！")            
            # 尝试加载本地 ReplayBuffer 数据以恢复强化特征
            if hasattr(self, 'buffer_path') and os.path.exists(self.buffer_path):
                self.buffer.load(self.buffer_path)
            else:
                print(">>> [Warning] 未发现可用的关联 ReplayBuffer，经验池将从零重新填充。")
            print(f">>> 从第 {self.start_episode+1} Episode 继续，初始 Epsilon={self.epsilon:.4f} <<<\n")
        except Exception as e:
            print(f"读取存档 {latest_ckpt} 读取发生异常: {e}, 自动降级从头开始训练。")

    def update_epsilon(self):
        """采用指数衰减模式逐渐削减 Epsilon 探索概率"""
        # epsilon = end + (start - end) * exp(-step / decay)
        self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * math.exp(-self.total_env_steps / self.epsilon_decay)
        # 前期高探索率干预与最终阈值控制
        if self.total_env_steps < self.epsilon_decay:
            self.epsilon = max(self.epsilon, 0.2)
        else:
            self.epsilon = max(self.epsilon, self.epsilon_min)

    def evaluate(self, num_eval_episodes=5):
        """
        纯评估模式 (Evaluation Mode)：
        以全监督预测验证能力，禁用 Epsilon 的随机搜索机制。
        """
        self.policy_net.eval() # 关闭 Dropout 和 Batch Normalization 等扰动项
        eval_scores = []
        eval_max_tiles = []
        
        with torch.no_grad():
            for _ in range(num_eval_episodes):
                state, info = self.env.reset()
                valid_actions = info["valid_actions"]
                while True:
                    # 执行全贪婪决策动作
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values, _ = self.policy_net(state_t)
                    q_values = q_values.cpu().numpy().flatten()
                    
                    # 掩码过滤无效动作
                    q_values[~valid_actions] = -np.inf
                    action = int(np.argmax(q_values))
                    
                    next_state, reward, done, _, next_info = self.env.step(action)
                    valid_actions = next_info["valid_actions"]
                    state = next_state
                    
                    if done:
                        eval_scores.append(next_info["score"])
                        eval_max_tiles.append(next_info["max_tile"])
                        break
                        
        self.policy_net.train() # 恢复训练状态
        avg_score = np.mean(eval_scores)
        avg_tile = np.mean(eval_max_tiles)
        print(f"\\n=======================================================")
        print(f"=== [模型测试评估] (测试 {num_eval_episodes} 局, 零探索率) ===")
        print(f"Average Score (平均分): {avg_score:.0f} | Average Max Tile (平均最大合成): {avg_tile:.0f}")
        print(f"=======================================================\\n")

    def select_action(self, state, valid_actions):
        """
        动作选择 (利用 epsilon-greedy 与 动作掩码)
        """
        # 随机探索
        if random.random() < self.epsilon:
            # 只在合法的动作（True 的位置）中随机挑一个
            valid_indices = np.where(valid_actions)[0]
            if len(valid_indices) == 0:
                return 0 # 理论上不会发生，如果在游戏结束前被调用
            return random.choice(valid_indices)

        # 模型推断最优选择
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            # 测试阶段忽略 SSL 分支预测结果
            q_values, _ = self.policy_net(state_t)
            
            # 剥离 Tensor 返回 Numpy
            q_values = q_values.cpu().numpy().flatten()

            # --- 动作掩码机制 Action Masking ---
            # Mask: 为非法操作设置无限低权重
            q_values[~valid_actions] = -np.inf
            
            return int(np.argmax(q_values))

    def update_model(self):
        """
        从经验池抽取数据并计算 Loss 优化网络
        """
        if len(self.buffer) < self.batch_size:
            return # 容量不足不执行梯度回传

        # ==========================================
        # 抽样
        b_s, b_a, b_r, b_ns, b_d, b_mask, b_next_mask = self.buffer.sample(self.batch_size)
        # 注意：禁止对 reward 执行 -1/1 clamp 裁剪，需维持 Q-value 对大额并列差异度的高度敏感
        
        # ==========================================
        # 分支 A：Double DQN (Huber TD Error)
        # ==========================================
        # 获取主干网络预估 Q 价值
        q_pred, ssl_pred = self.policy_net(b_s)
        # 按索引提取实际采样的 Q(s,a)
        q_current = q_pred.gather(1, b_a)

        # 求解 DDQN 之 Target Q 
        with torch.no_grad():
            # policy_net 执行最优 action 寻找
            q_next_policy, _ = self.policy_net(b_ns)
            q_next_policy[~b_next_mask] = -float('inf') 
            best_actions_next = q_next_policy.argmax(dim=1, keepdim=True)
            
            # target_net 进行解耦平滑估值计算
            q_next_target, _ = self.target_net(b_ns)
            q_max_next = q_next_target.gather(1, best_actions_next)
            
            # 如果达到终止局态则价值清零
            q_max_next[q_next_target.gather(1, best_actions_next) == -float('inf')] = 0.0

            # 计算 TD Target 公式
            q_target = b_r + self.gamma * q_max_next * (1 - b_d)

        # Smooth L1 Error
        loss_dqn = self.loss_fn(q_current, q_target)

        # ==========================================
        # 分支 B：SSL (Next State Reconstruction)
        # ==========================================
        # ssl_pred 维度：[batch, 4_actions, 16_features]
        # 选取符合采样的实际预测动作特征分支
        # batch 维度：(b, 1, 16) -> (b, 16)
        action_indices = b_a.unsqueeze(-1).expand(-1, -1, 16)
        predicted_next_states = ssl_pred.gather(1, action_indices).squeeze(1)

        # 分支 B 损失
        loss_ssl = self.loss_fn(predicted_next_states, b_ns)

        # ==========================================
        # 总损失并反向传播
        # ==========================================
        # 多任务协同加权结合
        total_loss = loss_dqn + self.ssl_weight * loss_ssl

        self.optimizer.zero_grad()
        total_loss.backward()
        # 梯度范数裁剪
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 目标网络定期同步
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return (
            q_current.detach().mean().item(),
            loss_dqn.item(),
            loss_ssl.item()
        )

    def train(self, total_episodes):
        print("====== 正在启动 2048 Transformer DQN 训练循环 ======")
        best_score = 0

        # 根据指定的步次启动 Episode 级训练流程
        for episode in range(self.start_episode + 1, self.start_episode + total_episodes + 1):
            state, info = self.env.reset()
            valid_actions = info["valid_actions"]
            episode_reward = 0
            ep_steps = 0  # 统计 episode 环境交互总步数
            ep_q_vals = []
            ep_dqn_losses = []
            ep_ssl_losses = []
            
            while True:
                # 动作选择
                action = self.select_action(state, valid_actions)
                
                # 环境交互
                next_state, reward, done, _, next_info = self.env.step(action)
                next_valid_actions = next_info["valid_actions"]
                
                # 存入经验回放池
                self.buffer.push(state, action, reward, next_state, done, valid_actions, next_valid_actions)
                
                # 梯度回传与参数更新
                metrics = self.update_model()
                if metrics is not None:
                    ep_q_vals.append(metrics[0])
                    ep_dqn_losses.append(metrics[1])
                    ep_ssl_losses.append(metrics[2])
                
                self.total_env_steps += 1
                ep_steps += 1
                self.update_epsilon()
                
                state = next_state
                valid_actions = next_valid_actions
                episode_reward += reward
                
                if done:
                    break

            final_score = next_info["score"]
            max_tile = next_info["max_tile"]
            mean_ep_q = np.mean(ep_q_vals) if len(ep_q_vals) > 0 else 0.0
            mean_dqn_loss = np.mean(ep_dqn_losses) if len(ep_dqn_losses) > 0 else 0.0
            mean_ssl_loss = np.mean(ep_ssl_losses) if len(ep_ssl_losses) > 0 else 0.0
            
            if final_score > best_score:
                best_score = final_score
            
            # 记录历史指标数据用于画图
            self.history['episodes'].append(episode)
            self.history['scores'].append(final_score)
            self.history['max_tiles'].append(max_tile)
            self.history['epsilons'].append(self.epsilon)
            self.history['rewards'].append(episode_reward)
            self.history['dqn_losses'].append(mean_dqn_loss)
            self.history['ssl_losses'].append(mean_ssl_loss)
            self.history['mean_qs'].append(mean_ep_q)
            self.history['episode_lengths'].append(ep_steps)
            
            # 控制台日志监控 (每 10 轮输出一次)
            if episode % 10 == 0:
                print(f"Epi: {episode} | Score: {final_score:5d} | Max Tile: {max_tile:4d} | Exp Rate: {self.epsilon:.4f} | Rwd: {episode_reward:5.1f} | Len: {ep_steps:4d} | DQN_L: {mean_dqn_loss:.4f} | SSL_L: {mean_ssl_loss:.4f} | Mean Q: {mean_ep_q:.4f}")
            
            # 保存快照、自动评估与绘制数据面板 (每 100 轮)
            if episode % 100 == 0:
                # 自动维护最新权重快照（清理历史权重）
                old_ckpts = glob.glob(os.path.join(self.save_dir, "dqn_ep*.pth"))
                for ckpt in old_ckpts:
                    try:
                        os.remove(ckpt)
                    except:
                        pass
                
                # 备份模型完整状态与参数
                save_path = os.path.join(self.save_dir, f"dqn_ep{episode}.pth")
                torch.save({
                    "model_state_dict": self.policy_net.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "total_env_steps": self.total_env_steps,
                    "step_count": self.step_count,
                    "history": self.history
                }, save_path)
                
                # 同步将经验池通过 numpy 保存
                self.buffer.save(self.buffer_path)
                
                print(f"--> [模型存档] 训练状态与网络参数已保存至 {save_path}")
                
                # 更新训练数据可视化面板
                self.plot_metrics()
                
                # 剔除探索机制进行本地纯贪心策略评估
                self.evaluate(num_eval_episodes=5)

    def plot_metrics(self):
        """更新并在本地保存 matplotlib 分析指标面"""
        if len(self.history['episodes']) == 0:
            return
            
        # 设置绘图底层规格
        fig, axs = plt.subplots(3, 2, figsize=(20, 15))
        fig.suptitle('2048 Transformer DQN Training Metrics', fontsize=20)
        
        eps = self.history['episodes']
        
        # [0, 0] 总分
        axs[0, 0].plot(eps, self.history['scores'], label='Score', alpha=0.4, linewidth=0.5)
        # 添加百轮移动平均线平滑观察实际收益趋势
        if len(eps) >= 100:
            avg_scores = np.convolve(self.history['scores'], np.ones(100)/100, mode='valid')
            axs[0, 0].plot(eps[99:], avg_scores, color='blue', linewidth=2, label='Avg(100)')
        axs[0, 0].set_title('Episode Score (Environment)', fontsize=14)
        axs[0, 0].legend()
        
        # [0, 1] 最大方块
        # 散点图绘制单局环境最大方块统计 (对数坐标)
        axs[0, 1].plot(eps, self.history['max_tiles'], label='Max Tile', color='orange', alpha=0.3, marker='.', markersize=1, linestyle='none')
        axs[0, 1].set_title('Max Tile Reached', fontsize=14)
        axs[0, 1].set_yscale('log', base=2)
        axs[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
        axs[0, 1].legend()
        
        # [1, 0] 强化学习累计 Reward
        axs[1, 0].plot(eps, self.history['rewards'], label='Episode Reward', color='green', alpha=0.4, linewidth=0.5)
        axs[1, 0].set_title('RL Episode Cumulative Reward', fontsize=14)
        
        # [1, 1] 动态探索率
        axs[1, 1].plot(eps, self.history['epsilons'], label='Epsilon', color='red', linewidth=1.5)
        axs[1, 1].set_title('Exploration Rate (Epsilon)', fontsize=14)
        
        
        # [2, 0] 损失监控: DQN主线与SSL次线
        axs[2, 0].plot(eps, self.history['dqn_losses'], label='DQN Loss', alpha=0.8, linewidth=0.5)
        axs[2, 0].plot(eps, self.history['ssl_losses'], label='SSL Loss (Representation)', alpha=0.8, linewidth=0.5)
        axs[2, 0].set_title('Training Losses', fontsize=14)
        axs[2, 0].set_yscale('log') # 损失差距大时使用对数轴
        axs[2, 0].legend()
        
        # [2, 1] 平均 Q值
        axs[2, 1].plot(eps, self.history['mean_qs'], label='Mean Q Value', color='purple', linewidth=1.5)
        axs[2, 1].set_title('Value Network: Expected Mean Q', fontsize=14)
        
        plt.tight_layout()
        # 写入高清文件
        img_path = os.path.join(self.save_dir, "training_dashboard.png")
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"--> [面板与评估更新] 绘图测试完毕。可视化走势图已写入 {img_path}")

if __name__ == "__main__":
    # 配置入口
    agent = DQNAgent()
    
    # 启动训练挂载 (总训练轮次控制)
    agent.train(total_episodes=50000) # 可惜到30000轮时已经局部收敛了，后续需要引入课程奖励机制与更复杂的探索策略来突破瓶颈。
