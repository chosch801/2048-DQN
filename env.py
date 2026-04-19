import numpy as np
from game_2048 import Game2048

class Env2048:
    def __init__(self):
        self.game = Game2048()

    def reset(self):
        # Gym 标准接口：返回 obs(状态), info(额外字典信息)
        state = self.game.reset()
        return self._process_state(state), self._get_info()

    def step(self, action):
        old_max_tile = np.max(self.game.board)
        # Gym 标准接口：返回 next_obs, reward, done, truncated, info
        next_state, reward, done = self.game.step(action)
        
        # Base Reward 采用平方根缩放 (Square Root Scaling)
        # 该方法在控制幅度与维持高权重奖赏的区分度中取得平衡
        if reward > 0:
            rl_reward = np.sqrt(reward) / 2.0 
        elif reward < 0:
            rl_reward = -0.5  # 惩罚无效动作 (非法移动)
        else:
            rl_reward = 0.0
            
        # 1. 留空奖励 (维持可下棋步数)
        empty_count = np.sum(next_state == 0)
        rl_reward += 0.01 * empty_count

        max_val = np.max(next_state)
        max_idx = np.argmax(next_state)
        
        # 2. 新最大 Tile (里程碑突破奖励)
        if max_val > old_max_tile:
            # 对合并出新最大值进行额外奖励，激励游戏进度推进
            rl_reward += np.sqrt(max_val)
            
        # 3. 强力战略奖励：角落锁定奖励
        if max_idx in [0, 3, 12, 15] and max_val >= 64:
            # 根据此时最大数字规模给予角落放置位置的战略加成
            rl_reward += np.sqrt(max_val) * 0.1
            
        # 4. 单调性惩罚 (Monotonicity Penalty) - “蛇形排布”策略
        # 顶级策略指出：数字必须延行或列【单向递减】，不能出现波浪形（大数夹在两个小数中间）
        log_board = np.zeros_like(next_state, dtype=np.float32)
        mask = next_state > 0
        log_board[mask] = np.log2(next_state[mask])
        
        # 计算相邻差值
        diff_row = log_board[:, :-1] - log_board[:, 1:]
        diff_col = log_board[:-1, :] - log_board[1:, :]
        
        # 计算相邻方块的绝对差值总和及单向绝对查总和的偏离度，求出单调偏差值
        mono_penalty_row = np.sum(np.abs(diff_row)) - np.sum(np.abs(np.sum(diff_row, axis=1)))
        mono_penalty_col = np.sum(np.abs(diff_col)) - np.sum(np.abs(np.sum(diff_col, axis=0)))
        
        # 增加打破单调性限制带来的直接惩罚项
        rl_reward -= (mono_penalty_row + mono_penalty_col) * 0.02
        
        # 5. 潜在连击奖励 (Merge Potential)
        # 如果行/列上已有成对相邻的数值，则对这部分合并潜力计算加成系数
        merge_row = np.sum((next_state[:, :-1] == next_state[:, 1:]) & (next_state[:, :-1] > 0))
        merge_col = np.sum((next_state[:-1, :] == next_state[1:, :]) & (next_state[:-1, :] > 0))
        rl_reward += (merge_row + merge_col) * 0.05
            
        return self._process_state(next_state), rl_reward, done, False, self._get_info()

    def _process_state(self, state):
        """
        关键的状态表征预处理：
        将 2048 的棋盘面值转换为对数空间（如：2->1, 4->2, ..., 1024->10）
        这使得模型的嵌入层可以直接根据该数字的大小感知量级差距。
        """
        obs = np.zeros_like(state, dtype=np.float32)
        mask = state > 0
        obs[mask] = np.log2(state[mask])
        
        # 将 4x4 矩阵展平为 16 维向量
        # 因为后续我们的 Transformer 打算接收由这16个位置构成的序列
        return obs.flatten()

    def _get_info(self):
        """
        把 Action Mask (动作掩码) 和当前最高数字放在 info 字典中向外界暴露
        """
        return {
            "valid_actions": self.game.get_valid_actions(),
            "score": self.game.score,
            "max_tile": np.max(self.game.board)
        }
        
    def render(self):
        self.game.render()

if __name__ == "__main__":
    print("【RL环境 - Env2048 测试】")
    env = Env2048()
    obs, info = env.reset()
    
    env.render()
    print("\n转换后的观察空间(log2 Flatten):")
    print(obs)
    print("\n可用的动作 Mask [上(W), 下(S), 左(A), 右(D)]:")
    print(info["valid_actions"])