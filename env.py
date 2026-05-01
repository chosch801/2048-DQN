"""2048 强化学习环境封装 (Gym 风格接口)。

奖励函数（简化后）：
  - 合并得分 sqrt 缩放: √score / 2 — 原生奖励的平滑缩放，保持高频小幅合并与大合并的区分度
  - 空格保留奖: +0.01 × 空格数 — 极弱引导，鼓励模型维持操作空间
  - 无效动作惩罚: −0.5 — 防止无意义的原地踏步

被移除的手工项（让模型自主发现策略）：
  - 里程碑突破奖励
  - 角落锁定奖励
  - 单调性惩罚
  - 合并潜力奖励

状态预处理：
  棋盘值 → log2 映射 → 展平为 16 维向量。
  例如: [2, 4, 8, ...] → [1, 2, 3, ...] ∈ [0, 15]
"""

import numpy as np
from game_2048 import Game2048


class Env2048:
    def __init__(self):
        self.game = Game2048()

    def reset(self):
        state = self.game.reset()
        return self._process_state(state), self._get_info()

    def step(self, action):
        """执行动作，返回 (next_obs, reward, done, truncated, info)。"""
        next_state, reward, done = self.game.step(action)

        # ── 奖励计算（简化为两项） ──────────────────────────

        # 1. 原生合并得分: 正分 → sqrt 缩放, 负分 → 无效动作惩罚
        if reward > 0:
            rl_reward = np.sqrt(reward) / 2.0       # √256 / 2 = 8.0（大合并不膨胀）
        elif reward < 0:
            rl_reward = -0.5                        # 浪费步数轻微惩罚
        else:
            rl_reward = 0.0

        # 2. 弱空格引导: 每空 +0.01，提供最基本的生存空间直觉
        empty_count = np.sum(next_state == 0)
        rl_reward += 0.01 * empty_count

        return self._process_state(next_state), rl_reward, done, False, self._get_info()

    def _process_state(self, state):
        """棋盘值 → log2 映射 → 展平 16 维。"""
        obs = np.zeros_like(state, dtype=np.float32)
        mask = state > 0
        obs[mask] = np.log2(state[mask])
        return obs.flatten()

    def _get_info(self):
        return {
            "valid_actions": self.game.get_valid_actions(),
            "score": self.game.score,
            "max_tile": np.max(self.game.board),
        }

    def render(self):
        self.game.render()


if __name__ == "__main__":
    print("【RL环境 - Env2048 测试（简化奖励函数）】")
    env = Env2048()
    obs, info = env.reset()
    env.render()
    print("\n转换后观察空间 (log2 + 展平):", obs)
    print("合法动作 [上 下 左 右]:", info["valid_actions"])
