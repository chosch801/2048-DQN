"""经验回放池：优先经验回放 (PER) + n-Step 多步回报 + 断点序列化。

SumTree 结构：
  1-索引完全二叉树，根在 index=1，叶节点从 tree_size 开始。
  tree_size 取 ≥ capacity 的最小 2 的幂，保证满树遍历正确。
  - add(priority, data_idx): 新经验以当前最大优先级入池
  - sample(value): 按值在 [0, total) 间遍历树，定位对应叶节点
  - update(tree_idx, priority): 更新优先级并向上传播

n-Step 回报：
  对于 buffer 中的每个采样位置 idx，向前看最多 n 步：
    R^{(n)}_t = Σ_{k=0}^{n-1} γ^k · r_{t+k} + γ^n · Q(s_{t+n}, a*)
  若中途遇到 done，截断并标记 reached_done=True。
"""

import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════
#  SumTree — 优先采样数据结构
# ═══════════════════════════════════════════════════════════════

class SumTree:
    """1-索引完全二叉树，叶节点存优先级，支持 O(log N) 采样与更新。"""

    def __init__(self, capacity):
        self.capacity = capacity
        # 取 ≥ capacity 的最小 2 的幂作为叶子层大小，保证树完全满
        self.tree_size = 1
        while self.tree_size < capacity:
            self.tree_size *= 2
        # tree[1] 是根，叶节点在 [tree_size, 2*tree_size-1]
        self.tree = np.zeros(2 * self.tree_size, dtype=np.float64)
        self.data = np.zeros(capacity, dtype=np.int64)   # 叶节点 → buffer 索引的映射
        self.write = 0                                   # 当前写入的叶节点偏移
        self.size = 0                                    # 实际写入的条目数

    def total(self):
        """树根的值 = 所有叶节点优先级之和。"""
        return self.tree[1]

    def add(self, priority, data_idx):
        """在 write 位置写入新优先级，指向 buffer 中的 data_idx。"""
        tree_idx = self.write + self.tree_size
        self.data[self.write] = data_idx
        self._update(tree_idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx, priority):
        """更新指定叶节点优先级，向上传播差值。"""
        self._update(tree_idx, priority)

    def _update(self, tree_idx, priority):
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        tree_idx //= 2
        while tree_idx >= 1:
            self.tree[tree_idx] += delta
            tree_idx //= 2

    def sample(self, value):
        """给定 [0, total) 的值，返回 (叶节点索引, buffer 索引)。"""
        idx = 1  # 从根开始
        while idx < self.tree_size:          # 未到达叶层
            left = 2 * idx
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1               # right child
        leaf_idx = idx
        data_idx = self.data[leaf_idx - self.tree_size]
        return leaf_idx, data_idx

    def max_priority(self):
        """返回当前最大优先级（新经验入池时使用）。"""
        if self.size == 0:
            return 1.0
        return float(max(np.max(self.tree[self.tree_size:self.tree_size + self.size]), 1e-3))


# ═══════════════════════════════════════════════════════════════
#  ReplayBuffer — 支持 PER + n-Step
# ═══════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, capacity=1_000_000, state_dim=16, action_dim=4,
                 alpha=0.6, beta=0.4, beta_increment=1e-5, eps=1e-6):
        self.capacity = capacity
        self.ptr = 0           # 循环写入指针
        self.size = 0          # 已存储条目数（≤ capacity）

        # 预分配连续内存数组
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, 1), dtype=np.int64)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.mask = np.zeros((capacity, action_dim), dtype=bool)
        self.next_mask = np.zeros((capacity, action_dim), dtype=bool)

        # 全局步号，用于 n-step 采样的时序连续性校验
        self.step_id = np.zeros((capacity, 1), dtype=np.int64)
        self.global_step = 0

        # PER 参数
        self.alpha = alpha                      # 优先级指数 (0=均匀, 1=完全按优先级)
        self.beta = beta                        # IS 权重指数 (0=不修正, 1=完全修正)
        self.beta_increment = beta_increment    # β 每步退火增量
        self.eps = eps                          # 最小优先级（防止零优先级导致永不采样）
        self.tree = SumTree(capacity)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 持久化 ──────────────────────────────────────────────────

    def save(self, filepath):
        np.savez_compressed(filepath,
                            state=self.state[:self.size],
                            action=self.action[:self.size],
                            reward=self.reward[:self.size],
                            next_state=self.next_state[:self.size],
                            done=self.done[:self.size],
                            mask=self.mask[:self.size],
                            next_mask=self.next_mask[:self.size],
                            step_id=self.step_id[:self.size],
                            global_step=np.array([self.global_step]),
                            tree_values=self.tree.tree,
                            tree_data=self.tree.data[:self.tree.size],
                            tree_write=np.array([self.tree.write]),
                            tree_total_size=np.array([self.tree.size]),
                            tree_leaf_base=np.array([self.tree.tree_size]),
                            beta=np.array([self.beta]),
                            ptr=self.ptr,
                            size=self.size)
        print(f"--> [ReplayBuffer] 已将 {self.size} 条经验保存至 {filepath}")

    def load(self, filepath):
        try:
            data = np.load(filepath)
            self.size = int(data['size'])
            self.ptr = int(data['ptr'])
            self.global_step = int(data['global_step'][0]) if 'global_step' in data else self.size

            self.state[:self.size] = data['state']
            self.action[:self.size] = data['action']
            self.reward[:self.size] = data['reward']
            self.next_state[:self.size] = data['next_state']
            self.done[:self.size] = data['done']
            self.mask[:self.size] = data['mask']
            self.next_mask[:self.size] = data['next_mask']
            if 'step_id' in data:
                self.step_id[:self.size] = data['step_id']
            if 'tree_data' in data and 'tree_values' in data:
                tree_data_arr = data['tree_data']
                self.tree.data[:tree_data_arr.shape[0]] = tree_data_arr
                self.tree.tree[:data['tree_values'].shape[0]] = data['tree_values']
                self.tree.write = int(data['tree_write'][0])
                self.tree.size = int(data['tree_total_size'][0])
                if 'tree_leaf_base' in data:
                    self.tree.tree_size = int(data['tree_leaf_base'][0])
                if 'beta' in data:
                    self.beta = float(data['beta'][0])
                else:
                    # 旧格式无 β 字段：经验充足时 β 已饱和至 1.0
                    self.beta = 1.0 if self.size > 100_000 else 0.4

            print(f"--> [ReplayBuffer] 成功恢复了 {self.size} 条数据。")
        except Exception as e:
            print(f"--> [ReplayBuffer] 读取存档失败 ({e})，以空池开局。")

    # ── 写入 ────────────────────────────────────────────────────

    def push(self, state, action, reward, next_state, done, mask, next_mask):
        """存入一条经验。新经验按当前最大优先级入 SumTree。"""
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        self.mask[self.ptr] = mask
        self.next_mask[self.ptr] = next_mask
        self.step_id[self.ptr] = self.global_step
        self.global_step += 1

        # 新经验以当前最大优先级入树（确保每条经验至少被采样一次）
        max_p = self.tree.max_priority()
        self.tree.add(max_p, self.ptr)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # ── 优先采样（支持 n-step 回报） ───────────────────────────

    def sample(self, batch_size, n_step=1, gamma=0.99):
        """按优先概率采样，同时计算 n-step 回报和 IS 权重。

        Returns: 13 个张量的元组
            0-6: 1-step 数据 (s, a, r, ns, d, mask, next_mask)
            7-10: n-step 数据 (n_step_reward, n_step_state, n_step_done, n_step_mask)
            11:  tree_indices（用于事后更新优先级）
            12:  is_weights（重要性采样权重）
        """
        if self.size < batch_size:
            batch_size_actual = self.size
        else:
            batch_size_actual = batch_size

        batch_indices = np.zeros(batch_size_actual, dtype=np.int64)
        tree_indices = np.zeros(batch_size_actual, dtype=np.int64)
        is_weights = np.zeros(batch_size_actual, dtype=np.float32)

        # 按段采样：将 [0, total) 均分为 batch 段，每段内随机取一点
        segment = self.tree.total() / batch_size_actual

        # β 向 1.0 退火（逐步增强 IS 修正）
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size_actual):
            value = np.random.uniform(segment * i, segment * (i + 1))
            value = min(value, self.tree.total() - 1e-9)   # 防浮点溢出
            tree_idx, data_idx = self.tree.sample(value)
            tree_indices[i] = tree_idx
            batch_indices[i] = data_idx

            # IS 权重: w = (N · P(i))^(−β) / max w
            prob = self.tree.tree[tree_idx] / max(self.tree.total(), 1e-9)
            is_weights[i] = (self.size * prob) ** (-self.beta)

        # IS 权重归一化（除以最大值）
        w_max = np.max(is_weights)
        if w_max > 0:
            is_weights /= w_max

        ind = batch_indices

        # ── n-Step 回报计算 ─────────────────────────────
        if n_step == 1:
            n_step_rewards = self.reward[ind]
            n_step_states = self.next_state[ind]
            n_step_dones = self.done[ind]
            n_step_masks = self.next_mask[ind]
        else:
            n_step_rewards = np.zeros((batch_size_actual, 1), dtype=np.float32)
            n_step_states = np.zeros((batch_size_actual, self.state.shape[1]), dtype=np.float32)
            n_step_dones = np.zeros((batch_size_actual, 1), dtype=np.float32)
            n_step_masks = np.zeros((batch_size_actual, self.mask.shape[1]), dtype=bool)

            for i, idx in enumerate(ind):
                cum_reward = 0.0
                reached_done = False
                actual_steps = 0

                for k in range(n_step):
                    curr_idx = (idx + k) % self.capacity
                    # step_id 校验：确保连续采样不跨越不连续的 buffer 边界
                    if k > 0:
                        prev_idx = (idx + k - 1) % self.capacity
                        if self.step_id[curr_idx] != self.step_id[prev_idx] + 1:
                            break
                    cum_reward += (gamma ** k) * float(self.reward[curr_idx][0])
                    actual_steps = k + 1
                    if self.done[curr_idx]:
                        reached_done = True
                        break

                n_step_rewards[i, 0] = cum_reward
                n_step_dones[i, 0] = float(reached_done)

                if reached_done:
                    # 提前终止 → Q 项在 update_model 中被置零
                    terminal_idx = (idx + actual_steps - 1) % self.capacity
                    n_step_states[i] = self.next_state[terminal_idx]
                    n_step_masks[i] = self.next_mask[terminal_idx]
                else:
                    target_idx = (idx + actual_steps) % self.capacity
                    n_step_states[i] = self.state[target_idx]
                    n_step_masks[i] = self.mask[target_idx]

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),       # 0: s
            torch.LongTensor(self.action[ind]).to(self.device),       # 1: a
            torch.FloatTensor(self.reward[ind]).to(self.device),      # 2: r (1-step)
            torch.FloatTensor(self.next_state[ind]).to(self.device),  # 3: ns (1-step)
            torch.FloatTensor(self.done[ind]).to(self.device),        # 4: d (1-step)
            torch.BoolTensor(self.mask[ind]).to(self.device),         # 5: mask
            torch.BoolTensor(self.next_mask[ind]).to(self.device),    # 6: next_mask
            torch.FloatTensor(n_step_rewards).to(self.device),        # 7: n_step_r
            torch.FloatTensor(n_step_states).to(self.device),         # 8: n_step_ns
            torch.FloatTensor(n_step_dones).to(self.device),          # 9: n_step_d
            torch.BoolTensor(n_step_masks).to(self.device),           # 10: n_step_mask
            torch.LongTensor(tree_indices).to(self.device),           # 11: tree_idx
            torch.FloatTensor(is_weights).to(self.device),            # 12: is_weights
        )

    def update_priorities(self, tree_indices, td_errors):
        """按 |TD 误差|^α 更新优先级。"""
        priorities = (np.abs(td_errors) + self.eps) ** self.alpha
        for tree_idx, priority in zip(tree_indices, priorities):
            self.tree.update(int(tree_idx), float(priority))

    def __len__(self):
        return self.size
