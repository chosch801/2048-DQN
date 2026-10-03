"""Prioritized replay buffer for V3 afterstate transitions."""

from __future__ import annotations

import os

import numpy as np


class SumTree:
    """A compact sum tree supporting prioritized sampling and updates."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.tree_size = 1
        while self.tree_size < self.capacity:
            self.tree_size *= 2
        self.tree = np.zeros(2 * self.tree_size, dtype=np.float64)
        self.data = np.zeros(self.capacity, dtype=np.int64)
        self.write = 0
        self.size = 0

    def total(self) -> float:
        return float(self.tree[1])

    def add(self, priority: float, data_index: int) -> None:
        tree_index = self.write + self.tree_size
        self.data[self.write] = data_index
        self._update(tree_index, priority)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_index: int, priority: float) -> None:
        self._update(tree_index, priority)

    def _update(self, tree_index: int, priority: float) -> None:
        delta = float(priority) - self.tree[tree_index]
        self.tree[tree_index] = float(priority)
        tree_index //= 2
        while tree_index >= 1:
            self.tree[tree_index] += delta
            tree_index //= 2

    def sample(self, value: float) -> tuple[int, int]:
        tree_index = 1
        while tree_index < self.tree_size:
            left = tree_index * 2
            if value <= self.tree[left]:
                tree_index = left
            else:
                value -= self.tree[left]
                tree_index = left + 1
        return tree_index, int(self.data[tree_index - self.tree_size])

    def sample_batch(
        self,
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Traverse many tree values in parallel."""

        values = np.asarray(values, dtype=np.float64).copy()
        tree_indices = np.ones(len(values), dtype=np.int64)
        levels = self.tree_size.bit_length() - 1
        for _ in range(levels):
            left_indices = tree_indices * 2
            left_values = self.tree[left_indices]
            go_right = values > left_values
            values = np.where(go_right, values - left_values, values)
            tree_indices = np.where(
                go_right,
                left_indices + 1,
                left_indices,
            )
        data_indices = self.data[tree_indices - self.tree_size]
        return tree_indices, data_indices

    def max_priority(self) -> float:
        if self.size == 0:
            return 1.0
        leaves = self.tree[self.tree_size:self.tree_size + self.capacity]
        return float(max(np.max(leaves), 1e-3))


class ReplayBuffer:
    """PER storage for state, afterstate, reward and sampled next state."""

    def __init__(
        self,
        capacity: int = 500_000,
        state_dim: int = 16,
        action_dim: int = 4,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 1e-5,
        eps: float = 1e-6,
    ):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.afterstate = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.action = np.zeros(self.capacity, dtype=np.int64)
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        self.next_state = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.done = np.zeros(self.capacity, dtype=np.float32)
        self.mask = np.zeros((self.capacity, self.action_dim), dtype=bool)
        self.next_mask = np.zeros((self.capacity, self.action_dim), dtype=bool)

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.beta_increment = float(beta_increment)
        self.eps = float(eps)
        self.tree = SumTree(self.capacity)

    def __len__(self) -> int:
        return self.size

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        afterstate: np.ndarray,
        next_state: np.ndarray,
        done: bool,
        mask: np.ndarray,
        next_mask: np.ndarray,
    ) -> None:
        index = self.ptr
        self.state[index] = state
        self.afterstate[index] = afterstate
        self.action[index] = int(action)
        self.reward[index] = float(reward)
        self.next_state[index] = next_state
        self.done[index] = float(done)
        self.mask[index] = mask
        self.next_mask[index] = next_mask

        self.tree.add(self.tree.max_priority(), index)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def push_batch(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        afterstate: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        mask: np.ndarray,
        next_mask: np.ndarray,
    ) -> None:
        """Append a small batch while keeping SumTree priorities in order."""

        state = np.asarray(state, dtype=np.float32)
        count = len(state)
        if count == 0:
            return
        if count > self.capacity:
            raise ValueError("push_batch cannot exceed replay capacity")

        indices = (self.ptr + np.arange(count)) % self.capacity
        self.state[indices] = state
        self.afterstate[indices] = np.asarray(afterstate, dtype=np.float32)
        self.action[indices] = np.asarray(action, dtype=np.int64)
        self.reward[indices] = np.asarray(reward, dtype=np.float32)
        self.next_state[indices] = np.asarray(next_state, dtype=np.float32)
        self.done[indices] = np.asarray(done, dtype=np.float32)
        self.mask[indices] = np.asarray(mask, dtype=bool)
        self.next_mask[indices] = np.asarray(next_mask, dtype=bool)

        priority = self.tree.max_priority()
        for data_index in indices:
            self.tree.add(priority, int(data_index))
        self.ptr = int((self.ptr + count) % self.capacity)
        self.size = min(self.size + count, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if self.size == 0:
            raise ValueError("cannot sample from an empty replay buffer")

        actual_batch = min(int(batch_size), self.size)
        total_priority = self.tree.total()
        if not np.isfinite(total_priority):
            raise FloatingPointError("replay priority total is not finite")
        indices = np.empty(actual_batch, dtype=np.int64)
        tree_indices = np.empty(actual_batch, dtype=np.int64)
        weights = np.empty(actual_batch, dtype=np.float32)

        if total_priority <= 0:
            data_indices = np.random.randint(0, self.size, size=actual_batch)
            indices[:] = data_indices
            tree_indices.fill(-1)
            weights.fill(1.0)
        else:
            segment = total_priority / actual_batch
            self.beta = min(1.0, self.beta + self.beta_increment)
            lower = segment * np.arange(actual_batch, dtype=np.float64)
            upper = segment * (np.arange(actual_batch, dtype=np.float64) + 1.0)
            values = np.random.uniform(lower, upper)
            values = np.minimum(values, total_priority - 1e-12)
            sampled_tree_indices, sampled_data_indices = self.tree.sample_batch(
                values
            )
            tree_indices[:] = sampled_tree_indices
            indices[:] = sampled_data_indices

            probabilities = (
                self.tree.tree[sampled_tree_indices] / total_priority
            )
            weights[:] = (
                self.size * np.maximum(probabilities, 1e-12)
            ) ** (-self.beta)

            max_weight = float(np.max(weights))
            if max_weight > 0:
                weights /= max_weight

        return {
            "state": self.state[indices].copy(),
            "afterstate": self.afterstate[indices].copy(),
            "action": self.action[indices].copy(),
            "reward": self.reward[indices].copy(),
            "next_state": self.next_state[indices].copy(),
            "done": self.done[indices].copy(),
            "mask": self.mask[indices].copy(),
            "next_mask": self.next_mask[indices].copy(),
            "tree_indices": tree_indices,
            "is_weights": weights,
        }

    def update_priorities(
        self,
        tree_indices: np.ndarray,
        td_errors: np.ndarray,
    ) -> None:
        tree_indices = np.asarray(tree_indices, dtype=np.int64)
        td_errors = np.asarray(td_errors, dtype=np.float64)
        valid = tree_indices >= 0
        if not np.any(valid):
            return

        leaves = tree_indices[valid]
        priorities = (np.abs(td_errors[valid]) + self.eps) ** self.alpha
        if not np.all(np.isfinite(priorities)):
            raise FloatingPointError("nonfinite TD priorities; replay tree was not modified")
        unique_leaves, inverse = np.unique(leaves, return_inverse=True)
        combined_priorities = np.zeros(len(unique_leaves), dtype=np.float64)
        np.maximum.at(combined_priorities, inverse, priorities)
        self.tree.tree[unique_leaves] = combined_priorities

        parents = np.unique(unique_leaves // 2)
        parents = parents[parents > 0]
        while len(parents):
            self.tree.tree[parents] = (
                self.tree.tree[parents * 2]
                + self.tree.tree[parents * 2 + 1]
            )
            parents = np.unique(parents // 2)
            parents = parents[parents > 0]

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        np.savez_compressed(
            filepath,
            state=self.state[:self.size],
            afterstate=self.afterstate[:self.size],
            action=self.action[:self.size],
            reward=self.reward[:self.size],
            next_state=self.next_state[:self.size],
            done=self.done[:self.size],
            mask=self.mask[:self.size],
            next_mask=self.next_mask[:self.size],
            tree=self.tree.tree,
            tree_data=self.tree.data,
            tree_write=np.array([self.tree.write]),
            tree_size=np.array([self.tree.size]),
            beta=np.array([self.beta]),
            ptr=np.array([self.ptr]),
            size=np.array([self.size]),
        )

    def load(self, filepath: str) -> None:
        data = np.load(filepath, allow_pickle=False)
        stored_size = int(data["size"][0])
        if stored_size > self.capacity:
            raise ValueError(
                f"checkpoint replay has {stored_size} entries, "
                f"but current capacity is {self.capacity}"
            )

        self.size = stored_size
        self.ptr = int(data["ptr"][0])
        self.state[:self.size] = data["state"]
        self.afterstate[:self.size] = data["afterstate"]
        self.action[:self.size] = data["action"]
        self.reward[:self.size] = data["reward"]
        self.next_state[:self.size] = data["next_state"]
        self.done[:self.size] = data["done"]
        self.mask[:self.size] = data["mask"]
        self.next_mask[:self.size] = data["next_mask"]

        if "tree" in data:
            if data["tree"].shape != self.tree.tree.shape:
                raise ValueError("replay SumTree shape does not match capacity")
            self.tree.tree[:] = data["tree"]
            self.tree.data[:] = data["tree_data"]
            self.tree.write = int(data["tree_write"][0])
            self.tree.size = int(data["tree_size"][0])
            self.beta = float(data["beta"][0])
        else:
            # A safe fallback for a data-only archive: new entries are all
            # assigned the same priority and can still be sampled.
            self.tree = SumTree(self.capacity)
            for index in range(self.size):
                self.tree.add(1.0, index)
