"""Bounded midpoint snapshots from real decision-state trajectories."""

from __future__ import annotations

from collections import deque

import numpy as np

from fast_afterstate import valid_actions


def validate_restart_board(board: np.ndarray) -> np.ndarray:
    """Return an independent integer board that has a legal continuation."""

    board = np.asarray(board)
    if board.shape != (4, 4) or not np.issubdtype(board.dtype, np.integer):
        raise ValueError("restart board must be a 4 by 4 integer array")
    if np.any(board < 0) or np.any(board == 1) or np.any(board > np.iinfo(np.int64).max):
        raise ValueError("restart tiles must be zero or positive powers of two")
    board = board.astype(np.int64, copy=True)
    if np.any((board & (board - 1)) != 0):
        raise ValueError("restart tiles must be zero or positive powers of two")
    if not np.any(valid_actions(board)):
        raise ValueError("restart board must have a legal action")
    return board


class RestartPool:
    """Keep one midpoint per completed trajectory, evicting oldest first."""

    def __init__(self, capacity: int = 1000):
        if int(capacity) < 1:
            raise ValueError("restart capacity must be at least 1")
        self.capacity = int(capacity)
        self.boards = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.boards)

    def add_trajectory(self, boards: list[np.ndarray]) -> None:
        if len(boards) <= 10:
            return
        midpoint = boards[len(boards) // 2]
        if not np.any(valid_actions(midpoint)):
            return
        self.boards.append(validate_restart_board(midpoint))

    def sample(self) -> np.ndarray | None:
        if not self.boards:
            return None
        return self.boards[int(np.random.randint(len(self.boards)))].copy()

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "boards": [board.copy() for board in self.boards],
        }

    def load_state_dict(self, state: dict) -> None:
        capacity = int(state["capacity"])
        if capacity < 1:
            raise ValueError("restart capacity must be at least 1")
        boards = [validate_restart_board(board) for board in state["boards"]]
        if len(boards) > capacity:
            raise ValueError("restart state exceeds its capacity")
        self.capacity = capacity
        self.boards = deque(boards, maxlen=capacity)
