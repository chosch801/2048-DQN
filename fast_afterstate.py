"""Optional Numba acceleration for V3 deterministic board transitions.

Only the small, deterministic slide kernel is accelerated here.  Search
policies remain outside the training path and can reuse the same helpers later.
When Numba is unavailable, every public function falls back to the reference
implementation in ``afterstate.py``.
"""

from __future__ import annotations

import numpy as np

from afterstate import (
    augment_observation_pairs,
    board_to_observation,
    boards_to_observations,
    observation_to_board,
    observations_to_boards,
    reward_from_log_merge_score,
    reward_from_merge_score,
    reward_from_transition,
    rewards_from_log_merge_scores,
    rewards_from_merge_scores,
    rewards_from_transitions,
    slide_board as reference_slide_board,
    spawn_outcomes,
    spawn_random_tile,
)

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without optional Numba
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _slide_board_numba(board: np.ndarray, action: int):
        rotated = np.zeros((4, 4), dtype=board.dtype)
        if action == 2:  # left
            for row in range(4):
                for column in range(4):
                    rotated[row, column] = board[row, column]
        elif action == 0:  # up -> rotate counter-clockwise
            for row in range(4):
                for column in range(4):
                    rotated[row, column] = board[column, 3 - row]
        elif action == 3:  # right -> rotate 180 degrees
            for row in range(4):
                for column in range(4):
                    rotated[row, column] = board[3 - row, 3 - column]
        elif action == 1:  # down -> rotate clockwise
            for row in range(4):
                for column in range(4):
                    rotated[row, column] = board[3 - column, row]

        slid = np.zeros((4, 4), dtype=board.dtype)
        total_reward = 0
        for row in range(4):
            new_row = np.zeros(4, dtype=board.dtype)
            position = 0
            merged = False
            for column in range(4):
                value = rotated[row, column]
                if value == 0:
                    continue
                if (
                    position > 0
                    and new_row[position - 1] == value
                    and not merged
                ):
                    new_row[position - 1] *= 2
                    total_reward += new_row[position - 1]
                    merged = True
                else:
                    new_row[position] = value
                    position += 1
                    merged = False
            for column in range(4):
                slid[row, column] = new_row[column]

        result = np.zeros((4, 4), dtype=board.dtype)
        if action == 2:  # left
            for row in range(4):
                for column in range(4):
                    result[row, column] = slid[row, column]
        elif action == 0:  # up -> rotate clockwise back
            for row in range(4):
                for column in range(4):
                    result[row, column] = slid[3 - column, row]
        elif action == 3:  # right -> rotate 180 degrees back
            for row in range(4):
                for column in range(4):
                    result[row, column] = slid[3 - row, 3 - column]
        elif action == 1:  # down -> rotate counter-clockwise back
            for row in range(4):
                for column in range(4):
                    result[row, column] = slid[column, 3 - row]

        changed = False
        for row in range(4):
            for column in range(4):
                if board[row, column] != result[row, column]:
                    changed = True
                    break
            if changed:
                break
        return result, total_reward, changed


    @njit(cache=True)
    def _slide_board_batch_numba(boards: np.ndarray, action: int):
        count = boards.shape[0]
        new_boards = np.zeros((count, 4, 4), dtype=boards.dtype)
        rewards = np.zeros(count, dtype=np.int64)
        changed = np.zeros(count, dtype=np.int8)
        for index in range(count):
            new_board, reward, did_change = _slide_board_numba(boards[index], action)
            new_boards[index] = new_board
            rewards[index] = reward
            changed[index] = 1 if did_change else 0
        return new_boards, rewards, changed


    @njit(cache=True)
    def _slide_all_actions_batch_numba(boards: np.ndarray):
        count = boards.shape[0]
        afterstates = np.zeros((count, 4, 4, 4), dtype=boards.dtype)
        rewards = np.zeros((count, 4), dtype=np.int64)
        changed = np.zeros((count, 4), dtype=np.int8)
        for index in range(count):
            for action in range(4):
                afterstate, reward, did_change = _slide_board_numba(
                    boards[index], action
                )
                afterstates[index, action] = afterstate
                rewards[index, action] = reward
                changed[index, action] = 1 if did_change else 0
        return afterstates, rewards, changed


def slide_board(board: np.ndarray, action: int):
    """Fast single-board equivalent of ``afterstate.slide_board``."""

    if action not in (0, 1, 2, 3):
        raise ValueError(f"action must be in [0, 1, 2, 3], got {action}")
    board = np.ascontiguousarray(board, dtype=np.int64).reshape(4, 4)
    if NUMBA_AVAILABLE:
        new_board, reward, changed = _slide_board_numba(board, action)
        return new_board, int(reward), bool(changed)
    return reference_slide_board(board, action)


def slide_board_batch(boards: np.ndarray | list[np.ndarray], action: int):
    """Fast batch equivalent of ``slide_board`` for one action."""

    if isinstance(boards, list):
        if not boards:
            return (
                np.empty((0, 4, 4), dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=bool),
            )
        boards = np.stack(boards)
    boards = np.ascontiguousarray(boards, dtype=np.int64).reshape(-1, 4, 4)
    if NUMBA_AVAILABLE:
        new_boards, rewards, changed = _slide_board_batch_numba(boards, action)
        return new_boards, rewards, changed.astype(bool)

    new_boards = np.empty_like(boards)
    rewards = np.zeros(len(boards), dtype=np.int64)
    changed = np.zeros(len(boards), dtype=bool)
    for index, board in enumerate(boards):
        new_boards[index], rewards[index], changed[index] = slide_board(board, action)
    return new_boards, rewards, changed


def slide_all_actions_batch(boards: np.ndarray | list[np.ndarray]):
    """Return all four afterstates for every board in a single batch call."""

    if isinstance(boards, list):
        if not boards:
            boards = np.empty((0, 4, 4), dtype=np.int64)
        else:
            boards = np.stack(boards)
    boards = np.ascontiguousarray(boards, dtype=np.int64).reshape(-1, 4, 4)
    if NUMBA_AVAILABLE:
        afterstates, rewards, changed = _slide_all_actions_batch_numba(boards)
        return afterstates, rewards, changed.astype(bool)

    count = len(boards)
    afterstates = np.zeros((count, 4, 4, 4), dtype=np.int64)
    rewards = np.zeros((count, 4), dtype=np.int64)
    changed = np.zeros((count, 4), dtype=bool)
    for index, board in enumerate(boards):
        for action in range(4):
            afterstates[index, action], rewards[index, action], changed[index, action] = (
                reference_slide_board(board, action)
            )
    return afterstates, rewards, changed


def valid_actions(board: np.ndarray) -> np.ndarray:
    """Return the legal action mask using the accelerated kernel."""

    _, _, changed = slide_all_actions_batch(np.asarray(board)[None, ...])
    return changed[0]


def action_candidates(
    board: np.ndarray,
    valid_mask: np.ndarray | None = None,
):
    """Return valid ``(action, afterstate, merge_score)`` candidates."""

    afterstates, rewards, changed = slide_all_actions_batch(
        np.asarray(board, dtype=np.int64).reshape(1, 4, 4)
    )
    changed = changed[0]
    if valid_mask is None:
        mask = changed
    else:
        mask = changed & np.asarray(valid_mask, dtype=bool).reshape(4)

    return [
        (action, afterstates[0, action].copy(), int(rewards[0, action]))
        for action in range(4)
        if mask[action]
    ]
