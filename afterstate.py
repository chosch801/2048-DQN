"""Deterministic 2048 transitions used by the V3 agent.

The important distinction in V3 is:

    player action -> deterministic afterstate -> random tile spawn -> next state

Keeping these operations separate lets the value network learn the value of an
afterstate and lets later inference code add Expectimax without maintaining a
second copy of the game rules.
"""

from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BOARD_SIZE = 4
ACTION_COUNT = 4
# 0=up, 1=down, 2=left, 3=right.  The rotations convert every action to a
# common left-slide operation and then rotate the result back.
ACTION_ROTATIONS = {2: 0, 0: 1, 3: 2, 1: 3}


def _make_symmetry_index_maps() -> np.ndarray:
    base = np.arange(BOARD_SIZE * BOARD_SIZE).reshape(BOARD_SIZE, BOARD_SIZE)
    maps = []
    for rotation in range(4):
        rotated = np.rot90(base, k=rotation)
        maps.append(rotated.reshape(-1))
        maps.append(np.fliplr(rotated).reshape(-1))
    return np.stack(maps).astype(np.int64)


SYMMETRY_INDEX_MAPS = _make_symmetry_index_maps()


def slide_board(board: np.ndarray, action: int) -> tuple[np.ndarray, int, bool]:
    """Return ``(afterstate, merge_score, changed)`` without spawning a tile."""

    board = np.asarray(board, dtype=np.int64).reshape(BOARD_SIZE, BOARD_SIZE)
    if action not in ACTION_ROTATIONS:
        raise ValueError(f"action must be in [0, 1, 2, 3], got {action}")

    rot_k = ACTION_ROTATIONS[action]
    rotated = np.rot90(board, k=rot_k)
    result = np.zeros_like(rotated)
    total_score = 0

    for row_index in range(BOARD_SIZE):
        row = rotated[row_index]
        non_zero = row[row != 0]
        merged: list[int] = []
        index = 0

        while index < len(non_zero):
            if index + 1 < len(non_zero) and non_zero[index] == non_zero[index + 1]:
                merged_value = int(non_zero[index]) * 2
                merged.append(merged_value)
                total_score += merged_value
                index += 2
            else:
                merged.append(int(non_zero[index]))
                index += 1

        if merged:
            result[row_index, :len(merged)] = merged

    afterstate = np.rot90(result, k=-rot_k).copy()
    changed = not np.array_equal(board, afterstate)
    return afterstate, int(total_score), changed


def valid_actions(board: np.ndarray) -> np.ndarray:
    """Return a boolean mask for actions that change the board."""

    return np.array(
        [slide_board(board, action)[2] for action in range(ACTION_COUNT)],
        dtype=bool,
    )


def action_candidates(
    board: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> list[tuple[int, np.ndarray, int]]:
    """Return valid ``(action, afterstate, merge_score)`` candidates."""

    if valid_mask is None:
        valid_mask = valid_actions(board)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool).reshape(ACTION_COUNT)

    candidates = []
    for action in range(ACTION_COUNT):
        if not valid_mask[action]:
            continue
        afterstate, merge_score, changed = slide_board(board, action)
        if changed:
            candidates.append((action, afterstate, merge_score))
    return candidates


def spawn_random_tile(board: np.ndarray, rng: random.Random) -> np.ndarray:
    """Return a copy with one random 2/4 tile added."""

    result = np.asarray(board, dtype=np.int64).reshape(BOARD_SIZE, BOARD_SIZE).copy()
    empty_cells = list(zip(*np.where(result == 0)))
    if not empty_cells:
        return result

    row, column = rng.choice(empty_cells)
    result[row, column] = 4 if rng.random() < 0.1 else 2
    return result


def spawn_outcomes(board: np.ndarray) -> Iterable[tuple[float, np.ndarray]]:
    """Enumerate all random-spawn outcomes from an afterstate.

    Each empty cell receives a 2 with probability ``0.9 / n`` or a 4 with
    probability ``0.1 / n``.  This helper is intentionally independent of
    Expectimax; it is part of the transition model and can be reused by the
    later inference module.
    """

    board = np.asarray(board, dtype=np.int64).reshape(BOARD_SIZE, BOARD_SIZE)
    empty_cells = list(zip(*np.where(board == 0)))
    if not empty_cells:
        yield 1.0, board.copy()
        return

    cell_probability = 1.0 / len(empty_cells)
    for row, column in empty_cells:
        state_2 = board.copy()
        state_2[row, column] = 2
        yield 0.9 * cell_probability, state_2

        state_4 = board.copy()
        state_4[row, column] = 4
        yield 0.1 * cell_probability, state_4


def board_to_observation(board: np.ndarray) -> np.ndarray:
    """Encode tile values as a flattened float32 log2 board."""

    return boards_to_observations(board)[0]


def boards_to_observations(boards: np.ndarray) -> np.ndarray:
    """Vectorized log2 encoding for ``(N, 4, 4)`` boards."""

    boards = np.asarray(boards, dtype=np.int64).reshape(-1, BOARD_SIZE, BOARD_SIZE)
    flat = boards.reshape(len(boards), -1)
    observations = np.zeros((len(boards), BOARD_SIZE * BOARD_SIZE), dtype=np.float32)
    occupied = flat > 0
    observations[occupied] = np.log2(flat[occupied]).astype(np.float32)
    return observations


def observation_to_board(observation: np.ndarray) -> np.ndarray:
    """Decode a V3 log2 observation back to integer tile values."""

    return observations_to_boards(observation)[0]


def observations_to_boards(observations: np.ndarray) -> np.ndarray:
    """Vectorized decoding for ``(N, 16)`` log2 observations."""

    levels = np.rint(np.asarray(observations)).astype(np.int64)
    levels = np.clip(levels, 0, 16).reshape(-1, BOARD_SIZE, BOARD_SIZE)
    boards = np.zeros_like(levels, dtype=np.int64)
    occupied = levels > 0
    boards[occupied] = np.exp2(levels[occupied]).astype(np.int64)
    return boards


def reward_from_merge_score(merge_score: int | float) -> float:
    """Convert a native merge score to the V3 learning reward.

    This is the V2 merge scale.  Keeping it makes larger merges sufficiently
    informative for the critic while Huber/quantile loss limits outliers.
    """

    if merge_score <= 0:
        return 0.0
    return float(np.sqrt(float(merge_score)) / 2.0)


def reward_from_log_merge_score(merge_score: int | float) -> float:
    """Convert a merge score with the archived V3.0 reward scale."""

    if merge_score <= 0:
        return 0.0
    return float(np.log2(float(merge_score) + 1.0))


def rewards_from_merge_scores(merge_scores: np.ndarray) -> np.ndarray:
    """Vectorized version of :func:`reward_from_merge_score`."""

    scores = np.asarray(merge_scores, dtype=np.float32)
    rewards = np.zeros_like(scores, dtype=np.float32)
    positive = scores > 0
    rewards[positive] = np.sqrt(scores[positive]) / 2.0
    return rewards


def rewards_from_log_merge_scores(merge_scores: np.ndarray) -> np.ndarray:
    """Vectorized archived V3.0 merge reward."""

    scores = np.asarray(merge_scores, dtype=np.float32)
    return np.log2(scores + 1.0).astype(np.float32)


def reward_from_transition(
    merge_score: int | float,
    afterstate: np.ndarray,
) -> float:
    """Return merge reward plus a small deterministic empty-cell bonus."""

    empty_afterstate = int(np.count_nonzero(np.asarray(afterstate) == 0))
    empty_after_spawn = max(empty_afterstate - 1, 0)
    return reward_from_merge_score(merge_score) + 0.01 * empty_after_spawn


def rewards_from_transitions(
    merge_scores: np.ndarray,
    afterstates: np.ndarray,
) -> np.ndarray:
    """Vectorized transition reward for one or many afterstates."""

    rewards = rewards_from_merge_scores(merge_scores)
    empty_afterstates = np.sum(np.asarray(afterstates) == 0, axis=(-2, -1))
    return rewards + 0.01 * np.maximum(empty_afterstates - 1, 0).astype(
        np.float32
    )


def augment_observation_pairs(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same random board symmetry to two observation batches."""

    first = np.asarray(first, dtype=np.float32).reshape(-1, 16)
    second = np.asarray(second, dtype=np.float32).reshape(-1, 16)
    if len(first) != len(second):
        raise ValueError("observation batches must have the same length")
    if len(first) == 0:
        return first.copy(), second.copy()

    choices = np.random.randint(len(SYMMETRY_INDEX_MAPS), size=len(first))
    batch_indices = np.arange(len(first))[:, None]
    maps = SYMMETRY_INDEX_MAPS[choices]
    return (
        first[batch_indices, maps].copy(),
        second[batch_indices, maps].copy(),
    )
