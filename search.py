"""Exact two-player-move Expectimax for evaluation with an afterstate critic."""

from __future__ import annotations

import numpy as np
import torch

from fast_afterstate import (
    rewards_from_log_merge_scores,
    rewards_from_transitions,
    slide_all_actions_batch,
    spawn_outcomes,
)
from policy import score_actions_batch


@torch.inference_mode()
def score_expectimax_actions_batch(
    model: torch.nn.Module,
    boards: np.ndarray,
    valid_masks: np.ndarray,
    device: torch.device,
    gamma: float,
    reward_mode: str = "transition",
    inference_batch_size: int = 16384,
) -> np.ndarray:
    """Return r + gamma * E_spawn[max_a(r_a + gamma * V(afterstate_a))].

    Depth two counts player moves, including the root move. The intervening
    chance node enumerates every empty cell and both tiles exactly. Terminal
    spawned states contribute zero. Supply an FP32 model in evaluation mode;
    no training policy or model state is changed. Model calls contain at most
    ``inference_batch_size`` leaf afterstates (minimum four).
    """
    if inference_batch_size < 4:
        raise ValueError("inference_batch_size must be at least 4")
    if reward_mode not in ("transition", "log"):
        raise ValueError(f"unsupported reward_mode: {reward_mode}")
    boards = np.ascontiguousarray(boards, dtype=np.int64).reshape(-1, 4, 4)
    valid_masks = np.asarray(valid_masks, dtype=bool).reshape(len(boards), 4)
    scores = np.full((len(boards), 4), -np.inf, dtype=np.float32)
    afterstates, merge_scores, changed = slide_all_actions_batch(boards)
    locations = np.argwhere(changed & valid_masks)
    if len(locations) == 0:
        return scores
    selected = afterstates[locations[:, 0], locations[:, 1]]
    merges = merge_scores[locations[:, 0], locations[:, 1]]
    if reward_mode == "log":
        rewards = rewards_from_log_merge_scores(merges)
    else:
        rewards = rewards_from_transitions(merges, selected)

    states, probabilities, parents = [], [], []
    for parent, afterstate in enumerate(selected):
        for probability, state in spawn_outcomes(afterstate):
            states.append(state)
            probabilities.append(probability)
            parents.append(parent)
    states = np.stack(states)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    parents = np.asarray(parents, dtype=np.int64)
    expected_values = np.zeros(len(selected), dtype=np.float64)
    # Four candidate actions per state bound the network's leaf batch size.
    state_batch_size = inference_batch_size // 4
    with torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, len(states), state_batch_size):
            stop = start + state_batch_size
            batch = states[start:stop]
            _, _, child_legal = slide_all_actions_batch(batch)
            child_scores = score_actions_batch(
                model, batch, child_legal,
                device, gamma, reward_mode=reward_mode,
            )
            best = child_scores.max(axis=1)
            terminal = ~child_legal.any(axis=1)
            best[terminal] = 0.0
            if not np.isfinite(best).all():
                raise FloatingPointError("nonfinite Expectimax leaf values")
            np.add.at(expected_values, parents[start:stop], probabilities[start:stop] * best)
    scores[locations[:, 0], locations[:, 1]] = rewards + gamma * expected_values
    return scores


def choose_expectimax_actions_batch(
    model: torch.nn.Module,
    boards: np.ndarray,
    valid_masks: np.ndarray,
    device: torch.device,
    gamma: float,
    active_mask: np.ndarray | None = None,
    reward_mode: str = "transition",
    inference_batch_size: int = 16384,
) -> np.ndarray:
    """Choose legal search actions; inactive environments return action zero."""
    boards = np.ascontiguousarray(boards, dtype=np.int64).reshape(-1, 4, 4)
    valid_masks = np.asarray(valid_masks, dtype=bool).reshape(len(boards), 4)
    active = np.ones(len(boards), dtype=bool) if active_mask is None else np.asarray(
        active_mask, dtype=bool
    ).reshape(len(boards))
    actions = np.zeros(len(boards), dtype=np.int64)
    if active.any():
        scores = score_expectimax_actions_batch(
            model, boards[active], valid_masks[active], device, gamma,
            reward_mode=reward_mode, inference_batch_size=inference_batch_size,
        )
        if np.isneginf(scores).all(axis=1).any():
            raise RuntimeError("cannot choose an action from a terminal board")
        actions[active] = scores.argmax(axis=1)
    return actions
