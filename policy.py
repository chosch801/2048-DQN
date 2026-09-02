"""Action scoring helpers shared by training and greedy evaluation."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch

from fast_afterstate import (
    action_candidates,
    boards_to_observations,
    reward_from_log_merge_score,
    reward_from_transition,
    rewards_from_log_merge_scores,
    rewards_from_transitions,
    slide_all_actions_batch,
)


@torch.inference_mode()
def score_actions(
    model: torch.nn.Module,
    board: np.ndarray,
    device: torch.device,
    gamma: float,
    valid_mask: np.ndarray | None = None,
    reward_mode: str = "transition",
) -> np.ndarray:
    """Score actions as immediate reward plus discounted afterstate value."""

    scores = np.full(4, -np.inf, dtype=np.float32)
    candidates = action_candidates(board, valid_mask)
    if not candidates:
        return scores

    observations = boards_to_observations(
        np.stack([afterstate for _, afterstate, _ in candidates])
    )
    observation_tensor = torch.as_tensor(
        observations, dtype=torch.float32, device=device
    )
    values = model(observation_tensor).detach().cpu().numpy()

    for value, (action, afterstate, merge_score) in zip(values, candidates):
        if reward_mode == "log":
            immediate_reward = reward_from_log_merge_score(merge_score)
        elif reward_mode == "transition":
            immediate_reward = reward_from_transition(merge_score, afterstate)
        else:
            raise ValueError(f"unsupported reward_mode: {reward_mode}")
        scores[action] = immediate_reward + gamma * float(
            np.asarray(value).mean()
        )
    return scores


def choose_action(
    model: torch.nn.Module,
    board: np.ndarray,
    valid_mask: np.ndarray,
    device: torch.device,
    gamma: float,
    epsilon: float = 0.0,
    reward_mode: str = "transition",
) -> int:
    """Choose a legal epsilon-greedy action using afterstate values."""

    legal_actions = np.flatnonzero(np.asarray(valid_mask, dtype=bool))
    if len(legal_actions) == 0:
        raise RuntimeError("cannot choose an action from a terminal board")

    if epsilon > 0.0 and np.random.random() < epsilon:
        return int(np.random.choice(legal_actions))

    scores = score_actions(
        model,
        board,
        device,
        gamma,
        valid_mask,
        reward_mode=reward_mode,
    )
    return int(np.argmax(scores))


@torch.inference_mode()
def score_actions_batch(
    model: torch.nn.Module,
    boards: np.ndarray,
    valid_masks: np.ndarray,
    device: torch.device,
    gamma: float,
    amp_enabled: bool = False,
    reward_mode: str = "transition",
) -> np.ndarray:
    """Score legal actions for many boards with one shared model call."""

    boards = np.ascontiguousarray(boards, dtype=np.int64).reshape(-1, 4, 4)
    valid_masks = np.asarray(valid_masks, dtype=bool).reshape(-1, 4)
    scores = np.full((len(boards), 4), -np.inf, dtype=np.float32)
    if len(boards) == 0:
        return scores

    afterstates, merge_scores, changed = slide_all_actions_batch(boards)
    legal = changed & valid_masks
    locations = np.argwhere(legal)
    if len(locations) == 0:
        return scores

    board_indices = locations[:, 0]
    action_indices = locations[:, 1]
    observations = boards_to_observations(
        afterstates[board_indices, action_indices]
    )
    observation_tensor = torch.as_tensor(
        observations,
        dtype=torch.float32,
        device=device,
    )
    if amp_enabled and device.type == "cuda":
        context = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        context = nullcontext()
    with context:
        values = model(observation_tensor)
    values = values.float().cpu().numpy()
    values = values.reshape(len(observations), -1).mean(axis=1)
    selected_merge_scores = merge_scores[board_indices, action_indices]
    selected_afterstates = afterstates[board_indices, action_indices]
    if reward_mode == "log":
        immediate_rewards = rewards_from_log_merge_scores(selected_merge_scores)
    elif reward_mode == "transition":
        immediate_rewards = rewards_from_transitions(
            selected_merge_scores,
            selected_afterstates,
        )
    else:
        raise ValueError(f"unsupported reward_mode: {reward_mode}")
    scores[board_indices, action_indices] = (
        immediate_rewards + gamma * values
    ).astype(np.float32)
    return scores


def choose_actions_batch(
    model: torch.nn.Module,
    boards: np.ndarray,
    valid_masks: np.ndarray,
    device: torch.device,
    gamma: float,
    epsilon: float = 0.0,
    active_mask: np.ndarray | None = None,
    amp_enabled: bool = False,
    reward_mode: str = "transition",
) -> np.ndarray:
    """Choose legal epsilon-greedy actions for a batch of boards.

    Random actions are selected without a model call.  Only the exploitative
    subset is sent to the network, which avoids wasting GPU work while the
    early high-exploration phase is still running.
    """

    boards = np.ascontiguousarray(boards, dtype=np.int64).reshape(-1, 4, 4)
    valid_masks = np.asarray(valid_masks, dtype=bool).reshape(-1, 4)
    if active_mask is None:
        active = np.ones(len(boards), dtype=bool)
    else:
        active = np.asarray(active_mask, dtype=bool).reshape(len(boards))

    actions = np.zeros(len(boards), dtype=np.int64)
    active_indices = np.flatnonzero(active)
    if len(active_indices) == 0:
        return actions

    random_flags = np.random.random(len(active_indices)) < float(epsilon)
    exploit_indices = []
    for local_index, board_index in enumerate(active_indices):
        legal_actions = np.flatnonzero(valid_masks[board_index])
        if len(legal_actions) == 0:
            raise RuntimeError("cannot choose an action from a terminal board")
        if random_flags[local_index]:
            actions[board_index] = int(np.random.choice(legal_actions))
        else:
            exploit_indices.append(int(board_index))

    if exploit_indices:
        exploit_indices = np.asarray(exploit_indices, dtype=np.int64)
        scores = score_actions_batch(
            model,
            boards[exploit_indices],
            valid_masks[exploit_indices],
            device,
            gamma,
            amp_enabled=amp_enabled,
            reward_mode=reward_mode,
        )
        actions[exploit_indices] = np.argmax(scores, axis=1)
    return actions
