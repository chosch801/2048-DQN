"""Small batched 2048 environment for high-throughput DQN collection."""

from __future__ import annotations

import numpy as np

from fast_afterstate import (
    boards_to_observations,
    rewards_from_merge_scores,
    slide_all_actions_batch,
)
from game_2048 import Game2048


class BatchEnv2048:
    """Run independent games in lockstep while keeping episode boundaries."""

    def __init__(self, num_envs: int, seed: int = 0):
        if int(num_envs) < 1:
            raise ValueError("num_envs must be at least 1")
        self.num_envs = int(num_envs)
        self.seed = int(seed)
        self.games = [
            Game2048(seed=self.seed + index)
            for index in range(self.num_envs)
        ]

    def boards(self) -> np.ndarray:
        """Return the current integer boards as ``(N, 4, 4)``."""

        return np.stack([game.board for game in self.games]).astype(
            np.int64,
            copy=True,
        )

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """Reset every game and return observations plus legal-action masks."""

        for index, game in enumerate(self.games):
            game.reset(seed=self.seed + index)
        boards = self.boards()
        return boards_to_observations(boards), self._valid_masks(boards)

    def _valid_masks(self, boards: np.ndarray) -> np.ndarray:
        _, _, changed = slide_all_actions_batch(boards)
        return changed.astype(bool, copy=False)

    def step(
        self,
        actions: np.ndarray,
        active_mask: np.ndarray | None = None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[dict],
    ]:
        """Apply one action to each active game and spawn random tiles.

        The returned ``afterstate_observations`` has shape ``(N, 4, 16)``;
        the trainer selects ``[environment, action]`` before writing replay.
        Inactive games are left untouched and are marked done.
        """

        actions = np.asarray(actions, dtype=np.int64).reshape(self.num_envs)
        if active_mask is None:
            active = np.ones(self.num_envs, dtype=bool)
        else:
            active = np.asarray(active_mask, dtype=bool).reshape(self.num_envs)

        boards = self.boards()
        afterstates, merge_scores, changed = slide_all_actions_batch(boards)
        active_indices = np.flatnonzero(active)
        if len(active_indices):
            selected_changed = changed[active_indices, actions[active_indices]]
            if not np.all(selected_changed):
                raise ValueError("batch action is not legal for an active board")

        for index in active_indices:
            action = int(actions[index])
            game = self.games[index]
            did_change = bool(changed[index, action])
            game.last_afterstate = afterstates[index, action].copy()
            game.last_merge_score = int(merge_scores[index, action])
            game.last_changed = did_change
            if did_change:
                game.board = afterstates[index, action].copy()
                game.score += int(merge_scores[index, action])
                game.add_new_tile()

        next_boards = self.boards()
        next_masks = self._valid_masks(next_boards)
        dones = ~np.any(next_masks, axis=1)
        dones[~active] = True
        for index in active_indices:
            self.games[index].game_over = bool(dones[index])

        selected_merge_scores = merge_scores[
            np.arange(self.num_envs),
            actions,
        ]
        rewards = rewards_from_merge_scores(selected_merge_scores)
        next_observations = boards_to_observations(next_boards)
        afterstate_observations = boards_to_observations(
            afterstates.reshape(-1, 4, 4)
        ).reshape(self.num_envs, 4, 16)
        infos = []
        for index, game in enumerate(self.games):
            infos.append(
                {
                    "score": int(game.score),
                    "max_tile": int(np.max(game.board)),
                    "valid_actions": next_masks[index].copy(),
                    "merge_score": int(merge_scores[index, actions[index]]),
                    "changed": bool(changed[index, actions[index]]),
                }
            )
        return (
            next_observations,
            rewards,
            dones.astype(bool),
            afterstate_observations,
            next_masks,
            infos,
        )
