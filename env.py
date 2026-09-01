"""RL environment exposing V3 afterstates."""

from __future__ import annotations

import numpy as np

from afterstate import board_to_observation, reward_from_transition
from game_2048 import Game2048


class Env2048:
    """Gym-like wrapper with log2 observations and score-aligned rewards."""

    def __init__(self, seed: int | None = None):
        self.game = Game2048(seed=seed)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        state = self.game.reset(seed=seed)
        return board_to_observation(state), self._get_info()

    def step(self, action: int):
        next_state, merge_score, done = self.game.step(action)
        afterstate = self.game.last_afterstate
        info = self._get_info(afterstate=afterstate, merge_score=merge_score)
        return (
            board_to_observation(next_state),
            reward_from_transition(merge_score, afterstate),
            done,
            False,
            info,
        )

    def _get_info(
        self,
        afterstate: np.ndarray | None = None,
        merge_score: int = 0,
    ) -> dict:
        if afterstate is None:
            afterstate_observation = None
        else:
            afterstate_observation = board_to_observation(afterstate)

        return {
            "valid_actions": self.game.get_valid_actions(),
            "score": self.game.score,
            "max_tile": int(np.max(self.game.board)),
            "afterstate": afterstate_observation,
            "merge_score": int(merge_score),
            "changed": bool(self.game.last_changed),
        }

    def render(self) -> None:
        self.game.render()
