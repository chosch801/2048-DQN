"""Small, seedable 2048 game engine for V3."""

from __future__ import annotations

import random

import numpy as np

from fast_afterstate import slide_board, spawn_random_tile, valid_actions


class Game2048:
    """2048 game state with deterministic afterstate bookkeeping."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.board = np.zeros((4, 4), dtype=np.int64)
        self.score = 0
        self.game_over = False
        self.last_afterstate = self.board.copy()
        self.last_merge_score = 0
        self.last_changed = False
        self.reset()

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng.seed(seed)

        self.board = np.zeros((4, 4), dtype=np.int64)
        self.score = 0
        self.game_over = False
        self.last_afterstate = self.board.copy()
        self.last_merge_score = 0
        self.last_changed = False
        self.add_new_tile()
        self.add_new_tile()
        return self.get_state()

    def get_state(self) -> np.ndarray:
        return self.board.copy()

    def add_new_tile(self) -> None:
        self.board = spawn_random_tile(self.board, self.rng)

    def slide_only(self, action: int) -> tuple[np.ndarray, int, bool]:
        """Inspect the deterministic afterstate without changing the game."""

        return slide_board(self.board, action)

    def step(self, action: int) -> tuple[np.ndarray, int, bool]:
        """Apply an action and then spawn one random tile.

        The return signature matches the original project.  The deterministic
        afterstate and merge score are also exposed as ``last_*`` attributes so
        the environment can store them without repeating the slide operation.
        """

        if self.game_over:
            return self.get_state(), 0, True

        afterstate, merge_score, changed = slide_board(self.board, action)
        self.last_afterstate = afterstate.copy()
        self.last_merge_score = merge_score
        self.last_changed = changed

        if changed:
            self.board = afterstate
            self.score += merge_score
            self.add_new_tile()

        self.game_over = not np.any(valid_actions(self.board))
        return self.get_state(), merge_score, self.game_over

    def check_game_over(self) -> bool:
        return not np.any(valid_actions(self.board))

    def get_valid_actions(self) -> np.ndarray:
        return valid_actions(self.board)

    def render(self) -> None:
        print("-" * 25)
        for row in self.board:
            row_text = "|" + "|".join(
                f"{int(value):^5}" if value else "     " for value in row
            ) + "|"
            print(row_text)
            print("-" * 25)
        print(f"当前得分 (Score): {self.score}")


if __name__ == "__main__":
    game = Game2048(seed=0)
    game.render()
