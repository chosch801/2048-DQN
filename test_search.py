"""Check exact chance weighting and afterstate search against scalar rules."""

import unittest

import numpy as np
import torch

from afterstate import (
    board_to_observation, reward_from_log_merge_score, reward_from_transition,
    slide_board, spawn_outcomes, valid_actions,
)
from search import choose_expectimax_actions_batch, score_expectimax_actions_batch


class LeafModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.largest_batch = 0

    def forward(self, observations):
        self.largest_batch = max(self.largest_batch, len(observations))
        weights = torch.arange(1, 17, device=observations.device)
        value = (observations * weights).sum(dim=1) / 100
        return torch.stack((value - 1, value + 1), dim=1)


def scalar_reference(board, gamma, mode):
    def reward(merge, afterstate):
        return reward_from_transition(merge, afterstate) if mode == "transition" else reward_from_log_merge_score(merge)

    scores = np.full(4, -np.inf)
    for action in range(4):
        afterstate, merge, changed = slide_board(board, action)
        if not changed:
            continue
        expected = 0.0
        # Independent chance enumeration (not the search helper).
        empty = np.argwhere(afterstate == 0)
        for row, col in empty:
            for tile, probability in ((2, .9 / len(empty)), (4, .1 / len(empty))):
                state = afterstate.copy()
                state[row, col] = tile
                options = []
                for child_action in range(4):
                    leaf, child_merge, child_changed = slide_board(state, child_action)
                    if child_changed:
                        value = np.dot(board_to_observation(leaf), np.arange(1, 17)) / 100
                        options.append(reward(child_merge, leaf) + gamma * value)
                expected += probability * (max(options) if options else 0.0)
        scores[action] = reward(merge, afterstate) + gamma * expected
    return scores


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.model = LeafModel().eval()
        self.device = torch.device("cpu")
        self.board = np.array([[2, 2, 8, 16], [16, 32, 64, 128], [256, 512, 1024, 2], [4, 8, 16, 32]])

    def test_spawn_probabilities_and_input_preservation(self):
        board = self.board.copy()
        board[0, :2] = 0
        before = board.copy()
        outcomes = list(spawn_outcomes(board))
        self.assertEqual(len(outcomes), 4)
        self.assertAlmostEqual(sum(p for p, _ in outcomes), 1)
        for p, state in outcomes:
            changed = state != board
            self.assertEqual(changed.sum(), 1)
            self.assertAlmostEqual(p, .45 if state[changed][0] == 2 else .05)
        np.testing.assert_array_equal(board, before)

    def test_scalar_reference_rewards_discounts_and_terminal_spawn(self):
        # A right merge creates one empty cell: spawning 2 ends the game,
        # whereas spawning 4 allows another merge. Terminal chance mass stays.
        right_afterstate = slide_board(self.board, 3)[0]
        outcomes = list(spawn_outcomes(right_afterstate))
        self.assertFalse(valid_actions(outcomes[0][1]).any())
        self.assertTrue(valid_actions(outcomes[1][1]).any())
        sparse = np.zeros((4, 4), dtype=np.int64)
        sparse[0, :2] = [2, 4]
        boards = np.stack((self.board, sparse))
        for mode in ("transition", "log"):
            for gamma in (0.0, .5, .99, 1.0):
                actual = score_expectimax_actions_batch(
                    self.model, boards, np.ones((2, 4), dtype=bool), self.device,
                    gamma, reward_mode=mode, inference_batch_size=12,
                )
                for index, board in enumerate(boards):
                    np.testing.assert_allclose(actual[index], scalar_reference(board, gamma, mode), rtol=1e-6, atol=1e-6)
        self.assertLessEqual(self.model.largest_batch, 12)

    def test_illegal_mask_active_and_terminal(self):
        terminal = self.board.copy()
        terminal[0, 0] = 4
        self.assertFalse(valid_actions(terminal).any())
        boards = np.stack((self.board, terminal))
        masks = np.zeros((2, 4), dtype=bool)
        masks[0, 2] = True
        scores = score_expectimax_actions_batch(self.model, boards, masks, self.device, .99)
        self.assertTrue(np.isfinite(scores[0, 2]))
        self.assertEqual(np.isfinite(scores).sum(), 1)
        actions = choose_expectimax_actions_batch(self.model, boards, masks, self.device, .99, active_mask=[True, False])
        np.testing.assert_array_equal(actions, [2, 0])
        with self.assertRaises(RuntimeError):
            choose_expectimax_actions_batch(self.model, boards, masks, self.device, .99)

    def test_chunking_does_not_change_scores(self):
        args = (self.model, self.board[None], np.ones((1, 4), dtype=bool), self.device, .99)
        np.testing.assert_array_equal(
            score_expectimax_actions_batch(*args, inference_batch_size=4),
            score_expectimax_actions_batch(*args, inference_batch_size=1024),
        )


if __name__ == "__main__":
    unittest.main()
