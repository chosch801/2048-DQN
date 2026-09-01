"""Focused smoke tests for the V3 transition and learning interfaces."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from afterstate import (
    action_candidates,
    augment_observation_pairs,
    board_to_observation,
    observation_to_board,
    reward_from_transition,
    slide_board,
    spawn_outcomes,
    valid_actions,
)
from env import Env2048
from fast_afterstate import NUMBA_AVAILABLE, slide_board as fast_slide_board
from model import AfterstateValueNet
from policy import choose_actions_batch
from replay_buffer import ReplayBuffer
from vector_env import BatchEnv2048


class AfterstateTests(unittest.TestCase):
    def test_left_merge_is_deterministic_and_reports_score(self):
        board = np.array(
            [
                [2, 2, 4, 4],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int64,
        )
        afterstate, merge_score, changed = slide_board(board, 2)
        np.testing.assert_array_equal(
            afterstate[0], np.array([4, 8, 0, 0], dtype=np.int64)
        )
        self.assertEqual(merge_score, 12)
        self.assertTrue(changed)

    def test_spawn_probabilities_sum_to_one(self):
        board = np.zeros((4, 4), dtype=np.int64)
        outcomes = list(spawn_outcomes(board))
        self.assertEqual(len(outcomes), 32)
        self.assertAlmostEqual(sum(probability for probability, _ in outcomes), 1.0)
        self.assertAlmostEqual(sum(probability for probability, _ in outcomes if np.max(_) == 2), 0.9)

    def test_observation_round_trip(self):
        board = np.array(
            [[0, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [4096, 0, 0, 32768]],
            dtype=np.int64,
        )
        reconstructed = observation_to_board(board_to_observation(board))
        np.testing.assert_array_equal(reconstructed, board)

    def test_symmetry_augmentation_keeps_pair_alignment(self):
        first = np.arange(16, dtype=np.float32)[None, :]
        second = first + 100.0
        augmented_first, augmented_second = augment_observation_pairs(
            first,
            second,
        )
        np.testing.assert_array_equal(augmented_second - augmented_first, 100.0)

    def test_transition_reward_uses_merge_and_empty_bonus(self):
        afterstate = np.zeros((4, 4), dtype=np.int64)
        afterstate[0, 0] = 2
        reward = reward_from_transition(16, afterstate)
        self.assertAlmostEqual(reward, 2.14)

    def test_action_candidates_match_valid_mask(self):
        board = np.array(
            [[2, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [4096, 8192, 16384, 32768]],
            dtype=np.int64,
        )
        mask = valid_actions(board)
        candidates = action_candidates(board, mask)
        self.assertEqual({candidate[0] for candidate in candidates}, set(np.flatnonzero(mask)))

    @unittest.skipUnless(NUMBA_AVAILABLE, "Numba is not installed")
    def test_numba_slide_matches_reference(self):
        rng = np.random.default_rng(123)
        levels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        for _ in range(25):
            board = np.power(2, rng.choice(levels, size=(4, 4))).astype(np.int64)
            board[rng.random((4, 4)) < 0.35] = 0
            for action in range(4):
                expected = slide_board(board, action)
                actual = fast_slide_board(board, action)
                np.testing.assert_array_equal(actual[0], expected[0])
                self.assertEqual(actual[1:], expected[1:])


class InterfaceTests(unittest.TestCase):
    def test_environment_exposes_afterstate(self):
        env = Env2048(seed=7)
        state, info = env.reset(seed=7)
        self.assertEqual(state.shape, (16,))
        action = int(np.flatnonzero(info["valid_actions"])[0])
        next_state, reward, done, truncated, next_info = env.step(action)
        self.assertEqual(next_state.shape, (16,))
        self.assertIsNotNone(next_info["afterstate"])
        self.assertIsInstance(reward, float)
        self.assertFalse(truncated)
        self.assertIsInstance(done, bool)

    def test_network_and_replay_shapes(self):
        network = AfterstateValueNet()
        observations = torch.zeros((4, 16), dtype=torch.float32)
        values = network(observations)
        self.assertEqual(tuple(values.shape), (4, 51))

        buffer = ReplayBuffer(capacity=8)
        mask = np.array([True, True, False, False])
        observation = np.zeros(16, dtype=np.float32)
        for index in range(4):
            buffer.push(observation, 0, 0.0, observation, observation, False, mask, mask)
        batch = buffer.sample(4)
        self.assertEqual(batch["state"].shape, (4, 16))
        buffer.update_priorities(
            batch["tree_indices"],
            np.linspace(0.1, 1.0, num=4, dtype=np.float32),
        )
        self.assertTrue(np.isfinite(buffer.tree.total()))
        self.assertEqual(buffer.sample(4)["state"].shape, (4, 16))
        self.assertEqual(batch["afterstate"].shape, (4, 16))
        self.assertEqual(batch["is_weights"].shape, (4,))

    def test_batch_environment_shapes_and_legal_step(self):
        env = BatchEnv2048(num_envs=4, seed=11)
        states, masks = env.reset()
        self.assertEqual(states.shape, (4, 16))
        self.assertEqual(masks.shape, (4, 4))
        actions = np.array(
            [np.flatnonzero(mask)[0] for mask in masks],
            dtype=np.int64,
        )
        (
            next_states,
            rewards,
            dones,
            afterstates,
            next_masks,
            infos,
        ) = env.step(actions)
        self.assertEqual(next_states.shape, (4, 16))
        self.assertEqual(rewards.shape, (4,))
        self.assertEqual(dones.shape, (4,))
        self.assertEqual(afterstates.shape, (4, 4, 16))
        self.assertEqual(next_masks.shape, (4, 4))
        self.assertEqual(len(infos), 4)

    def test_replay_push_batch(self):
        buffer = ReplayBuffer(capacity=8)
        states = np.zeros((4, 16), dtype=np.float32)
        actions = np.arange(4, dtype=np.int64)
        rewards = np.zeros(4, dtype=np.float32)
        masks = np.ones((4, 4), dtype=bool)
        buffer.push_batch(
            states,
            actions,
            rewards,
            states,
            states,
            np.zeros(4, dtype=bool),
            masks,
            masks,
        )
        self.assertEqual(len(buffer), 4)
        batch = buffer.sample(4)
        self.assertEqual(batch["state"].shape, (4, 16))

    def test_batch_policy_returns_only_legal_actions(self):
        env = BatchEnv2048(num_envs=8, seed=19)
        _, masks = env.reset()
        model = AfterstateValueNet()
        actions = choose_actions_batch(
            model,
            env.boards(),
            masks,
            torch.device("cpu"),
            gamma=0.99,
            epsilon=0.0,
        )
        self.assertTrue(
            np.all(masks[np.arange(len(actions)), actions])
        )


if __name__ == "__main__":
    unittest.main()
