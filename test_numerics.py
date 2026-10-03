"""Prevent invalid TD values from corrupting model and replay state."""

import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from replay_buffer import ReplayBuffer
from train import AfterstateDQNAgent


class NumericalTests(unittest.TestCase):
    def test_nonfinite_priority_update_rejects_entire_batch(self):
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid=invalid):
                buffer = ReplayBuffer(capacity=4)
                state = np.zeros(16, dtype=np.float32)
                mask = np.ones(4, dtype=bool)
                for _ in range(4):
                    buffer.push(state, 0, 0.0, state, state, True, mask, mask)
                before = buffer.tree.tree.copy()
                leaves = buffer.tree.tree_size + np.arange(2)
                with self.assertRaises(FloatingPointError):
                    buffer.update_priorities(leaves, np.array([2.0, invalid]))
                np.testing.assert_array_equal(before, buffer.tree.tree)
                self.assertTrue(np.isfinite(buffer.sample(2)["is_weights"]).all())

    def test_nonfinite_target_fails_before_optimizer_or_tree_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = AfterstateDQNAgent(
                save_dir=directory, use_cuda=False, num_quantiles=3,
                replay_capacity=4, batch_size=2, learning_starts=2,
            )
            state = np.zeros(16, dtype=np.float32)
            mask = np.ones(4, dtype=bool)
            for _ in range(2):
                agent.buffer.push(state, 0, 0.0, state, state, True, mask, mask)
            tree = agent.buffer.tree.tree.copy()
            weights = {k: v.clone() for k, v in agent.online_net.state_dict().items()}
            with patch.object(agent, "_next_value_targets", return_value=torch.full((2, 3), float("nan"))):
                with self.assertRaises(FloatingPointError):
                    agent.update_model()
            np.testing.assert_array_equal(tree, agent.buffer.tree.tree)
            for key, value in agent.online_net.state_dict().items():
                torch.testing.assert_close(value, weights[key])


if __name__ == "__main__":
    unittest.main()
