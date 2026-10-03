"""Train the V3 afterstate CNN-DQN agent.

V3 intentionally keeps the training loop small and explicit:

    state -> deterministic afterstate -> random next state

The network estimates afterstate value.  A legal action is scored as its
transformed merge reward plus the discounted value of its afterstate.  The
training target uses the sampled random next state, while the next player
action is evaluated through its deterministic afterstate.  Expectimax is not
used here; it will be an independent inference-time module later.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import os
import random
import time
import warnings
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from fast_afterstate import (
    augment_observation_pairs,
    boards_to_observations,
    observations_to_boards,
    rewards_from_transitions,
    slide_all_actions_batch,
)
from env import Env2048
from model import AfterstateValueNet
from policy import choose_actions_batch
from replay_buffer import ReplayBuffer
from restart import RestartPool
from vector_env import BatchEnv2048


def file_sha256(path: str) -> str:
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def set_seed(seed: int) -> None:
    """Seed all RNGs used by V3 without forcing slow deterministic kernels."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Target/action candidate batches have dynamic lengths; benchmarking
        # every new candidate count can cost more than the convolution saves.
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("high")


class AfterstateDQNAgent:
    """DQN-style learner whose critic is evaluated on afterstates."""

    CONFIG_FIELDS = (
        "seed", "gamma", "batch_size", "num_quantiles", "learning_starts",
        "update_frequency", "target_update_frequency",
        "max_optimizer_steps_per_collect", "epsilon_start", "epsilon_final",
        "epsilon_decay_steps", "num_envs", "restart_probability",
    )

    def __init__(
        self,
        device: torch.device | None = None,
        seed: int = 0,
        gamma: float = 0.99,
        learning_rate: float = 2e-4,
        batch_size: int = 128,
        num_quantiles: int = 51,
        replay_capacity: int = 500_000,
        learning_starts: int = 20_000,
        update_frequency: int = 4,
        target_update_frequency: int = 2_500,
        max_optimizer_steps_per_collect: int = 4,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.05,
        epsilon_decay_steps: int = 1_000_000,
        save_dir: str | None = None,
        use_cuda: bool = True,
        use_amp: bool = True,
        num_envs: int = 64,
        restart_probability: float = 0.0,
    ):
        set_seed(seed)
        if device is None:
            device = torch.device(
                "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
            )

        self.device = device
        self.amp_enabled = bool(use_amp and self.device.type == "cuda")
        self.seed = int(seed)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.num_quantiles = int(num_quantiles)
        self.learning_starts = int(learning_starts)
        self.update_frequency = int(update_frequency)
        self.target_update_frequency = int(target_update_frequency)
        self.max_optimizer_steps_per_collect = int(
            max_optimizer_steps_per_collect
        )
        self.epsilon_start = float(epsilon_start)
        self.epsilon_final = float(epsilon_final)
        self.epsilon_decay_steps = int(epsilon_decay_steps)
        self.num_envs = int(num_envs)
        self.restart_probability = float(restart_probability)
        if not 0.0 <= self.restart_probability <= 1.0:
            raise ValueError("restart_probability must be between 0 and 1")
        self.restart_pool = RestartPool()
        if self.num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        if self.update_frequency < 1:
            raise ValueError("update_frequency must be at least 1")
        if self.num_quantiles < 1:
            raise ValueError("num_quantiles must be at least 1")
        if self.max_optimizer_steps_per_collect < 1:
            raise ValueError("max_optimizer_steps_per_collect must be at least 1")
        self.update_credit = 0

        self.env = Env2048(seed=seed)
        self.buffer = ReplayBuffer(capacity=replay_capacity)
        self.online_net = AfterstateValueNet(
            num_quantiles=self.num_quantiles
        ).to(self.device)
        self.target_net = AfterstateValueNet(
            num_quantiles=self.num_quantiles
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.tau = (
            torch.arange(
                0.5,
                self.num_quantiles,
                1.0,
                device=self.device,
            )
            / self.num_quantiles
        )

        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=learning_rate,
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.amp_enabled,
        )
        default_save_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "models", "v3_1"
        )
        self.save_dir = save_dir or default_save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.buffer_path = os.path.join(self.save_dir, "replay_buffer.npz")

        self.start_episode = 0
        self.total_env_steps = 0
        self.update_steps = 0
        self.optimizer_steps = 0
        self.optimizer_steps_before_resume_unknown = False
        self.training_seconds = 0.0
        self.restart_history: list[dict[str, Any]] = []
        self.resume_source: str | None = None
        self.history: dict[str, list[Any]] = {
            "episodes": [],
            "scores": [],
            "max_tiles": [],
            "rewards": [],
            "episode_lengths": [],
            "losses": [],
            "mean_values": [],
        }

        print(f"训练设备: {self.device} | AMP: {self.amp_enabled}")
        print(
            f"V3 参数量: {sum(p.numel() for p in self.online_net.parameters()):,}"
        )

    def epsilon(self) -> float:
        progress = min(1.0, self.total_env_steps / self.epsilon_decay_steps)
        return self.epsilon_start + progress * (
            self.epsilon_final - self.epsilon_start
        )

    def _tensor(self, array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(array, dtype=dtype, device=self.device)

    def _autocast_context(self):
        if self.amp_enabled:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    @torch.no_grad()
    def _next_value_targets(
        self,
        next_states: np.ndarray,
        _next_masks: np.ndarray,
        dones: np.ndarray,
    ) -> torch.Tensor:
        """Compute a sampled-chance, vectorized quantile Double-DQN target."""

        batch_count = len(next_states)
        targets = torch.zeros(
            (batch_count, self.num_quantiles),
            dtype=torch.float32,
            device=self.device,
        )
        active = np.asarray(dones, dtype=np.float32) < 0.5
        if not np.any(active):
            return targets

        boards = observations_to_boards(next_states)
        afterstates, merge_scores, changed = slide_all_actions_batch(boards)
        # ``changed`` is recomputed from the transformed board and is therefore
        # also the correct mask after symmetry augmentation.  The stored mask
        # remains part of replay for diagnostics and compatibility.
        legal = changed & active[:, None]
        locations = np.argwhere(legal)
        if len(locations) == 0:
            return targets

        board_indices = locations[:, 0]
        action_indices = locations[:, 1]
        candidate_observations = boards_to_observations(
            afterstates[board_indices, action_indices]
        )
        candidate_tensor = self._tensor(candidate_observations, torch.float32)
        with self._autocast_context():
            online_values = self.online_net(candidate_tensor)
            target_values = self.target_net(candidate_tensor)
        rewards = self._tensor(
            rewards_from_transitions(
                merge_scores[board_indices, action_indices],
                afterstates[board_indices, action_indices],
            ),
            torch.float32,
        )
        candidate_online_scores = (
            rewards.unsqueeze(1) + self.gamma * online_values.float()
        )
        candidate_target_quantiles = (
            rewards.unsqueeze(1) + self.gamma * target_values.float()
        )

        online_scores = torch.full(
            (batch_count, 4),
            -torch.inf,
            dtype=torch.float32,
            device=self.device,
        )
        target_quantiles = torch.full(
            (batch_count, 4, self.num_quantiles),
            -torch.inf,
            dtype=torch.float32,
            device=self.device,
        )
        board_index_tensor = self._tensor(board_indices, torch.long)
        action_index_tensor = self._tensor(action_indices, torch.long)
        online_scores[board_index_tensor, action_index_tensor] = (
            candidate_online_scores.mean(dim=1)
        )
        target_quantiles[board_index_tensor, action_index_tensor] = (
            candidate_target_quantiles
        )

        active_tensor = self._tensor(np.flatnonzero(active), torch.long)
        best_actions = torch.argmax(online_scores, dim=1)
        targets[active_tensor] = target_quantiles[
            active_tensor, best_actions[active_tensor]
        ]
        return targets

    def update_model(
        self,
        batch_size: int | None = None,
        logical_update_count: int = 1,
    ) -> dict[str, float] | None:
        """Run one optimizer step, optionally representing several small updates."""

        sample_size = self.batch_size if batch_size is None else int(batch_size)
        if sample_size < 1:
            raise ValueError("batch_size must be at least 1")
        if len(self.buffer) < max(sample_size, self.learning_starts):
            return None

        batch = self.buffer.sample(sample_size)
        afterstates, next_states = augment_observation_pairs(
            batch["afterstate"],
            batch["next_state"],
        )
        afterstates = self._tensor(afterstates, torch.float32)
        with self._autocast_context():
            current_quantiles = self.online_net(afterstates).float()
            targets = self._next_value_targets(
                next_states,
                batch["next_mask"],
                batch["done"],
            )

            pairwise_td = targets.unsqueeze(1) - current_quantiles.unsqueeze(2)
            abs_td = pairwise_td.abs()
            huber = torch.where(
                abs_td <= 1.0,
                0.5 * pairwise_td.square(),
                abs_td - 0.5,
            )
            tau_view = self.tau.view(1, self.num_quantiles, 1)
            quantile_weight = (
                tau_view - (pairwise_td.detach() < 0).float()
            ).abs()
            per_sample_loss = (quantile_weight * huber).mean(dim=(1, 2))
            is_weights = self._tensor(batch["is_weights"], torch.float32)
            loss = (is_weights * per_sample_loss).mean()
            td_errors = (
                targets.mean(dim=1) - current_quantiles.mean(dim=1)
            )

        numerical_values = {
            "current_quantiles": current_quantiles,
            "targets": targets, "loss": loss, "td_errors": td_errors,
        }
        if not all(torch.isfinite(value).all().item() for value in numerical_values.values()):
            self._save_numerical_failure(batch, numerical_values)
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        if not self.amp_enabled and not torch.isfinite(grad_norm).item():
            self._save_numerical_failure(batch, {**numerical_values, "grad_norm": grad_norm})
        self.scaler.step(self.optimizer)
        previous_scale = self.scaler.get_scale()
        self.scaler.update()
        if self.scaler.get_scale() >= previous_scale:
            self.optimizer_steps += 1

        self.buffer.update_priorities(
            batch["tree_indices"],
            td_errors.detach().cpu().numpy(),
        )

        previous_update_steps = self.update_steps
        self.update_steps += int(logical_update_count)
        if (
            previous_update_steps // self.target_update_frequency
            != self.update_steps // self.target_update_frequency
        ):
            self.target_net.load_state_dict(self.online_net.state_dict())

        return {
            "loss": float(loss.item()),
            "mean_value": float(current_quantiles.detach().mean().item()),
            "mean_abs_td": float(td_errors.detach().abs().mean().item()),
        }

    def _save_numerical_failure(self, batch: dict, values: dict) -> None:
        path = os.path.join(self.save_dir, f"numerical_failure_{time.time_ns()}.pth")
        payload = self._checkpoint_payload(self.start_episode)
        payload["diagnostic_batch"] = batch
        payload["diagnostic_values"] = {
            key: value.detach().cpu() for key, value in values.items()
        }
        payload["diagnostic_note"] = "Failure snapshot, not a resumable checkpoint; episode is last loaded boundary."
        with open(path, "xb") as output:
            torch.save(payload, output)
        raise FloatingPointError(f"nonfinite training values; diagnostic saved to {path}")

    def _checkpoint_payload(self, episode: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": "v3.1-afterstate-quantile-cnn-dqn",
            "episode": int(episode),
            "total_env_steps": int(self.total_env_steps),
            "update_steps": int(self.update_steps),
            "replay_capacity": int(self.buffer.capacity),
            "num_quantiles": self.num_quantiles,
            "max_optimizer_steps_per_collect": self.max_optimizer_steps_per_collect,
            "model_state_dict": self.online_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "history": self.history,
            "seed": self.seed,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "env_rng_state": self.env.game.rng.getstate(),
            "num_envs": self.num_envs,
            "update_credit": int(self.update_credit),
            "training_config": {
                **{key: getattr(self, key) for key in self.CONFIG_FIELDS},
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "replay_capacity": self.buffer.capacity,
                "use_amp": self.amp_enabled,
            },
            "optimizer_steps": self.optimizer_steps,
            "optimizer_steps_before_resume_unknown": self.optimizer_steps_before_resume_unknown,
            "training_seconds": self.training_seconds,
            "restart_pool": self.restart_pool.state_dict(),
            "restart_history": self.restart_history,
            "resume_source": self.resume_source,
        }
        if torch.cuda.is_available():
            payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        return payload

    def save_checkpoint(self, episode: int) -> str:
        checkpoint_path = os.path.join(self.save_dir, f"dqn_ep{episode}.pth")
        replay_path = os.path.join(self.save_dir, f"replay_ep{episode}.npz")
        for path in (checkpoint_path, replay_path):
            if os.path.exists(path):
                raise FileExistsError(f"refusing to overwrite {path}")
        self.buffer.save(replay_path)
        payload = self._checkpoint_payload(episode)
        payload["replay_file"] = os.path.basename(replay_path)
        payload["replay_sha256"] = file_sha256(replay_path)
        with open(checkpoint_path, "xb") as checkpoint_file:
            torch.save(payload, checkpoint_file)
        print(f"[保存] {checkpoint_path} | replay={len(self.buffer)}")
        return checkpoint_path

    def load_checkpoint(
        self, checkpoint_path: str, *, replay_path: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        if checkpoint.get("version") != "v3.1-afterstate-quantile-cnn-dqn":
            raise ValueError("checkpoint is not a compatible V3.1 checkpoint")

        config = checkpoint.get("training_config")
        if config is None:
            warnings.warn("Legacy checkpoint lacks full configuration; using current settings for missing fields.")
            config = {key: checkpoint[key] for key in self.CONFIG_FIELDS if key in checkpoint}
        config = {**config, **(overrides or {})}
        if int(config.get("num_quantiles", self.num_quantiles)) != self.num_quantiles:
            raise ValueError("construct the agent with the checkpoint num_quantiles")
        bound_replay = checkpoint.get("replay_file")
        if bound_replay:
            if replay_path is not None:
                raise ValueError("new checkpoints already bind their replay; do not override it")
            replay_path = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), bound_replay)
            if file_sha256(replay_path) != checkpoint["replay_sha256"]:
                raise ValueError("checkpoint replay checksum mismatch")
        elif replay_path is None:
            raise ValueError("legacy checkpoint requires explicit replay_path / --resume-replay; its original replay cannot be inferred")
        else:
            warnings.warn("Legacy replay selected explicitly; checkpoint association cannot be verified.")
        saved_capacity = int(checkpoint.get("replay_capacity", self.buffer.capacity))
        restored_buffer = ReplayBuffer(capacity=saved_capacity)
        restored_buffer.load(replay_path)
        for key in self.CONFIG_FIELDS:
            if key in config:
                setattr(self, key, config[key])

        self.online_net.load_state_dict(checkpoint["model_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if overrides and "learning_rate" in overrides:
            for group in self.optimizer.param_groups:
                group["lr"] = overrides["learning_rate"]
        if checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.start_episode = int(checkpoint["episode"])
        self.total_env_steps = int(checkpoint["total_env_steps"])
        self.update_steps = int(checkpoint["update_steps"])
        self.update_credit = int(checkpoint.get("update_credit", 0))
        self.history = checkpoint.get("history", self.history)
        self.optimizer_steps = int(checkpoint.get("optimizer_steps", 0))
        self.optimizer_steps_before_resume_unknown = bool(checkpoint.get(
            "optimizer_steps_before_resume_unknown", "optimizer_steps" not in checkpoint,
        ))
        self.training_seconds = float(checkpoint.get("training_seconds", 0.0))
        self.restart_history = checkpoint.get("restart_history", [])
        if "restart_pool" in checkpoint:
            self.restart_pool.load_state_dict(checkpoint["restart_pool"])
        self.resume_source = os.path.abspath(checkpoint_path)
        self.buffer = restored_buffer

        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng_state_all"]]
            )
        if "env_rng_state" in checkpoint:
            self.env.game.rng.setstate(checkpoint["env_rng_state"])

        print(f"[恢复] replay={len(self.buffer)}")
        print(f"[恢复] episode={self.start_episode} env_steps={self.total_env_steps}")

    def train(
        self,
        target_episodes: int,
        save_every: int = 5_000,
        target_env_steps: int | None = None,
    ) -> None:
        """Train in exact-size chunks of independent vectorized episodes."""

        target_episodes = int(target_episodes)
        if target_episodes <= self.start_episode:
            print(
                f"[跳过] checkpoint 已到 episode={self.start_episode}, "
                f"目标为 {target_episodes}"
            )
            return

        start_time = time.time()
        starting_env_steps = self.total_env_steps
        completed = self.start_episode
        while completed < target_episodes:
            if target_env_steps is not None and self.total_env_steps >= target_env_steps:
                break
            chunk_start = time.perf_counter()
            next_boundary = target_episodes
            if save_every > 0:
                next_boundary = min(
                    next_boundary,
                    ((completed // save_every) + 1) * save_every,
                )
            chunk_size = min(self.num_envs, next_boundary - completed)
            batch_env = BatchEnv2048(
                num_envs=chunk_size,
                seed=self.seed + completed,
            )
            initial_boards = [None] * chunk_size
            if self.restart_probability > 0.0 and len(self.restart_pool):
                for index in range(chunk_size):
                    if np.random.random() < self.restart_probability:
                        initial_boards[index] = self.restart_pool.sample()
            restarted = [board is not None for board in initial_boards]
            states, masks = batch_env.reset(initial_boards=initial_boards)
            trajectories: list[list[np.ndarray]] = [[] for _ in range(chunk_size)]
            active = np.ones(chunk_size, dtype=bool)
            episode_rewards = np.zeros(chunk_size, dtype=np.float32)
            episode_steps = np.zeros(chunk_size, dtype=np.int64)
            episode_losses: list[list[float]] = [
                [] for _ in range(chunk_size)
            ]
            episode_values: list[list[float]] = [
                [] for _ in range(chunk_size)
            ]

            while np.any(active):
                active_indices = np.flatnonzero(active)
                boards = batch_env.boards()
                if self.restart_probability > 0.0:
                    for index in active_indices:
                        trajectories[index].append(boards[index].copy())
                actions = choose_actions_batch(
                    self.online_net,
                    boards,
                    masks,
                    self.device,
                    self.gamma,
                    epsilon=self.epsilon(),
                    active_mask=active,
                    amp_enabled=self.amp_enabled,
                )
                (
                    next_states,
                    rewards,
                    dones,
                    afterstate_observations,
                    next_masks,
                    infos,
                ) = batch_env.step(actions, active_mask=active)

                selected_afterstates = afterstate_observations[
                    active_indices,
                    actions[active_indices],
                ]
                self.buffer.push_batch(
                    state=states[active_indices],
                    action=actions[active_indices],
                    reward=rewards[active_indices],
                    afterstate=selected_afterstates,
                    next_state=next_states[active_indices],
                    done=dones[active_indices],
                    mask=masks[active_indices],
                    next_mask=next_masks[active_indices],
                )

                active_count = len(active_indices)
                previous_steps = self.total_env_steps
                self.total_env_steps += active_count
                previous_eligible = max(
                    0,
                    previous_steps - self.learning_starts,
                )
                current_eligible = max(
                    0,
                    self.total_env_steps - self.learning_starts,
                )
                self.update_credit += current_eligible - previous_eligible

                step_losses: list[float] = []
                step_values: list[float] = []
                updates_due = self.update_credit // self.update_frequency
                if updates_due > 0:
                    optimizer_steps = min(
                        updates_due,
                        self.max_optimizer_steps_per_collect,
                    )
                    updates_per_step = updates_due // optimizer_steps
                    extra_updates = updates_due % optimizer_steps
                    for optimizer_index in range(optimizer_steps):
                        logical_updates = updates_per_step + int(
                            optimizer_index < extra_updates
                        )
                        metrics = self.update_model(
                            batch_size=self.batch_size * logical_updates,
                            logical_update_count=logical_updates,
                        )
                        if metrics is not None:
                            step_losses.append(metrics["loss"])
                            step_values.append(metrics["mean_value"])
                    self.update_credit -= updates_due * self.update_frequency

                states = next_states
                masks = next_masks
                episode_rewards[active_indices] += rewards[active_indices]
                episode_steps[active_indices] += 1
                for index in active_indices:
                    if step_losses:
                        episode_losses[index].extend(step_losses)
                    if step_values:
                        episode_values[index].extend(step_values)
                    if not dones[index]:
                        continue

                    completed += 1
                    if self.restart_probability > 0.0:
                        self.restart_pool.add_trajectory(trajectories[index])
                    if restarted[index]:
                        self.restart_history.append({
                            "episode": completed,
                            "remaining_score": int(infos[index]["score"]),
                            "max_tile": int(infos[index]["max_tile"]),
                            "length": int(episode_steps[index]),
                            "seed": batch_env.seed + int(index),
                            "start_board": initial_boards[index].tolist(),
                        })
                        active[index] = False
                        continue
                    self.history["episodes"].append(completed)
                    self.history["scores"].append(infos[index]["score"])
                    self.history["max_tiles"].append(infos[index]["max_tile"])
                    self.history["rewards"].append(
                        float(episode_rewards[index])
                    )
                    self.history["episode_lengths"].append(
                        int(episode_steps[index])
                    )
                    self.history["losses"].append(
                        float(np.mean(episode_losses[index]))
                        if episode_losses[index]
                        else 0.0
                    )
                    self.history["mean_values"].append(
                        float(np.mean(episode_values[index]))
                        if episode_values[index]
                        else 0.0
                    )
                    active[index] = False

                    if completed % 10 == 0 or completed == self.start_episode + 1:
                        elapsed = max(time.time() - start_time, 1e-6)
                        print(
                            f"[{completed:6d}/{target_episodes}] "
                            f"score={infos[index]['score']:6d} "
                            f"tile={infos[index]['max_tile']:5d} "
                            f"steps={episode_steps[index]:4d} "
                            f"eps={self.epsilon():.3f} "
                            f"loss={self.history['losses'][-1]:.4f} "
                            f"speed={(self.total_env_steps - starting_env_steps) / elapsed:.1f} "
                            "step/s"
                        )

            self.training_seconds += time.perf_counter() - chunk_start
            budget_reached = target_env_steps is not None and self.total_env_steps >= target_env_steps
            should_save = (
                save_every > 0
                and (completed % save_every == 0 or completed == target_episodes or budget_reached)
            )
            if should_save:
                self.save_checkpoint(completed)
        self.start_episode = completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V3 Afterstate CNN-DQN 训练")
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume-replay", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--target-env-steps", type=int, default=None,
                        help="Total environment-step threshold; stop after the current chunk finishes")
    parser.add_argument("--save-every", type=int, default=5_000)
    for name in ("seed", "batch-size", "num-quantiles", "replay-capacity",
                 "learning-starts", "update-frequency", "target-update-frequency",
                 "max-optimizer-steps-per-collect", "epsilon-decay-steps", "num-envs"):
        parser.add_argument("--" + name, type=int, default=None)
    for name in ("learning-rate", "gamma", "epsilon-final", "restart-probability"):
        parser.add_argument("--" + name, type=float, default=None)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.resume and args.save_dir is None:
        parser.error("--resume requires --save-dir to preserve existing training artifacts")
    if args.resume_replay and not args.resume:
        parser.error("--resume-replay requires --resume")
    return args


def main() -> None:
    args = parse_args()
    keys = AfterstateDQNAgent.CONFIG_FIELDS + ("learning_rate", "replay_capacity")
    overrides = {key: getattr(args, key) for key in keys
                 if getattr(args, key, None) is not None}
    config = {}
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        config = checkpoint.get("training_config", {
            key: checkpoint[key] for key in keys if key in checkpoint
        })
        if "replay_capacity" in overrides and overrides["replay_capacity"] != checkpoint["replay_capacity"]:
            raise ValueError("resuming cannot change replay capacity")
        if "num_quantiles" in overrides and overrides["num_quantiles"] != checkpoint["num_quantiles"]:
            raise ValueError("resuming cannot change num_quantiles")
        del checkpoint
    config.update(overrides)
    config["use_amp"] = False if args.no_amp else config.get("use_amp", True)
    agent = AfterstateDQNAgent(
        **config, save_dir=args.save_dir,
        use_cuda=not args.no_cuda,
    )
    if args.resume:
        agent.load_checkpoint(args.resume, replay_path=args.resume_replay, overrides=overrides)
    agent.train(
        target_episodes=args.episodes, save_every=args.save_every,
        target_env_steps=args.target_env_steps,
    )


if __name__ == "__main__":
    main()
