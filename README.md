# 2048 V3.1：Afterstate Quantile CNN-DQN

这是一个独立于原 V2 的训练目录。V3 不修改 `D:\Desktop\pp\2048`，也不读取 V2 的 checkpoint。

## 当前目标

V3 只负责训练一个更贴近 2048 转移结构的 DQN 网络：

```text
state → 确定性滑动/合并 → afterstate → 随机生成 2/4 → next_state
```

网络估计 afterstate 的回报分布，动作分数使用分位数均值，动作分数为：

```text
Q(state, action) = 当前合并奖励 + γ × V(afterstate)
```

训练侧保留 DQN 的 online/target 网络、Double-DQN 动作选择、QR 分布值、PER、Quantile Huber loss 和 ε-greedy；同时只在量化价值头加入轻量 Factorized NoisyNet 探索。V3.1 不使用 Transformer、SSL 或 MCTS。

Expectimax 不在训练循环中使用，后续会作为单独的推理策略加入。

## 文件

- `afterstate.py`：确定性滑动、随机生成枚举、状态编码和奖励缩放。
- `fast_afterstate.py`：从副本提取的可选 Numba 单盘/批量滑动加速后端。
- `game_2048.py`：可设 seed 的游戏引擎。
- `env.py`：返回 afterstate 信息的环境封装。
- `model.py`：带残差块、坐标通道和 51 分位数输出的 CNN afterstate value network。
- `model_v30.py`：只用于读取旧版 V3.0 scalar checkpoint 的兼容网络，不参与 V3.1 训练。
- `policy.py`：基于 afterstate value 的动作选择。
- `replay_buffer.py`：PER 回放池。
- `train.py`：批量环境训练与 checkpoint 恢复。
- `vector_env.py`：多个独立棋局的批量采样环境。
- `evaluate.py`：不使用搜索的纯贪心评估，自动匹配 V3.0/V3.1 checkpoint 的网络和奖励尺度。
- `test_v3.py`：转移、编码、网络和 replay smoke tests。

如果安装了 Numba，V3 会自动使用 JIT 加速的确定性滑动和批量候选生成；没有 Numba 时会回退到 `afterstate.py` 的参考实现。

## 运行

在 PowerShell 中：

```powershell
cd D:\Desktop\pp\2048-v3
python -m unittest test_v3.py
python train.py --episodes 10000 --seed 0 --num-envs 64
python evaluate.py --episodes 1000 --seed 100000
```

对两个版本进行可复现的纯贪心对比：

```powershell
python evaluate.py --ckpt models/v3/dqn_ep10000.pth --episodes 1000 --seed 100000 --num-envs 64 --no-cuda --output models/v3/eval_ep10000_greedy_1000.json
python evaluate.py --ckpt models/v3_1/dqn_ep10000.pth --episodes 1000 --seed 100000 --num-envs 64 --no-cuda --output models/v3_1/eval_ep10000_greedy_1000.json
python plot_training.py --checkpoint models/v3/dqn_ep10000.pth --output training_curve_v30.png
python plot_training.py --checkpoint models/v3_1/dqn_ep10000.pth --output training_curve_v31.png
```

评估脚本当前不使用 Expectimax；它只测量网络本身的贪心策略。旧版 V3.0 的训练 checkpoint 与 V3.1 网络不兼容，但 `evaluate.py` 可以通过 `model_v30.py` 读取旧版权重，并自动使用旧版 log reward，因此可以直接生成同口径对比结果。

短 smoke run 可以使用：

```powershell
python train.py --episodes 3 --learning-starts 32 --batch-size 8 --save-every 0
```

恢复 V3.1 训练：

```powershell
python train.py --episodes 50000 --resume models/v3_1/dqn_ep10000.pth
```

默认 V3.1 checkpoint 位于 `models/v3_1/`；`models/v3/` 保留上一版 scalar afterstate baseline。目录根部可能存在历史 V2 文件，但 V3.1 训练和评估不会扫描或加载它们。V3.1 checkpoint 与 V2 和 V3.0 checkpoint 都不兼容。评估结果需要分别记录纯贪心和后续加入搜索后的结果。

训练默认同时采样 64 个独立棋局，并把 exploitative 动作候选合并为一次 CNN 前向；`--num-envs 1` 可退回单棋局批处理模式。每次采样批次最多执行 4 个真实 optimizer step，避免把 64 个环境步对应的多次更新压成一次过大的梯度。训练过程中不运行评估，`--save-every 5000` 时会在第 5000 和第 10000 局保存完整 checkpoint。

## 最近验证

### 2026-09-02：V3.0/V3.1 曲线与 1000 局对比评估

- V3.1 正式训练保持 `num_envs=64`，没有切换到 256，也没有引入额外训练变量；从 5000 局 checkpoint 恢复后正常完成到 10000 局。
- V3.1 `models/v3_1/dqn_ep5000.pth`：`episode=5000`、`total_env_steps=1,732,218`、`update_steps=428,054`、历史长度 5000、51 个分位数。
- V3.1 `models/v3_1/dqn_ep10000.pth`：`episode=10000`、`total_env_steps=7,117,666`、`update_steps=1,774,416`、历史长度 10000、51 个分位数。
- V3.1 网络参数量为 769,734。两个 checkpoint 的 `.pth` 都保存了完整网络、target 网络、优化器、AMP scaler、训练进度和 RNG 状态；训练结束后的 `models/v3_1/replay_buffer.npz` 是最终回放旁车文件，不是单独冻结在 5000 局的 replay 快照。
- 曲线文件：`training_curve_v30.png` 和 `training_curve_v31.png`。V3.0 在约 5000 局后趋于平台；V3.1 在 5000 局之后仍继续上升，训练历史中已出现 4096 方块。
- 下表为相同 `seed=100000`、相同 `num_envs=64`、各 1000 局、纯贪心且不含 Expectimax 的结果。V3.0 使用其原有 log reward 重新计算，V3.1 使用 transition reward；这样避免把不同训练奖励尺度混在一起。

| 指标 | V3.0 | V3.1 | 变化 |
| --- | ---: | ---: | ---: |
| 平均分 | 7,445.3 | 27,634.4 | 3.71x |
| 中位分 | 7,056.0 | 30,302.0 | 4.29x |
| 最高分 | 16,360 | 60,816 | — |
| `>=512` | 80.8% | 100.0% | +19.2 个百分点 |
| `>=1024` | 20.5% | 95.9% | +75.4 个百分点 |
| `>=2048` | 0.0% | 72.2% | +72.2 个百分点 |
| `>=4096` | 0.0% | 3.4% | +3.4 个百分点 |
| 实际最高方块 | 1024 | 4096 | — |
| 平均步数 | 491.8 | 1,442.1 | 2.93x |

- 结果文件：`models/v3/eval_ep10000_greedy_1000.json` 和 `models/v3_1/eval_ep10000_greedy_1000.json`。
- 本轮没有把 Expectimax 加入评估；后续若加入 Expectimax-2，应在相同 seed 下另存一组搜索评估结果，与这里的纯网络基线分开比较。
