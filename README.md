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
- `policy.py`：基于 afterstate value 的动作选择。
- `replay_buffer.py`：PER 回放池。
- `train.py`：批量环境训练与 checkpoint 恢复。
- `vector_env.py`：多个独立棋局的批量采样环境。
- `evaluate.py`：不使用搜索的纯贪心评估。
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
