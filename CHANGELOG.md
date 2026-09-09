# CHANGELOG

## v3.1（相对 v2.0）

### 网络与状态表示

- 将 V2 的 Transformer Encoder 改为保留 4×4 空间结构的 CNN 残差主干，输入使用 20 个棋盘特征通道。
- 移除 V2 的 SSL 辅助任务、动作条件的 Dueling Q head 与对应的 value/advantage 分支，改为共享的 afterstate 分布价值头，输出 51 个分位数。
- 将环境转移明确拆为 `slide → afterstate → spawn`，批量保存 afterstate；沿用 V2 的合法动作 mask，并将 mask 统一到确定性 afterstate/Numba 批处理路径。

### 学习目标

- 将 V2 的 3-step action-Q 目标改为 1-step afterstate Double DQN：动作价值由当前合并奖励与 afterstate value 组成，并用 online 网络选动作、target 网络估值。
- 保留 V2 的 `sqrt(merge_score) / 2` 合并奖励尺度；空格奖励改为在确定性 afterstate 上计算，并扣除即将生成的一个方块；训练路径不再依赖 V2 的非法动作负奖励。
- 保留 QR-DQN（51 个分位数）、Double DQN、PER 和 NoisyNet；探索改为合法动作 ε-greedy（`1.0 → 0.05`）叠加 NoisyNet，V2 则主要依赖 NoisyNet 后的纯贪心动作。
- Replay 从 V2 的 1,000,000 容量改为独立的 500,000 容量格式，记录 afterstate，移除 n-step 采样，并增加批量写入。
- 新增 D4 对称数据增强；量化回归损失改为按样本与分位数维度取均值，与 V2 的损失尺度不同。

### 训练配置改动

- 从 V2 的单环境训练改为 `num-envs=64` 的向量环境，使用 Numba 加速批量滑动、afterstate 候选生成，并默认启用 CUDA AMP；V2 不使用 AMP。
- 保持不变的基础配置：`gamma=0.99`、基础 `batch_size=128`、`num_quantiles=51`、`update_frequency=4`、Quantile Huber 的 `κ=1.0`，以及 Adam 优化器。
- 主要配置调整为：学习率 `1e-4 → 2e-4`；学习启动从 V2 的至少 128 条 replay 延后至 20,000 条；target 同步从每 1,000 个 optimizer step 改为每 2,500 个逻辑更新；梯度裁剪 `1.0 → 10.0`。
- V3.1 增加 `max_optimizer_steps_per_collect=4`：每次采集最多执行 4 个 optimizer step，并按需要合并为 `128 × logical_update_count` 的批次；V2 没有这一批量调度机制。
- V2 的 `ssl_weight=0.1` 随 SSL 分支移除；V3.1 增加显式 ε-greedy 配置（`epsilon=1.0 → 0.05`，衰减步数 `1,000,000`），与 NoisyNet 共同用于探索，V2 没有 ε-greedy。
- 调整 checkpoint/评估节奏：V2 每 100 局保存、绘图并进行 5 局中途评估且清理旧快照；V3.1 默认每 5,000 局（以及训练终点）保存 checkpoint，不在训练中途评估，评估单独运行。
- 增加显式 seed、独立随机源及 RNG 状态保存；checkpoint 同时保存 online/target 网络、optimizer、AMP scaler、训练历史和随机状态，格式与 V2 不兼容。

### 推理结果

- Expectimax 不包含在当前 1,000 局评估中，在v2版本进行过推理测试，已经证明有明显效果提升。
- episode 15000 权重在固定 `seed=100000` 的 1,000 局纯贪心测试中：平均分 `35,186.3`，`>=2048` 达成率 `81.3%`，`>=4096` 达成率 `20.5%`。
