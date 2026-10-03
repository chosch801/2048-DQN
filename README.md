# 2048-CNN-DQN V3.1

基于 afterstate 的 CNN-DQN 过渡版本。训练侧使用 Double DQN、QR-DQN、PER、NoisyNet 和合法动作 ε-greedy；不再使用 V2 的 Transformer、SSL 和 Dueling Q head。

## 结果

`num-envs=64`，模型训练至 15,000 局，仍有持续上升趋势，不再继续训练了。第二张图片是 episode 15000 权重在固定 `seed=100000` 下进行的 1,000 局纯贪心测试结果。

![episode 15000 training curve](training_curve_v31_ep15000.png)

![episode 15000 1000-game evaluation](evaluation_summary_ep15000.png)

| 指标 | 结果 |
| --- | ---: |
| 平均分 | 35,186.3 |
| 中位数分数 | 34,710 |
| 最高分 | 78,940 |
| `>=512` | 100.0% |
| `>=1024` | 98.0% |
| `>=2048` | 81.3% |
| `>=4096` | 20.5% |
| 平均游戏长度 | 1,764.1 步 |
| 最长游戏 | 3,649 步 |

## 参考

- [Optimistic Temporal Difference Learning for 2048（arXiv:2111.11090v1）](https://arxiv.org/abs/2111.11090v1)
- [2048: Reinforcement Learning in a Delayed Reward Environment（arXiv:2507.05465v1）](https://arxiv.org/abs/2507.05465v1)


## V3.1.1 小更新

- 修复混合精度训练中的数值溢出和经验回放异常，补充训练恢复与诊断信息。
- 新增 Expectimax-2。固定20,000局FP32模型的千局对照中，平均分由38,586.8提高至61,807.1，达到4096的比例由28.6%提高至81.8%；搜索会增加计算开销。
- 新增Windows视觉玩家：框选棋盘、识别数字，使用WASD或方向键操作；支持异步截图和识别缓存，识别异常时暂停，确认终局后退出。

视觉模块位于 `visual/`。安装：`python -m pip install torch numpy Pillow opencv-python onnxruntime rapidocr-onnxruntime==1.4.4`；启动：`python -m visual.desktop_player`，或双击 `visual/launch_desktop_player.cmd`。默认权重路径为 `models/v3_1/dqn_ep20000.pth`，也可在窗口中选择模型。
