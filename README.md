# 强化学习基础算法 (Reinforcement Learning Basic Algorithms)

本项目实现了强化学习中的经典基础算法，包括多臂老虎机、动态规划和时序差分算法。

## 项目结构

```
rl_basic_algorithms/
├── mab_env.py           # 多臂老虎机环境（BernoulliBandit, Solver 基类）
├── mab_algorithms.py    # MAB 算法（EpsilonGreedy, UCB, ThompsonSampling）
├── cliff_walking_env.py # 悬崖漫步环境（DP 和 TD 两个版本）
├── dp_algorithms.py     # 动态规划算法（PolicyIteration, ValueIteration）
├── td_algorithms.py     # TD 算法（Sarsa, NStepSarsa, QLearning, DynaQ）
├── utils.py             # 工具函数（print_agent 等）
├── main.py              # 演示入口
├── requirements.txt     # 依赖要求
└── README.md            # 本文件
```

## 依赖要求

- Python >= 3.7
- numpy >= 1.20.0

安装依赖：
```bash
pip install -r requirements.txt
```

## 运行演示

运行所有 demo：
```bash
python main.py
```

运行特定 demo：
```bash
python main.py --demo mab      # 多臂老虎机算法
python main.py --demo dp       # 动态规划算法
python main.py --demo td       # Sarsa vs Q-learning 对比
python main.py --demo td_all   # 所有 TD 算法
python main.py --demo dyna     # Dyna-Q 规划次数对比
```

## 算法说明

### 多臂老虎机 (MAB)
- **Epsilon-Greedy**: 以ε概率探索，1-ε概率利用
- **UCB**: 上置信界，自动平衡探索与利用
- **Thompson Sampling**: 贝叶斯方法，概率匹配

### 动态规划 (DP)
- **Policy Iteration**: 策略评估 + 策略提升
- **Value Iteration**: Bellman 最优方程迭代

### 时序差分 (TD)
- **Sarsa**: On-policy，使用实际执行的动作更新
- **N-step Sarsa**: n 步回报，结合 MC 和 TD 优点
- **Q-learning**: Off-policy，使用 max 操作学习最优策略
- **Dyna-Q**: 结合模型学习的 Q-learning 变体

## On-policy vs Off-policy

| 特性 | On-policy (Sarsa) | Off-policy (Q-learning) |
|------|-------------------|------------------------|
| 目标策略 | 与行为策略相同 | 最优贪婪策略 |
| 更新方式 | 使用实际动作 A' | 使用 max 操作 |
| 特点 | 更保守安全 | 更激进最优 |
| 适用场景 | 需要考虑探索风险 | 学习最优策略 |

## 作者

Created for learning reinforcement learning fundamentals.
