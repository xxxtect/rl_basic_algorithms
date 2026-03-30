"""
强化学习基础算法演示入口

本模块包含多个 demo 函数，用于演示和测试各种强化学习算法：
- 多臂老虎机算法 (MAB)
- 动态规划算法 (DP)
- 时序差分算法 (TD)

运行方式：
    python main.py
    
或运行特定 demo：
    python main.py --demo td_comparison
"""

import numpy as np
import argparse
from typing import List

# 导入环境
from cliff_walking_env import CliffWalkingEnvTD, CliffWalkingEnvDP
from mab_env import BernoulliBandit

# 导入 MAB 算法
from mab_algorithms import EpsilonGreedy, UCB, ThompsonSampling

# 导入 DP 算法
from dp_algorithms import PolicyIteration, ValueIteration

# 导入 TD 算法
from td_algorithms import Sarsa, NStepSarsa, QLearning, DynaQ

# 导入工具函数
from utils import print_agent, print_training_progress, compare_agents


def demo_mab_algorithms():
    """
    演示多臂老虎机算法
    
    比较 Epsilon-Greedy、UCB 和 Thompson Sampling 三种算法的性能。
    """
    print("\n" + "=" * 60)
    print("  多臂老虎机算法演示")
    print("=" * 60)
    
    # 创建一个 10 臂老虎机
    np.random.seed(42)
    true_probs = np.random.uniform(0.1, 0.9, 10).tolist()
    bandit = BernoulliBandit(probs=true_probs)
    
    print(f"\n老虎机配置：{len(true_probs)} 个臂")
    print(f"真实概率：{[f'{p:.2f}' for p in true_probs]}")
    print(f"最优臂：{bandit.get_optimal_arm()}, 最优概率：{bandit.get_optimal_value():.3f}")
    
    # 配置算法
    algorithms = [
        ("Epsilon-Greedy (ε=0.1)", EpsilonGreedy(bandit, epsilon=0.1)),
        ("Epsilon-Greedy (ε=0.01)", EpsilonGreedy(bandit, epsilon=0.01)),
        ("UCB (c=2.0)", UCB(bandit, c=2.0)),
        ("Thompson Sampling", ThompsonSampling(bandit)),
    ]
    
    # 运行比较
    num_steps = 1000
    results = {}
    
    print(f"\n运行 {num_steps} 步...")
    
    for name, solver in algorithms:
        solver.reset()
        regrets = solver.run(num_steps)
        results[name] = regrets[-1]
        print(f"  {name}: 最终累积 regret = {regrets[-1]:.2f}")
    
    # 找出最佳算法
    best_algo = min(results, key=results.get)
    print(f"\n最佳算法：{best_algo}")


def demo_dp_algorithms():
    """
    演示动态规划算法
    
    使用策略迭代和价值迭代求解悬崖漫步问题。
    """
    print("\n" + "=" * 60)
    print("  动态规划算法演示")
    print("=" * 60)
    
    # 创建环境（DP 版本，需要转移矩阵）
    env = CliffWalkingEnvDP()
    
    print(f"\n环境：悬崖漫步 (Cliff Walking)")
    print(f"状态数：{env.n_states}, 动作数：{env.n_actions}")
    
    # 策略迭代
    print("\n--- 策略迭代 (Policy Iteration) ---")
    pi = PolicyIteration(env, gamma=0.99, theta=1e-6)
    V_pi, policy_pi = pi.run()
    print_agent(pi.get_deterministic_policy(), title="策略迭代结果")
    
    # 价值迭代
    print("\n--- 价值迭代 (Value Iteration) ---")
    vi = ValueIteration(env, gamma=0.99, theta=1e-6)
    V_vi, policy_vi = vi.run()
    print_agent(vi.get_deterministic_policy(), title="价值迭代结果")


def demo_td_comparison():
    """
    演示时序差分算法对比
    
    比较 Sarsa (On-policy) 和 Q-learning (Off-policy) 在悬崖漫步环境中的表现。
    """
    print("\n" + "=" * 60)
    print("  时序差分算法对比：Sarsa vs Q-learning")
    print("=" * 60)
    
    # 算法参数
    num_episodes = 500
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1
    
    print(f"\n训练参数:")
    print(f"  回合数：{num_episodes}")
    print(f"  学习率 α: {alpha}")
    print(f"  折扣因子 γ: {gamma}")
    print(f"  探索概率 ε: {epsilon}")
    
    # 训练 Sarsa
    print("\n--- 训练 Sarsa (On-policy) ---")
    env_sarsa = CliffWalkingEnvTD()
    sarsa = Sarsa(env_sarsa, alpha=alpha, gamma=gamma, epsilon=epsilon)
    sarsa_rewards: List[float] = []
    
    for episode in range(num_episodes):
        reward = sarsa.run_episode()
        sarsa_rewards.append(reward)
        
        if (episode + 1) % 100 == 0:
            avg = np.mean(sarsa_rewards[-100:])
            print(f"  回合 {episode + 1}/{num_episodes}, 最近 100 局平均奖励：{avg:.2f}")
    
    print_training_progress(sarsa_rewards)
    
    # 训练 Q-learning
    print("\n--- 训练 Q-learning (Off-policy) ---")
    env_q = CliffWalkingEnvTD()
    qlearning = QLearning(env_q, alpha=alpha, gamma=gamma, epsilon=epsilon)
    qlearning_rewards: List[float] = []
    
    for episode in range(num_episodes):
        reward = qlearning.run_episode()
        qlearning_rewards.append(reward)
        
        if (episode + 1) % 100 == 0:
            avg = np.mean(qlearning_rewards[-100:])
            print(f"  回合 {episode + 1}/{num_episodes}, 最近 100 局平均奖励：{avg:.2f}")
    
    print_training_progress(qlearning_rewards)
    
    # 对比结果
    compare_agents("Sarsa", sarsa_rewards, "Q-learning", qlearning_rewards)
    
    # 打印最终策略
    print("\n" + "=" * 60)
    print("  最终策略对比")
    print("=" * 60)
    
    print("\n--- Sarsa 策略 (On-policy, 更保守安全) ---")
    print_agent(sarsa.get_policy(), sarsa.Q, title="Sarsa")
    
    print("\n--- Q-learning 策略 (Off-policy, 更激进最优) ---")
    print_agent(qlearning.get_policy(), qlearning.Q, title="Q-learning")


def demo_all_td_algorithms():
    """
    演示所有 TD 算法
    
    包括 Sarsa, N-step Sarsa, Q-learning, 和 Dyna-Q。
    """
    print("\n" + "=" * 60)
    print("  所有 TD 算法演示")
    print("=" * 60)
    
    num_episodes = 300
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1
    
    algorithms = [
        ("Sarsa", lambda env: Sarsa(env, alpha=alpha, gamma=gamma, epsilon=epsilon)),
        ("N-step Sarsa (n=4)", lambda env: NStepSarsa(env, alpha=alpha, gamma=gamma, epsilon=epsilon, n=4)),
        ("Q-learning", lambda env: QLearning(env, alpha=alpha, gamma=gamma, epsilon=epsilon)),
        ("Dyna-Q (planning=10)", lambda env: DynaQ(env, alpha=alpha, gamma=gamma, epsilon=epsilon, n_planning=10)),
    ]
    
    results = {}
    
    for name, algo_factory in algorithms:
        print(f"\n--- 训练 {name} ---")
        env = CliffWalkingEnvTD()
        agent = algo_factory(env)
        rewards = []
        
        for episode in range(num_episodes):
            reward = agent.run_episode()
            rewards.append(reward)
        
        # 计算最近 50 局的平均奖励
        recent_avg = np.mean(rewards[-50:])
        results[name] = recent_avg
        print(f"  最近 50 局平均奖励：{recent_avg:.2f}")
    
    # 排名
    print("\n" + "=" * 60)
    print("  算法排名（按最近 50 局平均奖励）")
    print("=" * 60)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, score) in enumerate(sorted_results, 1):
        print(f"  {rank}. {name}: {score:.2f}")


def demo_dyna_q_planning():
    """
    演示 Dyna-Q 中规划次数的影响
    """
    print("\n" + "=" * 60)
    print("  Dyna-Q 规划次数对比")
    print("=" * 60)
    
    num_episodes = 200
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1
    
    planning_values = [0, 5, 10, 50]
    results = {}
    
    for n_plan in planning_values:
        print(f"\n--- Dyna-Q (planning={n_plan}) ---")
        env = CliffWalkingEnvTD()
        agent = DynaQ(env, alpha=alpha, gamma=gamma, epsilon=epsilon, n_planning=n_plan)
        rewards = []
        
        for episode in range(num_episodes):
            reward = agent.run_episode()
            rewards.append(reward)
        
        recent_avg = np.mean(rewards[-50:])
        results[n_plan] = recent_avg
        print(f"  最近 50 局平均奖励：{recent_avg:.2f}")
    
    print("\n" + "=" * 60)
    print("  结果总结")
    print("=" * 60)
    for n_plan, score in results.items():
        bar = "█" * int(score / 10)
        print(f"  planning={n_plan:2d}: {score:7.2f} {bar}")


def main():
    """
    主函数：解析命令行参数并运行相应的 demo
    """
    parser = argparse.ArgumentParser(
        description="强化学习基础算法演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
可用 demo:
  mab         - 多臂老虎机算法对比
  dp          - 动态规划算法（策略迭代、价值迭代）
  td          - TD 算法对比（Sarsa vs Q-learning）
  td_all      - 所有 TD 算法演示
  dyna        - Dyna-Q 规划次数对比
  all         - 运行所有 demo（默认）
        """
    )
    
    parser.add_argument(
        "--demo", 
        type=str, 
        default="all",
        choices=["mab", "dp", "td", "td_all", "dyna", "all"],
        help="选择要运行的 demo"
    )
    
    args = parser.parse_args()
    
    print("\n" + "#" * 60)
    print("#  强化学习基础算法演示程序")
    print("#" * 60)
    
    if args.demo == "mab" or args.demo == "all":
        demo_mab_algorithms()
    
    if args.demo == "dp" or args.demo == "all":
        demo_dp_algorithms()
    
    if args.demo == "td" or args.demo == "all":
        demo_td_comparison()
    
    if args.demo == "td_all" or args.demo == "all":
        demo_all_td_algorithms()
    
    if args.demo == "dyna" or args.demo == "all":
        demo_dyna_q_planning()
    
    print("\n" + "#" * 60)
    print("#  演示结束")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
