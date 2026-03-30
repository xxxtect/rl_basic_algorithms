"""
工具函数模块

本模块包含强化学习算法的辅助工具函数，主要用于可视化和调试。
"""

import numpy as np
from typing import Optional


# 动作符号映射（上、下、左、右）
ACTION_SYMBOLS = ['^', 'v', '<', '>']

# 网格尺寸（与 cliff_walking_env 保持一致）
GRID_ROWS = 4
GRID_COLS = 12


def print_agent(policy: np.ndarray, Q: Optional[np.ndarray] = None, 
                title: str = "Agent Strategy"):
    """
    打印 CliffWalking 环境下 agent 的策略和状态价值
    
    在终端以网格形式展示：
    - 策略：使用 ^ (上)、v (下)、< (左)、> (右) 表示每个状态的最优动作
    - 价值：可选，显示每个状态的最大 Q 值或 V 值
    
    特殊标记：
    - S: 起点 (Start)
    - G: 终点 (Goal)
    - C: 悬崖 (Cliff)
    
    Args:
        policy: 策略数组，形状为 (n_states,) 或 (n_states, n_actions)
                如果是确定性策略，每个元素是对应状态的动作索引
                如果是随机策略，形状为 (n_states, n_actions)
        Q: 可选，动作价值函数，形状为 (n_states, n_actions)
           如果提供，将显示每个状态的最大 Q 值
        title: 输出标题
    
    Example:
        >>> policy = np.array([...])  # 长度为 48 的数组
        >>> Q = np.zeros((48, 4))
        >>> print_agent(policy, Q, title="Q-learning 策略")
    """
    n_states = len(policy)
    
    print()
    print("=" * 50)
    print(f"  {title}")
    print("=" * 50)
    
    # 计算状态价值（如果提供了 Q 函数）
    if Q is not None:
        V = np.max(Q, axis=1)
    else:
        V = None
    
    # 起点、终点、悬崖的状态索引
    start_state = (GRID_ROWS - 1) * GRID_COLS  # 36
    goal_state = (GRID_ROWS - 1) * GRID_COLS + (GRID_COLS - 1)  # 47
    cliff_states = list(range((GRID_ROWS - 1) * GRID_COLS + 1, 
                               (GRID_ROWS - 1) * GRID_COLS + GRID_COLS - 1))  # 37-46
    
    # 打印表头
    print("策略可视化 (^=上，v=下，<=左，>=右，S=起点，G=终点，C=悬崖)")
    if V is not None:
        print("括号内为状态价值 V(s)")
    print("-" * 60)
    
    # 逐行打印
    for row in range(GRID_ROWS):
        # 策略行
        policy_line = "|"
        value_line = "|"
        
        for col in range(GRID_COLS):
            state = row * GRID_COLS + col
            
            if state == start_state:
                policy_line += "  S  |"
                if V is not None:
                    value_line += f"{V[state]:6.1f}|"
            elif state == goal_state:
                policy_line += "  G  |"
                if V is not None:
                    value_line += f"{V[state]:6.1f}|"
            elif state in cliff_states:
                policy_line += "  C  |"
                if V is not None:
                    value_line += f"{V[state]:6.1f}|"
            else:
                # 获取动作
                if policy.ndim == 1:
                    action = policy[state]
                else:
                    action = np.argmax(policy[state])
                
                symbol = ACTION_SYMBOLS[action]
                policy_line += f"  {symbol}  |"
                
                if V is not None:
                    value_line += f"{V[state]:6.1f}|"
        
        print(policy_line)
        if V is not None:
            print(value_line)
            print("-" * 60)
        else:
            print("-" * 60)
    
    print()


def print_training_progress(episode_rewards: list, window: int = 50):
    """
    打印训练进度信息
    
    显示最近 window 个 episode 的平均奖励和总趋势。
    
    Args:
        episode_rewards: 每个 episode 的总奖励列表
        window: 滑动窗口大小，用于计算平均奖励
    """
    if len(episode_rewards) == 0:
        return
    
    print("\n" + "=" * 50)
    print("  训练进度")
    print("=" * 50)
    
    total_episodes = len(episode_rewards)
    print(f"总回合数：{total_episodes}")
    
    # 最近 window 局的平均奖励
    recent_rewards = episode_rewards[-window:]
    avg_recent = np.mean(recent_rewards)
    print(f"最近 {window} 局平均奖励：{avg_recent:.2f}")
    
    # 整体统计
    print(f"最佳单局奖励：{max(episode_rewards):.2f}")
    print(f"最差单局奖励：{min(episode_rewards):.2f}")
    print(f"全局平均奖励：{np.mean(episode_rewards):.2f}")
    
    # 简单趋势分析
    if len(episode_rewards) >= window * 2:
        first_half_avg = np.mean(episode_rewards[:window])
        second_half_avg = np.mean(episode_rewards[-window:])
        if second_half_avg > first_half_avg:
            print("趋势：↑ 性能提升中")
        elif second_half_avg < first_half_avg:
            print("趋势：↓ 性能下降")
        else:
            print("趋势：→ 性能稳定")
    
    print()


def compare_agents(agent1_name: str, agent1_rewards: list, 
                   agent2_name: str, agent2_rewards: list, window: int = 50):
    """
    对比两个 agent 的训练效果
    
    Args:
        agent1_name: 第一个 agent 的名称
        agent1_rewards: 第一个 agent 的 episode 奖励列表
        agent2_name: 第二个 agent 的名称
        agent2_rewards: 第二个 agent 的 episode 奖励列表
        window: 滑动窗口大小
    """
    print("\n" + "=" * 50)
    print(f"  Agent 对比：{agent1_name} vs {agent2_name}")
    print("=" * 50)
    
    # 计算最近 window 局的平均奖励
    a1_recent = np.mean(agent1_rewards[-window:]) if len(agent1_rewards) >= window else np.mean(agent1_rewards)
    a2_recent = np.mean(agent2_rewards[-window:]) if len(agent2_rewards) >= window else np.mean(agent2_rewards)
    
    print(f"\n{agent1_name}:")
    print(f"  最近 {window} 局平均奖励：{a1_recent:.2f}")
    
    print(f"\n{agent2_name}:")
    print(f"  最近 {window} 局平均奖励：{a2_recent:.2f}")
    
    # 判断胜负
    print("\n" + "-" * 50)
    if a1_recent > a2_recent:
        print(f"胜者：{agent1_name} (奖励高出 {a1_recent - a2_recent:.2f})")
    elif a2_recent > a1_recent:
        print(f"胜者：{agent2_name} (奖励高出 {a2_recent - a1_recent:.2f})")
    else:
        print("结果：平局")
    
    print()


# ==================== 示例用法 ====================
if __name__ == "__main__":
    # 测试 print_agent 函数
    from cliff_walking_env import CliffWalkingEnvDP
    from dp_algorithms import PolicyIteration
    
    env = CliffWalkingEnvDP()
    pi = PolicyIteration(env)
    V, policy = pi.run()
    
    # 打印策略
    print_agent(pi.get_deterministic_policy(), title="策略迭代结果")
