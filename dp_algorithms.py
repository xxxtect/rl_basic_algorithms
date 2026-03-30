"""
动态规划（DP）算法实现

本模块包含两种基于动态规划的强化学习算法：
1. PolicyIteration: 策略迭代
2. ValueIteration: 价值迭代

这两种算法都需要完整的环境模型（状态转移概率 P）。
适用于离散状态空间和离散动作空间的 MDP 问题。
"""

import numpy as np
from typing import Tuple, Optional


class PolicyIteration:
    """
    策略迭代算法 (Policy Iteration)
    
    策略迭代通过交替进行策略评估和策略提升来找到最优策略：
    1. 策略评估 (Policy Evaluation): 计算当前策略的状态价值函数 V(s)
    2. 策略提升 (Policy Improvement): 基于 V(s) 贪婪地改进策略
    
    重复上述过程直到策略收敛。
    
    Attributes:
        env: 环境实例，必须包含 P (转移矩阵), n_states, n_actions
        gamma (float): 折扣因子
        theta (float): 收敛阈值
        V (np.ndarray): 状态价值函数
        policy (np.ndarray): 策略（每个状态的动作分布或确定性动作）
    """
    
    def __init__(self, env, gamma: float = 0.99, theta: float = 1e-6):
        """
        初始化策略迭代算法
        
        Args:
            env: 环境实例，需包含 P, n_states, n_actions 属性
            gamma: 折扣因子，默认为 0.99
            theta: 收敛阈值，默认为 1e-6
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        
        # 初始化状态价值函数为 0
        self.V = np.zeros(self.n_states)
        # 初始化策略为均匀随机（每个动作等概率）
        self.policy = np.ones((self.n_states, self.n_actions)) / self.n_actions
    
    def policy_evaluation(self) -> np.ndarray:
        """
        策略评估：计算当前策略的状态价值函数
        
        使用迭代策略评估方法，反复应用 Bellman 期望方程：
        V(s) = Σ_a π(a|s) * Σ_{s',r} P(s'|s,a) * [r + γ * V(s')]
        
        Returns:
            V: 收敛后的状态价值函数
        """
        while True:
            delta = 0
            # 对每个状态进行更新
            for s in range(self.n_states):
                v = self.V[s]
                
                # 计算当前策略下的状态价值
                # V(s) = Σ_a π(a|s) * Σ_{s',r} P(s'|s,a,r) * [r + γ * V(s')]
                new_v = 0.0
                for a in range(self.n_actions):
                    action_prob = self.policy[s][a]
                    # 获取状态 s 执行动作 a 的转移概率
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        if done:
                            # 终止状态，没有后续价值
                            new_v += action_prob * prob * reward
                        else:
                            new_v += action_prob * prob * (reward + self.gamma * self.V[next_state])
                
                self.V[s] = new_v
                delta = max(delta, abs(v - self.V[s]))
            
            # 检查是否收敛
            if delta < self.theta:
                break
        
        return self.V
    
    def policy_improvement(self) -> np.ndarray:
        """
        策略提升：基于当前价值函数改进策略
        
        对每个状态，选择能获得最大动作价值的动作：
        π'(s) = argmax_a Σ_{s',r} P(s'|s,a) * [r + γ * V(s')]
        
        Returns:
            new_policy: 改进后的策略（确定性策略，one-hot 编码）
        """
        new_policy = np.zeros((self.n_states, self.n_actions))
        policy_stable = True
        
        for s in range(self.n_states):
            # 记录当前动作（用于检查策略是否稳定）
            old_action = np.argmax(self.policy[s])
            
            # 计算每个动作的动作价值 Q(s,a)
            action_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                for prob, next_state, reward, done in self.env.P[s][a]:
                    if done:
                        action_values[a] += prob * reward
                    else:
                        action_values[a] += prob * (reward + self.gamma * self.V[next_state])
            
            # 贪婪策略：选择价值最高的动作
            best_action = np.argmax(action_values)
            new_policy[s][best_action] = 1.0
            
            # 如果有任何状态的动作改变，策略不稳定
            if best_action != old_action:
                policy_stable = False
        
        self.policy = new_policy
        return self.policy, policy_stable
    
    def run(self, max_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        运行策略迭代算法
        
        Args:
            max_iterations: 最大迭代次数
            
        Returns:
            V: 最优状态价值函数
            policy: 最优策略
        """
        for i in range(max_iterations):
            # 策略评估
            self.policy_evaluation()
            # 策略提升
            new_policy, policy_stable = self.policy_improvement()
            
            if policy_stable:
                print(f"策略迭代在第 {i+1} 次迭代后收敛")
                break
        
        return self.V, self.policy
    
    def get_deterministic_policy(self) -> np.ndarray:
        """
        获取确定性策略（每个状态的最优动作）
        
        Returns:
            actions: 长度为 n_states 的数组，每个元素是对应状态的最优动作
        """
        return np.argmax(self.policy, axis=1)


class ValueIteration:
    """
    价值迭代算法 (Value Iteration)
    
    价值迭代直接通过 Bellman 最优方程迭代更新价值函数：
    V(s) = max_a Σ_{s',r} P(s'|s,a) * [r + γ * V(s')]
    
    价值迭代可以看作是策略迭代的简化版本，它将策略评估
    和策略提升合并为一步更新。每次迭代都向最优价值函数
    靠近。
    
    Attributes:
        env: 环境实例，必须包含 P (转移矩阵), n_states, n_actions
        gamma (float): 折扣因子
        theta (float): 收敛阈值
        V (np.ndarray): 状态价值函数
        policy (np.ndarray): 从价值函数推导出的最优策略
    """
    
    def __init__(self, env, gamma: float = 0.99, theta: float = 1e-6):
        """
        初始化价值迭代算法
        
        Args:
            env: 环境实例，需包含 P, n_states, n_actions 属性
            gamma: 折扣因子，默认为 0.99
            theta: 收敛阈值，默认为 1e-6
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        
        # 初始化状态价值函数为 0
        self.V = np.zeros(self.n_states)
        # 策略（从价值函数推导）
        self.policy = np.zeros(self.n_states, dtype=int)
    
    def run(self, max_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        运行价值迭代算法
        
        使用 Bellman 最优方程进行迭代更新：
        V(s) = max_a Σ_{s',r} P(s'|s,a) * [r + γ * V(s')]
        
        Args:
            max_iterations: 最大迭代次数
            
        Returns:
            V: 最优状态价值函数
            policy: 最优策略
        """
        for i in range(max_iterations):
            delta = 0
            
            # 对每个状态进行更新
            for s in range(self.n_states):
                v = self.V[s]
                
                # 计算所有动作的价值，选择最大值
                action_values = []
                for a in range(self.n_actions):
                    q_sa = 0.0
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        if done:
                            q_sa += prob * reward
                        else:
                            q_sa += prob * (reward + self.gamma * self.V[next_state])
                    action_values.append(q_sa)
                
                # Bellman 最优更新
                self.V[s] = max(action_values)
                delta = max(delta, abs(v - self.V[s]))
            
            # 检查是否收敛
            if delta < self.theta:
                print(f"价值迭代在第 {i+1} 次迭代后收敛")
                break
        
        # 从最优价值函数推导最优策略
        self._extract_policy()
        
        return self.V, self.policy
    
    def _extract_policy(self):
        """
        从价值函数中提取最优策略
        
        对每个状态，选择能获得最大期望回报的动作：
        π(s) = argmax_a Σ_{s',r} P(s'|s,a) * [r + γ * V(s')]
        """
        for s in range(self.n_states):
            action_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                for prob, next_state, reward, done in self.env.P[s][a]:
                    if done:
                        action_values[a] += prob * reward
                    else:
                        action_values[a] += prob * (reward + self.gamma * self.V[next_state])
            
            self.policy[s] = np.argmax(action_values)
    
    def get_deterministic_policy(self) -> np.ndarray:
        """
        获取确定性策略
        
        Returns:
            policy: 长度为 n_states 的数组，每个元素是对应状态的最优动作
        """
        return self.policy


# ==================== 示例用法 ====================
if __name__ == "__main__":
    from cliff_walking_env import CliffWalkingEnvDP
    
    # 创建环境
    env = CliffWalkingEnvDP()
    
    print("=" * 40)
    print("策略迭代测试")
    print("=" * 40)
    
    pi = PolicyIteration(env, gamma=0.99, theta=1e-6)
    V, policy = pi.run()
    print(f"最优策略动作序列：{pi.get_deterministic_policy()}")
    
    print("\n" + "=" * 40)
    print("价值迭代测试")
    print("=" * 40)
    
    vi = ValueIteration(env, gamma=0.99, theta=1e-6)
    V, policy = vi.run()
    print(f"最优策略动作序列：{vi.get_deterministic_policy()}")
