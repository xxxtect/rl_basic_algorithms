"""
多臂老虎机（MAB）算法实现

本模块包含三种经典的多臂老虎机算法：
1. EpsilonGreedy: epsilon-贪婪算法
2. UCB: 上置信界算法 (Upper Confidence Bound)
3. ThompsonSampling: 汤普森采样算法

所有算法都继承自 mab_env 中的 Solver 基类。
"""

import numpy as np
from typing import Optional
from mab_env import BernoulliBandit, Solver


class EpsilonGreedy(Solver):
    """
    Epsilon-贪婪算法
    
    策略：
    - 以 epsilon 的概率随机探索（均匀选择任意臂）
    - 以 1-epsilon 的概率利用（选择当前估计价值最高的臂）
    
    这是一种简单而有效的探索 - 利用平衡策略。
    
    Attributes:
        epsilon (float): 探索概率，范围 [0, 1]
    """
    
    def __init__(self, bandit: BernoulliBandit, epsilon: float = 0.1):
        """
        初始化 Epsilon-Greedy 求解器
        
        Args:
            bandit: BernoulliBandit 环境实例
            epsilon: 探索概率，默认为 0.1
        """
        super().__init__(bandit)
        self.epsilon = epsilon
    
    def select_action(self) -> int:
        """
        根据 epsilon-贪婪策略选择动作
        
        Returns:
            选择的臂的索引
        """
        # 以 epsilon 的概率进行探索（随机选择）
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            # 以 1-epsilon 的概率进行利用（选择最优估计）
            # 使用 argmax 选择价值最高的臂，平局时随机选择
            return np.argmax(self.values)
    
    def run_one_step(self) -> float:
        """
        运行一步算法
        
        执行一次动作选择、环境交互和价值更新。
        
        Returns:
            reward: 获得的奖励
        """
        # 选择动作
        action = self.select_action()
        # 执行动作并获得奖励
        reward = self.bandit.step(action)
        # 更新估计值
        self.update(action, reward)
        return reward


class UCB(Solver):
    """
    上置信界算法 (Upper Confidence Bound)
    
    策略：
    - 选择 UCB 值最高的臂：UCB(a) = Q(a) + c * sqrt(ln(t) / N(a))
    - 其中 Q(a) 是臂 a 的平均奖励估计
    - N(a) 是臂 a 被选择的次数
    - t 是总步数
    - c 是探索参数，控制探索强度
    
    UCB 算法通过置信上界自动平衡探索和利用：
    - 被选择次数少的臂有更高的不确定性，因此 UCB 值更高
    - 随着选择次数增加，不确定性项逐渐减小
    
    Attributes:
        c (float): 探索参数，值越大越倾向于探索
    """
    
    def __init__(self, bandit: BernoulliBandit, c: float = 2.0):
        """
        初始化 UCB 求解器
        
        Args:
            bandit: BernoulliBandit 环境实例
            c: 探索参数，默认为 2.0
        """
        super().__init__(bandit)
        self.c = c
    
    def select_action(self) -> int:
        """
        根据 UCB 策略选择动作
        
        Returns:
            选择的臂的索引
        """
        # 首先确保每个臂至少被选择一次
        untried_arms = np.where(self.counts == 0)[0]
        if len(untried_arms) > 0:
            return np.random.choice(untried_arms)
        
        # 计算每个臂的 UCB 值
        # UCB(a) = Q(a) + c * sqrt(ln(t) / N(a))
        t = self.total_steps
        ucb_values = self.values + self.c * np.sqrt(np.log(t) / self.counts)
        
        # 选择 UCB 值最高的臂
        return np.argmax(ucb_values)
    
    def run_one_step(self) -> float:
        """
        运行一步算法
        
        Returns:
            reward: 获得的奖励
        """
        action = self.select_action()
        reward = self.bandit.step(action)
        self.update(action, reward)
        return reward


class ThompsonSampling(Solver):
    """
    汤普森采样算法 (Thompson Sampling)
    
    策略：
    - 为每个臂维护一个后验分布（Beta 分布适用于伯努利奖励）
    - 每次从每个臂的后验分布中采样一个价值
    - 选择采样价值最高的臂
    
    对于伯努利老虎机，使用 Beta(α, β) 分布：
    - α = 成功次数 + 1（先验）
    - β = 失败次数 + 1（先验）
    
    Thompson Sampling 是一种贝叶斯方法，通过概率匹配实现
    探索和利用的自然平衡。
    
    Attributes:
        alpha (np.ndarray): Beta 分布的成功参数
        beta (np.ndarray): Beta 分布的失败参数
    """
    
    def __init__(self, bandit: BernoulliBandit):
        """
        初始化汤普森采样求解器
        
        使用 Beta(1, 1) 作为先验分布（均匀分布）。
        
        Args:
            bandit: BernoulliBandit 环境实例
        """
        super().__init__(bandit)
        # Beta 分布参数，初始为 Beta(1, 1) 即均匀分布
        # alpha 表示成功次数 + 1，beta 表示失败次数 + 1
        self.alpha = np.ones(self.n_arms)
        self.beta = np.ones(self.n_arms)
    
    def select_action(self) -> int:
        """
        根据汤普森采样策略选择动作
        
        从每个臂的后验 Beta 分布中采样，选择采样值最大的臂。
        
        Returns:
            选择的臂的索引
        """
        # 从每个臂的 Beta 后验分布中采样
        sampled_values = np.random.beta(self.alpha, self.beta)
        # 选择采样值最高的臂
        return np.argmax(sampled_values)
    
    def update(self, action: int, reward: float):
        """
        更新后验分布参数
        
        根据观察到的奖励更新 Beta 分布参数：
        - 奖励为 1（成功）：alpha += 1
        - 奖励为 0（失败）：beta += 1
        
        Args:
            action: 执行的臂的索引
            reward: 获得的奖励（0 或 1）
        """
        if reward == 1:
            self.alpha[action] += 1
        else:
            self.beta[action] += 1
        
        # 更新计数和价值估计（用于 regret 计算）
        self.counts[action] += 1
        n = self.counts[action]
        self.values[action] += (reward - self.values[action]) / n
        self.total_steps += 1
    
    def run_one_step(self) -> float:
        """
        运行一步算法
        
        Returns:
            reward: 获得的奖励
        """
        action = self.select_action()
        reward = self.bandit.step(action)
        self.update(action, reward)
        return reward


# ==================== 示例用法 ====================
if __name__ == "__main__":
    # 创建一个 10 臂老虎机进行测试
    np.random.seed(42)
    true_probs = np.random.uniform(0.1, 0.9, 10)
    bandit = BernoulliBandit(probs=true_probs.tolist())
    
    print(f"真实概率：{true_probs}")
    print(f"最优臂：{bandit.get_optimal_arm()}, 最优概率：{bandit.get_optimal_value():.3f}\n")
    
    # 测试三种算法
    algorithms = [
        ("Epsilon-Greedy (ε=0.1)", EpsilonGreedy(bandit, epsilon=0.1)),
        ("UCB (c=2.0)", UCB(bandit, c=2.0)),
        ("Thompson Sampling", ThompsonSampling(bandit)),
    ]
    
    for name, solver in algorithms:
        solver.reset()
        regrets = solver.run(num_steps=1000)
        print(f"{name}: 最终累积 regret = {regrets[-1]:.2f}")
