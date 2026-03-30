"""
多臂老虎机（Multi-Armed Bandit, MAB）环境实现

本模块包含：
1. BernoulliBandit: 伯努利多臂老虎机环境类
2. Solver: 通用求解器基类，包含 run 逻辑和 regret 记录
"""

import numpy as np
from typing import List, Optional


class BernoulliBandit:
    """
    伯努利多臂老虎机环境
    
    每个臂（action）都有一个固定的获奖概率 p，
    拉动臂时以概率 p 获得奖励 1，以概率 1-p 获得奖励 0。
    
    Attributes:
        probs (List[float]): 每个臂的获奖概率列表
        n_arms (int): 臂的数量
    """
    
    def __init__(self, probs: List[float]):
        """
        初始化多臂老虎机
        
        Args:
            probs: 每个臂的获奖概率列表，例如 [0.3, 0.5, 0.7]
        """
        self.probs = probs
        self.n_arms = len(probs)
        # 计算最优臂的索引（获奖概率最高的臂）
        self.optimal_arm = np.argmax(probs)
        # 最优臂的期望奖励
        self.optimal_value = max(probs)
    
    def step(self, action: int) -> float:
        """
        执行一次动作（拉动一个臂），返回奖励
        
        Args:
            action: 选择的臂的索引，范围 [0, n_arms)
            
        Returns:
            reward: 获得的奖励（0 或 1）
            
        Raises:
            ValueError: 当 action 超出有效范围时
        """
        if action < 0 or action >= self.n_arms:
            raise ValueError(f"Action {action} out of range [0, {self.n_arms})")
        
        # 根据该臂的获奖概率生成伯努利随机奖励
        reward = 1.0 if np.random.random() < self.probs[action] else 0.0
        return reward
    
    def get_optimal_value(self) -> float:
        """
        获取最优臂的期望奖励
        
        Returns:
            最优臂的获奖概率
        """
        return self.optimal_value
    
    def get_optimal_arm(self) -> int:
        """
        获取最优臂的索引
        
        Returns:
            最优臂的索引
        """
        return self.optimal_arm


class Solver:
    """
    多臂老虎机求解器基类
    
    定义了求解器的基本接口和通用的运行逻辑。
    具体的算法（如 ε-greedy、UCB、Thompson Sampling 等）
    需要继承此类并实现 select_action 方法。
    
    Attributes:
        bandit (BernoulliBandit): 老虎机环境实例
        n_arms (int): 臂的数量
        counts (np.ndarray): 每个臂被选择的次数
        values (np.ndarray): 每个臂的平均奖励估计
        total_steps (int): 总步数
        regrets (List[float]): 每一步的累积 regret 记录
    """
    
    def __init__(self, bandit: BernoulliBandit):
        """
        初始化求解器
        
        Args:
            bandit: BernoulliBandit 环境实例
        """
        self.bandit = bandit
        self.n_arms = bandit.n_arms
        # 每个臂被选择的次数，初始为 0
        self.counts = np.zeros(self.n_arms)
        # 每个臂的平均奖励估计，初始为 0
        self.values = np.zeros(self.n_arms)
        # 总步数
        self.total_steps = 0
        # 累积 regret 记录列表
        self.regrets = []
    
    def select_action(self) -> int:
        """
        选择下一个要执行的臂（动作）
        
        这是一个抽象方法，需要子类实现具体的选择策略。
        
        Returns:
            选择的臂的索引
            
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("Subclasses must implement select_action method")
    
    def update(self, action: int, reward: float):
        """
        根据观察到的奖励更新臂的估计值
        
        使用增量式更新公式：
        Q_{n+1} = Q_n + (1/n) * (R_n - Q_n)
        
        Args:
            action: 执行的臂的索引
            reward: 获得的奖励
        """
        self.counts[action] += 1
        n = self.counts[action]
        # 增量式更新平均奖励估计
        self.values[action] += (reward - self.values[action]) / n
        self.total_steps += 1
    
    def run(self, num_steps: int, verbose: bool = False) -> List[float]:
        """
        运行求解器指定步数
        
        Args:
            num_steps: 运行的总步数
            verbose: 是否打印运行信息
            
        Returns:
            regrets: 每一步的累积 regret 列表
        """
        self.regrets = []
        cumulative_regret = 0.0
        optimal_value = self.bandit.get_optimal_value()
        
        for step in range(num_steps):
            # 根据策略选择动作
            action = self.select_action()
            # 执行动作并获得奖励
            reward = self.bandit.step(action)
            # 更新估计值
            self.update(action, reward)
            
            # 计算当前步的 regret（期望损失）
            # regret = 最优臂的期望奖励 - 当前臂的期望奖励
            instant_regret = optimal_value - self.bandit.probs[action]
            cumulative_regret += instant_regret
            self.regrets.append(cumulative_regret)
            
            if verbose and (step + 1) % 1000 == 0:
                print(f"Step {step + 1}/{num_steps}, Cumulative Regret: {cumulative_regret:.2f}")
        
        return self.regrets
    
    def reset(self):
        """
        重置求解器状态，以便重新运行
        """
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.total_steps = 0
        self.regrets = []


# ==================== 示例用法 ====================
if __name__ == "__main__":
    # 创建一个简单的 3 臂老虎机
    bandit = BernoulliBandit(probs=[0.3, 0.5, 0.7])
    print(f"最优臂索引：{bandit.get_optimal_arm()}, 最优期望奖励：{bandit.get_optimal_value()}")
    
    # 示例：一个简单的随机策略求解器
    class RandomSolver(Solver):
        def select_action(self) -> int:
            return np.random.randint(self.n_arms)
    
    solver = RandomSolver(bandit)
    regrets = solver.run(num_steps=1000)
    print(f"最终累积 regret: {regrets[-1]:.2f}")
