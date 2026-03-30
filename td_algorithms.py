"""
时序差分（TD）算法实现

本模块包含四种无模型的强化学习算法：
1. Sarsa: 单步同策 TD 控制
2. NStepSarsa: n 步 Sarsa
3. QLearning: 异策 TD 控制
4. DynaQ: 结合模型学习的 Q-learning 变体

重要概念：
- On-policy (同策): 学习和评估的是同一个策略（行为策略=目标策略）
  代表算法：Sarsa, NStepSarsa
  
- Off-policy (异策): 学习的策略和行为策略不同
  代表算法：Q-learning, DynaQ
  优势：可以从历史数据中学习，可以学习最优策略同时保持探索
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


class Sarsa:
    """
    Sarsa 算法 (单步同策 TD 控制)
    
    Sarsa 是 On-policy 算法，意味着它学习的是当前行为策略的价值函数。
    更新公式：
    Q(S_t, A_t) ← Q(S_t, A_t) + α * [R_{t+1} + γ * Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
    
    名称来源：State-Action-Reward-State-Action (S-A-R-S-A)
    
    On-policy 特点：
    - 使用实际执行的下一个动作 A_{t+1} 来更新 Q 值
    - 更保守，会考虑探索带来的风险
    - 在悬崖漫步等危险环境中表现更安全
    
    Attributes:
        env: 环境实例，需包含 step(), reset(), n_states, n_actions
        alpha (float): 学习率
        gamma (float): 折扣因子
        epsilon (float): 探索概率
        Q (np.ndarray): 动作价值函数
    """
    
    def __init__(self, env, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1):
        """
        初始化 Sarsa 算法
        
        Args:
            env: 环境实例
            alpha: 学习率，默认为 0.1
            gamma: 折扣因子，默认为 0.99
            epsilon: 探索概率，默认为 0.1
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        
        # 初始化 Q 值为 0
        self.Q = np.zeros((self.n_states, self.n_actions))
    
    def take_action(self, state: int, training: bool = True) -> int:
        """
        基于 epsilon-greedy 策略选择动作
        
        Args:
            state: 当前状态
            training: 是否用于训练（训练时进行探索，测试时纯利用）
            
        Returns:
            action: 选择的动作
        """
        if training and np.random.random() < self.epsilon:
            # 探索：随机选择动作
            return np.random.randint(self.n_actions)
        else:
            # 利用：选择 Q 值最大的动作
            return np.argmax(self.Q[state])
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, next_action: int, done: bool):
        """
        更新 Q 值（Sarsa 核心更新公式）
        
        Q(S, A) ← Q(S, A) + α * [R + γ * Q(S', A') - Q(S, A)]
        
        Args:
            state: 当前状态 S
            action: 当前动作 A
            reward: 获得的奖励 R
            next_state: 下一个状态 S'
            next_action: 下一个动作 A'
            done: 是否 episode 结束
        """
        if done:
            # 终止状态，没有后续价值
            td_target = reward
        else:
            # Sarsa 使用实际执行的下一个动作的 Q 值
            td_target = reward + self.gamma * self.Q[next_state, next_action]
        
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
    
    def run_episode(self) -> float:
        """
        运行一个完整的 episode
        
        Returns:
            total_reward: 该 episode 的总奖励
        """
        state = self.env.reset()
        action = self.take_action(state, training=True)
        
        total_reward = 0
        
        while True:
            next_state, reward, done = self.env.step(action)
            next_action = self.take_action(next_state, training=True)
            
            # 更新 Q 值
            self.update(state, action, reward, next_state, next_action, done)
            
            total_reward += reward
            
            if done:
                break
            
            state = next_state
            action = next_action
        
        return total_reward
    
    def get_policy(self) -> np.ndarray:
        """
        获取当前策略（每个状态的最优动作）
        
        Returns:
            policy: 长度为 n_states 的数组
        """
        return np.argmax(self.Q, axis=1)


class NStepSarsa:
    """
    n 步 Sarsa 算法
    
    n 步 Sarsa 是 Sarsa 的推广，使用 n 步回报来更新 Q 值：
    - 1 步 Sarsa 就是标准的 Sarsa
    - n 步回报：G_{t:t+n} = R_{t+1} + γ*R_{t+2} + ... + γ^{n-1}*R_{t+n} + γ^n*Q(S_{t+n}, A_{t+n})
    
    更新公式：
    Q(S_t, A_t) ← Q(S_t, A_t) + α * [G_{t:t+n} - Q(S_t, A_t)]
    
    特点：
    - 结合了 MC（多步采样）和 TD（单步自举）的优点
    - n 越大，越接近 MC；n 越小，越接近 TD
    
    Attributes:
        n (int): n 步更新的步数
        states_buffer (List): 状态缓存列表
        actions_buffer (List): 动作缓存列表
        rewards_buffer (List): 奖励缓存列表
    """
    
    def __init__(self, env, alpha: float = 0.1, gamma: float = 0.99, 
                 epsilon: float = 0.1, n: int = 4):
        """
        初始化 n 步 Sarsa 算法
        
        Args:
            env: 环境实例
            alpha: 学习率
            gamma: 折扣因子
            epsilon: 探索概率
            n: n 步更新的步数
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n = n
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        
        # 初始化 Q 值
        self.Q = np.zeros((self.n_states, self.n_actions))
        
        # 缓存列表，用于存储最近的状态、动作和奖励
        self.states_buffer: List[int] = []
        self.actions_buffer: List[int] = []
        self.rewards_buffer: List[float] = []
    
    def take_action(self, state: int, training: bool = True) -> int:
        """
        基于 epsilon-greedy 策略选择动作
        
        Args:
            state: 当前状态
            training: 是否用于训练
            
        Returns:
            action: 选择的动作
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])
    
    def _compute_n_step_return(self, t: int, T: int) -> float:
        """
        计算从时刻 t 开始的 n 步回报
        
        G_{t:t+n} = Σ_{k=0}^{n-1} γ^k * R_{t+k+1} + γ^n * Q(S_{t+n}, A_{t+n})
        
        Args:
            t: 起始时刻（在 buffer 中的索引）
            T: episode 结束时刻（buffer 长度）
            
        Returns:
            G: n 步回报
        """
        # 计算折扣奖励和
        G = 0.0
        for k in range(min(self.n, T - t)):
            G += (self.gamma ** k) * self.rewards_buffer[t + k]
        
        # 如果还没到终止状态，加上 bootstrap 项
        if t + self.n < T:
            next_state = self.states_buffer[t + self.n]
            next_action = self.actions_buffer[t + self.n]
            G += (self.gamma ** self.n) * self.Q[next_state, next_action]
        
        return G
    
    def update(self):
        """
        使用 n 步回报更新 Q 值
        
        当 buffer 中有足够的步数时，更新最早的状态 - 动作对的 Q 值。
        """
        if len(self.states_buffer) > self.n:
            # 要更新的状态在 buffer 中的索引
            t = 0
            state = self.states_buffer[t]
            action = self.actions_buffer[t]
            
            # 计算 n 步回报
            G = self._compute_n_step_return(t, len(self.states_buffer))
            
            # 更新 Q 值
            td_error = G - self.Q[state, action]
            self.Q[state, action] += self.alpha * td_error
            
            # 移除已更新的记录
            self.states_buffer.pop(0)
            self.actions_buffer.pop(0)
            self.rewards_buffer.pop(0)
    
    def run_episode(self) -> float:
        """
        运行一个完整的 episode
        
        Returns:
            total_reward: 该 episode 的总奖励
        """
        state = self.env.reset()
        
        # 清空缓存
        self.states_buffer = []
        self.actions_buffer = []
        self.rewards_buffer = []
        
        total_reward = 0
        T = float('inf')  # 终止时刻
        
        for t in range(1000):  # 最大步数限制
            if t < T:
                # 选择动作
                action = self.take_action(state, training=True)
                
                # 执行动作
                next_state, reward, done = self.env.step(action)
                
                # 添加到缓存
                self.states_buffer.append(state)
                self.actions_buffer.append(action)
                self.rewards_buffer.append(reward)
                
                total_reward += reward
                
                if done:
                    T = t + 1
            
            # 更新 Q 值（当有足够的步数时）
            if len(self.states_buffer) >= self.n:
                self.update()
            
            if t >= T:
                break
            
            state = next_state
        
        # 处理剩余的缓存（episode 结束后的更新）
        while len(self.states_buffer) >= 1:
            self.update()
        
        return total_reward
    
    def get_policy(self) -> np.ndarray:
        """获取当前策略"""
        return np.argmax(self.Q, axis=1)


class QLearning:
    """
    Q-learning 算法 (异策 TD 控制)
    
    Q-learning 是 Off-policy 算法，它学习的是最优策略的价值函数，
    而行为策略可以是任意策略（通常是 epsilon-greedy）。
    
    更新公式：
    Q(S_t, A_t) ← Q(S_t, A_t) + α * [R_{t+1} + γ * max_a Q(S_{t+1}, a) - Q(S_t, A_t)]
    
    Off-policy 特点：
    - 使用 max 操作来选择下一个状态的最优动作，而不是实际执行的动作
    - 更激进，直接学习最优策略
    - 在悬崖漫步等环境中可能更冒险
    
    与 Sarsa 的关键区别：
    - Sarsa: Q(S, A) ← Q(S, A) + α * [R + γ * Q(S', A') - Q(S, A)]  (使用实际动作 A')
    - Q-learning: Q(S, A) ← Q(S, A) + α * [R + γ * max_a Q(S', a) - Q(S, A)]  (使用最优动作)
    
    Attributes:
        env: 环境实例
        alpha (float): 学习率
        gamma (float): 折扣因子
        epsilon (float): 探索概率
        Q (np.ndarray): 动作价值函数
    """
    
    def __init__(self, env, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1):
        """
        初始化 Q-learning 算法
        
        Args:
            env: 环境实例
            alpha: 学习率
            gamma: 折扣因子
            epsilon: 探索概率
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        
        # 初始化 Q 值为 0
        self.Q = np.zeros((self.n_states, self.n_actions))
    
    def take_action(self, state: int, training: bool = True) -> int:
        """
        基于 epsilon-greedy 策略选择动作（行为策略）
        
        Args:
            state: 当前状态
            training: 是否用于训练
            
        Returns:
            action: 选择的动作
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, done: bool):
        """
        更新 Q 值（Q-learning 核心更新公式）
        
        Q(S, A) ← Q(S, A) + α * [R + γ * max_a Q(S', a) - Q(S, A)]
        
        注意：这里使用 max 操作，而不是实际执行的下一个动作。
        这是 Off-policy 的关键：目标策略是贪婪策略，行为策略是 epsilon-greedy。
        
        Args:
            state: 当前状态 S
            action: 当前动作 A
            reward: 获得的奖励 R
            next_state: 下一个状态 S'
            done: 是否 episode 结束
        """
        if done:
            td_target = reward
        else:
            # Q-learning 使用 max 操作（目标策略是贪婪策略）
            td_target = reward + self.gamma * np.max(self.Q[next_state])
        
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
    
    def run_episode(self) -> float:
        """
        运行一个完整的 episode
        
        Returns:
            total_reward: 该 episode 的总奖励
        """
        state = self.env.reset()
        total_reward = 0
        
        while True:
            action = self.take_action(state, training=True)
            next_state, reward, done = self.env.step(action)
            
            # 更新 Q 值
            self.update(state, action, reward, next_state, done)
            
            total_reward += reward
            
            if done:
                break
            
            state = next_state
        
        return total_reward
    
    def get_policy(self) -> np.ndarray:
        """获取当前策略"""
        return np.argmax(self.Q, axis=1)


class DynaQ:
    """
    Dyna-Q 算法
    
    Dyna-Q 结合了 Q-learning 和基于模型的规划：
    1. 通过与环境交互学习 Q 值和模型
    2. 使用学到的模型进行"模拟"更新（planning）
    
    算法流程：
    1. 执行动作，观察 (s, a, r, s')
    2. 更新 Q(s, a)（Q-learning 更新）
    3. 更新模型 Model(s, a) = (r, s')
    4. 重复 n_planning 次：
       - 随机选择之前见过的 (s, a)
       - 从模型中获取 (r, s')
       - 用 Q-learning 更新 Q(s, a)
    
    Attributes:
        n_planning (int): 每次真实交互后的规划次数
        model (Dict): 环境模型，model[(s,a)] = (reward, next_state)
        visited_pairs (List): 已访问的状态 - 动作对列表
    """
    
    def __init__(self, env, alpha: float = 0.1, gamma: float = 0.99, 
                 epsilon: float = 0.1, n_planning: int = 10):
        """
        初始化 Dyna-Q 算法
        
        Args:
            env: 环境实例
            alpha: 学习率
            gamma: 折扣因子
            epsilon: 探索概率
            n_planning: 每次真实交互后的规划次数
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_planning = n_planning
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        
        # 初始化 Q 值
        self.Q = np.zeros((self.n_states, self.n_actions))
        
        # 环境模型：存储 (state, action) -> (reward, next_state) 的映射
        self.model: Dict[Tuple[int, int], Tuple[float, int]] = {}
        
        # 记录已访问的状态 - 动作对，用于规划阶段采样
        self.visited_pairs: List[Tuple[int, int]] = []
    
    def take_action(self, state: int, training: bool = True) -> int:
        """
        基于 epsilon-greedy 策略选择动作
        
        Args:
            state: 当前状态
            training: 是否用于训练
            
        Returns:
            action: 选择的动作
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state])
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, done: bool):
        """
        更新 Q 值和模型
        
        包括两个部分：
        1. 使用真实经验更新 Q 值（Q-learning 更新）
        2. 更新环境模型
        3. 进行 n_planning 次模拟更新
        
        Args:
            state: 当前状态
            action: 当前动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否 episode 结束
        """
        # 1. Q-learning 更新（使用真实经验）
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[next_state])
        
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
        
        # 2. 更新模型（学习环境的转移）
        if not done:
            self.model[(state, action)] = (reward, next_state)
        else:
            # 终止状态的处理：存储特殊标记
            self.model[(state, action)] = (reward, state)  # 终止后回到自身
        
        # 记录访问过的状态 - 动作对
        if (state, action) not in self.visited_pairs:
            self.visited_pairs.append((state, action))
        
        # 3. 规划阶段：使用模型进行模拟更新
        for _ in range(self.n_planning):
            if len(self.visited_pairs) == 0:
                break
            
            # 随机选择一个之前访问过的状态 - 动作对
            s, a = self.visited_pairs[np.random.randint(len(self.visited_pairs))]
            
            # 从模型中获取模拟的奖励和下一个状态
            r, s_prime = self.model[(s, a)]
            
            # 检查是否是终止状态（s_prime == s 且是终止状态的情况）
            # 这里简化处理：如果模型中存储的是终止转移，则不进行 bootstrap
            if s_prime == s and r == -100:  # 悬崖情况
                simulated_target = r
            else:
                simulated_target = r + self.gamma * np.max(self.Q[s_prime])
            
            # 使用模拟经验更新 Q 值
            self.Q[s, a] += self.alpha * (simulated_target - self.Q[s, a])
    
    def run_episode(self) -> float:
        """
        运行一个完整的 episode
        
        Returns:
            total_reward: 该 episode 的总奖励
        """
        state = self.env.reset()
        total_reward = 0
        
        while True:
            action = self.take_action(state, training=True)
            next_state, reward, done = self.env.step(action)
            
            # 更新 Q 值和模型（包括规划）
            self.update(state, action, reward, next_state, done)
            
            total_reward += reward
            
            if done:
                break
            
            state = next_state
        
        return total_reward
    
    def get_policy(self) -> np.ndarray:
        """获取当前策略"""
        return np.argmax(self.Q, axis=1)


# ==================== 示例用法 ====================
if __name__ == "__main__":
    from cliff_walking_env import CliffWalkingEnvTD
    
    env = CliffWalkingEnvTD()
    
    print("=" * 40)
    print("Sarsa vs Q-learning 对比")
    print("=" * 40)
    
    # 测试 Sarsa (On-policy)
    sarsa = Sarsa(env, alpha=0.1, gamma=0.99, epsilon=0.1)
    for episode in range(500):
        sarsa.run_episode()
    
    print(f"\nSarsa 最终策略:")
    print(sarsa.get_policy())
    
    # 测试 Q-learning (Off-policy)
    env.reset()
    qlearning = QLearning(env, alpha=0.1, gamma=0.99, epsilon=0.1)
    for episode in range(500):
        qlearning.run_episode()
    
    print(f"\nQ-learning 最终策略:")
    print(qlearning.get_policy())
