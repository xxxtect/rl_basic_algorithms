"""
悬崖漫步（Cliff Walking）环境实现

本模块包含两个环境类：
1. CliffWalkingEnvDP: 包含完整转移矩阵 P 的静态环境（用于动态规划）
2. CliffWalkingEnvTD: 包含实时 step() 和 reset() 交互逻辑的动态环境（用于时序差分）

环境说明：
- 网格世界：4 行 x 12 列
- 起点：左下角 (3, 0)
- 终点：右下角 (3, 11)
- 悬崖：底部一行 (3, 1) 到 (3, 10)
- 动作：上、下、左、右（0, 1, 2, 3）
- 奖励：每步 -1，掉下悬崖 -100
"""

import numpy as np
from typing import Dict, Tuple, Optional

# 动作常量定义
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# 网格尺寸
GRID_ROWS = 4
GRID_COLS = 12


class CliffWalkingEnvDP:
    """
    悬崖漫步环境（动态规划版本）
    
    提供完整的 MDP 描述，包括状态转移概率矩阵 P 和奖励 R。
    适用于需要完整环境模型的动态规划算法（如策略迭代、价值迭代）。
    
    Attributes:
        n_states (int): 状态总数（48 个）
        n_actions (int): 动作总数（4 个）
        P (Dict): 状态转移概率字典 P[s][a] = [(prob, next_state, reward, done), ...]
        start_state (int): 起始状态索引
    """
    
    def __init__(self):
        """
        初始化悬崖漫步环境（DP 版本）
        
        构建完整的状态转移矩阵和奖励函数。
        """
        self.n_states = GRID_ROWS * GRID_COLS  # 48 个状态
        self.n_actions = 4  # 4 个动作：上、下、左、右
        
        # 起始状态：左下角 (3, 0)，状态索引 = 3 * 12 + 0 = 36
        self.start_state = (GRID_ROWS - 1) * GRID_COLS
        
        # 终止状态：右下角 (3, 11)，状态索引 = 3 * 12 + 11 = 47
        self.goal_state = (GRID_ROWS - 1) * GRID_COLS + (GRID_COLS - 1)
        
        # 悬崖区域：第 4 行（索引 3）的第 2 列到第 11 列（索引 1 到 10）
        self.cliff_states = list(range((GRID_ROWS - 1) * GRID_COLS + 1, 
                                        (GRID_ROWS - 1) * GRID_COLS + GRID_COLS - 1))
        
        # 构建状态转移矩阵
        # P[s][a] = [(prob, next_state, reward, done), ...]
        self.P = self._build_transition_matrix()
    
    def _state_to_idx(self, row: int, col: int) -> int:
        """
        将网格坐标转换为状态索引
        
        Args:
            row: 行索引（0 到 3）
            col: 列索引（0 到 11）
            
        Returns:
            状态索引（0 到 47）
        """
        return row * GRID_COLS + col
    
    def _idx_to_state(self, idx: int) -> Tuple[int, int]:
        """
        将状态索引转换为网格坐标
        
        Args:
            idx: 状态索引（0 到 47）
            
        Returns:
            (row, col): 行索引和列索引
        """
        return idx // GRID_COLS, idx % GRID_COLS
    
    def _build_transition_matrix(self) -> Dict:
        """
        构建完整的状态转移概率矩阵
        
        Returns:
            P: 状态转移字典，P[s][a] = [(prob, next_state, reward, done), ...]
        """
        P = {s: {a: [] for a in range(self.n_actions)} for s in range(self.n_states)}
        
        for s in range(self.n_states):
            row, col = self._idx_to_state(s)
            
            # 如果当前状态是悬崖状态，转移到起始状态（episode 结束）
            if s in self.cliff_states:
                for a in range(self.n_actions):
                    P[s][a] = [(1.0, self.start_state, -100, True)]
                continue
            
            # 如果当前状态是终止状态，转移到自身（episode 结束）
            if s == self.goal_state:
                for a in range(self.n_actions):
                    P[s][a] = [(1.0, s, 0, True)]
                continue
            
            # 对于其他状态，计算每个动作的转移
            for a in range(self.n_actions):
                next_row, next_col = self._get_next_position(row, col, a)
                next_s = self._state_to_idx(next_row, next_col)
                
                # 检查是否掉下悬崖
                if next_s in self.cliff_states:
                    # 掉下悬崖：奖励 -100，回到起始状态，episode 结束
                    P[s][a] = [(1.0, self.start_state, -100, True)]
                elif next_s == self.goal_state:
                    # 到达终点：奖励 -1，episode 结束
                    P[s][a] = [(1.0, next_s, -1, True)]
                else:
                    # 普通移动：奖励 -1，episode 继续
                    P[s][a] = [(1.0, next_s, -1, False)]
        
        return P
    
    def _get_next_position(self, row: int, col: int, action: int) -> Tuple[int, int]:
        """
        根据动作计算下一个位置（考虑边界）
        
        Args:
            row: 当前行索引
            col: 当前列索引
            action: 动作（0: 上，1: 下，2: 左，3: 右）
            
        Returns:
            (next_row, next_col): 下一个位置的坐标
        """
        if action == UP:
            next_row = max(0, row - 1)
            next_col = col
        elif action == DOWN:
            next_row = min(GRID_ROWS - 1, row + 1)
            next_col = col
        elif action == LEFT:
            next_row = row
            next_col = max(0, col - 1)
        elif action == RIGHT:
            next_row = row
            next_col = min(GRID_COLS - 1, col + 1)
        else:
            raise ValueError(f"Invalid action: {action}")
        
        return next_row, next_col
    
    def get_transition_prob(self, state: int, action: int) -> list:
        """
        获取指定状态和动作下的转移概率列表
        
        Args:
            state: 当前状态索引
            action: 动作索引
            
        Returns:
            [(prob, next_state, reward, done), ...] 列表
        """
        return self.P[state][action]
    
    def render_policy(self, policy: np.ndarray):
        """
        可视化策略（打印在网格上）
        
        Args:
            policy: 策略数组，policy[s] = action
        """
        symbols = ['^', 'v', '<', '>']  # 上、下、左、右的符号表示
        
        print("策略可视化 (^=上，v=下，<=左，>=右):")
        print("-" * (GRID_COLS * 3 + 1))
        
        for row in range(GRID_ROWS):
            line = "|"
            for col in range(GRID_COLS):
                s = self._state_to_idx(row, col)
                if s == self.goal_state:
                    line += " G|"  # 终点
                elif s in self.cliff_states:
                    line += " C|"  # 悬崖
                elif s == self.start_state:
                    line += " S|"  # 起点
                else:
                    action = policy[s]
                    line += f" {symbols[action]}|"
            print(line)
            print("-" * (GRID_COLS * 3 + 1))


class CliffWalkingEnvTD:
    """
    悬崖漫步环境（时序差分版本）
    
    提供 step() 和 reset() 接口，用于与环境进行实时交互。
    适用于不需要完整环境模型的时序差分算法（如 Q-learning、SARSA）。
    
    Attributes:
        state (int): 当前状态索引
        n_states (int): 状态总数
        n_actions (int): 动作总数
    """
    
    def __init__(self):
        """
        初始化悬崖漫步环境（TD 版本）
        """
        self.n_states = GRID_ROWS * GRID_COLS  # 48 个状态
        self.n_actions = 4  # 4 个动作
        
        # 起始状态：左下角 (3, 0)
        self.start_state = (GRID_ROWS - 1) * GRID_COLS
        
        # 终止状态：右下角 (3, 11)
        self.goal_state = (GRID_ROWS - 1) * GRID_COLS + (GRID_COLS - 1)
        
        # 悬崖区域：第 4 行（索引 3）的第 2 列到第 11 列（索引 1 到 10）
        self.cliff_states = list(range((GRID_ROWS - 1) * GRID_COLS + 1, 
                                        (GRID_ROWS - 1) * GRID_COLS + GRID_COLS - 1))
        
        # 当前状态
        self.state = self.start_state
    
    def reset(self) -> int:
        """
        重置环境到初始状态
        
        Returns:
            起始状态索引
        """
        self.state = self.start_state
        return self.state
    
    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        执行一个动作，返回下一个状态、奖励和是否结束
        
        Args:
            action: 动作索引（0: 上，1: 下，2: 左，3: 右）
            
        Returns:
            next_state: 下一个状态索引
            reward: 获得的奖励
            done: 是否 episode 结束
            
        Raises:
            ValueError: 当 action 超出有效范围时
        """
        if action < 0 or action >= self.n_actions:
            raise ValueError(f"Action {action} out of range [0, {self.n_actions})")
        
        row, col = self._idx_to_state(self.state)
        next_row, next_col = self._get_next_position(row, col, action)
        next_state = self._state_to_idx(next_row, next_col)
        
        # 默认奖励为 -1（每移动一步）
        reward = -1
        done = False
        
        # 检查是否掉下悬崖
        if next_state in self.cliff_states:
            reward = -100
            done = True
            self.state = self.start_state  # 回到起始状态
            next_state = self.start_state
        # 检查是否到达终点
        elif next_state == self.goal_state:
            done = True
            self.state = next_state
        else:
            self.state = next_state
        
        return next_state, reward, done
    
    def _state_to_idx(self, row: int, col: int) -> int:
        """
        将网格坐标转换为状态索引
        
        Args:
            row: 行索引（0 到 3）
            col: 列索引（0 到 11）
            
        Returns:
            状态索引（0 到 47）
        """
        return row * GRID_COLS + col
    
    def _idx_to_state(self, idx: int) -> Tuple[int, int]:
        """
        将状态索引转换为网格坐标
        
        Args:
            idx: 状态索引（0 到 47）
            
        Returns:
            (row, col): 行索引和列索引
        """
        return idx // GRID_COLS, idx % GRID_COLS
    
    def _get_next_position(self, row: int, col: int, action: int) -> Tuple[int, int]:
        """
        根据动作计算下一个位置（考虑边界）
        
        Args:
            row: 当前行索引
            col: 当前列索引
            action: 动作（0: 上，1: 下，2: 左，3: 右）
            
        Returns:
            (next_row, next_col): 下一个位置的坐标
        """
        if action == UP:
            next_row = max(0, row - 1)
            next_col = col
        elif action == DOWN:
            next_row = min(GRID_ROWS - 1, row + 1)
            next_col = col
        elif action == LEFT:
            next_row = row
            next_col = max(0, col - 1)
        elif action == RIGHT:
            next_row = row
            next_col = min(GRID_COLS - 1, col + 1)
        else:
            raise ValueError(f"Invalid action: {action}")
        
        return next_row, next_col
    
    def render(self, state: Optional[int] = None):
        """
        可视化当前环境状态
        
        Args:
            state: 要显示的状态索引，如果为 None 则显示当前状态
        """
        if state is None:
            state = self.state
        
        row, col = self._idx_to_state(state)
        
        print("当前环境状态:")
        print("-" * (GRID_COLS * 3 + 1))
        
        for r in range(GRID_ROWS):
            line = "|"
            for c in range(GRID_COLS):
                s = self._state_to_idx(r, c)
                if s == self.goal_state:
                    line += " G|"  # 终点
                elif s in self.cliff_states:
                    line += " C|"  # 悬崖
                elif s == self.start_state:
                    line += " S|"  # 起点
                elif s == state:
                    line += " *|"  # 智能体当前位置
                else:
                    line += "  |"
            print(line)
            print("-" * (GRID_COLS * 3 + 1))


# ==================== 示例用法 ====================
if __name__ == "__main__":
    # 测试 DP 版本环境
    print("=" * 40)
    print("测试 CliffWalkingEnvDP（动态规划版本）")
    print("=" * 40)
    
    env_dp = CliffWalkingEnvDP()
    print(f"状态总数：{env_dp.n_states}")
    print(f"动作总数：{env_dp.n_actions}")
    print(f"起始状态：{env_dp.start_state}")
    print(f"终止状态：{env_dp.goal_state}")
    print(f"悬崖状态：{env_dp.cliff_states}")
    
    # 测试转移概率
    state = 36  # 起始状态
    action = RIGHT
    transitions = env_dp.get_transition_prob(state, action)
    print(f"\n状态 {state} 执行动作 {action} (右) 的转移:")
    print(transitions)
    
    # 测试 TD 版本环境
    print("\n" + "=" * 40)
    print("测试 CliffWalkingEnvTD（时序差分版本）")
    print("=" * 40)
    
    env_td = CliffWalkingEnvTD()
    state = env_td.reset()
    print(f"重置后状态：{state}")
    env_td.render()
    
    # 模拟几步
    print("\n模拟向右移动 5 步:")
    for i in range(5):
        next_state, reward, done = env_td.step(RIGHT)
        print(f"步 {i+1}: 状态={next_state}, 奖励={reward}, 结束={done}")
        env_td.render()
        if done:
            break
