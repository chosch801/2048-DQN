import numpy as np
import random

class Game2048:
    def __init__(self):
        self.size = 4
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.game_over = False
        self.reset()

    def reset(self):
        # 初始化空棋盘
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.game_over = False
        # 初始随机生成两个数字
        self.add_new_tile()
        self.add_new_tile()
        return self.get_state()

    def add_new_tile(self):
        # 找出所有空格子的坐标
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            # 随机挑选一个空格子
            row, col = random.choice(empty_cells)
            # 有 10% 的概率生成 4，90% 的概率生成 2
            self.board[row, col] = 4 if random.random() < 0.1 else 2

    def get_state(self):
        return self.board.copy()

    def slide_and_merge_row(self, row):
        # 取出行中的非 0 项
        new_row = [i for i in row if i != 0]
        score = 0
        merged_row = []
        skip = False
        
        # 遍历数组，执行合并
        for i in range(len(new_row)):
            if skip:
                skip = False
                continue
            # 判断当前元素与其后缀是否相同，相同则合并
            if i < len(new_row) - 1 and new_row[i] == new_row[i + 1]:
                merged_val = new_row[i] * 2
                score += merged_val
                merged_row.append(merged_val)
                skip = True
            else:
                merged_row.append(new_row[i])
                
        # 行末尾补满 0，补偿因合并导致的长度缩减
        while len(merged_row) < self.size:
            merged_row.append(0)
            
        return merged_row, score

    def step(self, action):
        """
        在环境中执行行动: action: 0=上(Up), 1=下(Down), 2=左(Left), 3=右(Right)
        返回: next_state, reward, done
        """
        # 如果游戏结束，直接返回
        if self.game_over:
            return self.get_state(), 0, True

        original_board = self.board.copy()
        total_reward = 0

        # 将棋盘旋转到“向左滑动”的基础状态上，然后进行统一处理
        # 动作 2(左) -> 旋转0次
        # 动作 0(上) -> 向左旋转1次
        # 动作 3(右) -> 向左旋转2次
        # 动作 1(下) -> 向左旋转3次
        rot_k = {2: 0, 0: 1, 3: 2, 1: 3}[action]
        
        # 旋转棋盘
        self.board = np.rot90(self.board, k=rot_k)
        
        for r in range(self.size):
            self.board[r], reward = self.slide_and_merge_row(self.board[r])
            total_reward += reward
            
        # 将棋盘旋转回原本状态
        self.board = np.rot90(self.board, k=-rot_k)

        # 检查棋盘是否有变化（如果没有变化，说明是无效移动）
        if not np.array_equal(original_board, self.board):
            self.add_new_tile()
            self.score += total_reward
        else:
            total_reward = -1 # 对无效移动给予微小的惩罚

        self.game_over = self.check_game_over()
        
        return self.get_state(), total_reward, self.game_over

    def check_game_over(self):
        # 如果存在空格，说明还可继续移动
        if np.any(self.board == 0):
            return False
            
        # 检查上下左右四个方向是否有可以合并的相邻格子
        for r in range(self.size):
            for c in range(self.size):
                if r < self.size - 1 and self.board[r, c] == self.board[r + 1, c]:
                    return False
                if c < self.size - 1 and self.board[r, c] == self.board[r, c + 1]:
                    return False
                    
        return True


    def get_valid_actions(self):
        """
        获取当前状态下的合法动作掩码 (Action Mask)
        返回一个长度为 4 的布尔数组: [上(0), 下(1), 左(2), 右(3)]
        """
        valid_actions = []
        for action_idx in range(4):
            rot_k = {2: 0, 0: 1, 3: 2, 1: 3}[action_idx]
            test_board = np.rot90(self.board, k=rot_k)
            changed = False
            for r in range(self.size):
                merged, _ = self.slide_and_merge_row(test_board[r])
                if not np.array_equal(test_board[r], merged):
                    changed = True
                    break
            valid_actions.append(changed)
        return np.array(valid_actions, dtype=bool)

    def render(self):
        # 简单使用横杠切分并居中呈现可视化棋盘
        print("-" * 25)
        for row in self.board:
            # {val:^5} : 保证数字居中占用 5 宽的空间
            row_str = "|" + "|".join([f"{val:^5}" if val != 0 else "     " for val in row]) + "|"
            print(row_str)
            print("-" * 25)
        print(f"当前得分 (Score): {self.score}")

if __name__ == "__main__":
    game = Game2048()
    
    # 定义动作输入映射规则: W=0:上, S=1:下, A=2:左, D=3:右
    action_dict = {"w": 0, "s": 1, "a": 2, "d": 3}
    
    print("【2048 环境验证用终端界面】")
    print("操作规则: W=上, S=下, A=左, D=右, Q=退出")
    
    while True:
        game.render()
        
        if game.game_over:
            print(f"游戏结束！你的最终得分是: {game.score}")
            break
            
        move = input("请输入移动方向 (W/A/S/D) 或按 Q 退出: ").strip().lower()
        
        if move == 'q':
            print("退出测试。")
            break
            
        if move in action_dict:
            action = action_dict[move]
            next_state, reward, done = game.step(action)
            print(f"\n--> 动作反馈: 奖励值 (Reward): {reward}")
        else:
            print("\n--> 输入无效！请确认按键。")
