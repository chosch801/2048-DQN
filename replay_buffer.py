import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity=100000, state_dim=16, action_dim=4):
        # 经验回放池最大容量配置
        self.capacity = capacity
        # 循环指针
        self.ptr = 0
        # 实际条数
        self.size = 0
        
        # 避免临时申请列表引起的碎片化，在初始化时就申请大块连续内存
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, 1), dtype=np.int64)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        
        # 为当前状态、下一个状态专门保存动作掩码
        self.mask = np.zeros((capacity, action_dim), dtype=bool)
        self.next_mask = np.zeros((capacity, action_dim), dtype=bool)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def save(self, filepath):
        """序列化并保存当前的回放池缓冲数据"""
        np.savez_compressed(filepath, 
                            state=self.state[:self.size], 
                            action=self.action[:self.size],
                            reward=self.reward[:self.size],
                            next_state=self.next_state[:self.size],
                            done=self.done[:self.size],
                            mask=self.mask[:self.size],
                            next_mask=self.next_mask[:self.size],
                            ptr=self.ptr,
                            size=self.size)
        print(f"--> [ReplayBuffer] 已将 {self.size} 条经验序列化保存至 {filepath}")

    def load(self, filepath):
        """加载并反序列化指定路径的回放池缓冲数据"""
        try:
            data = np.load(filepath)
            self.size = int(data['size'])
            self.ptr = int(data['ptr'])
            
            # 使用 numpy 切片赋值，避免重新申请内存引发 OOM
            self.state[:self.size] = data['state']
            self.action[:self.size] = data['action']
            self.reward[:self.size] = data['reward']
            self.next_state[:self.size] = data['next_state']
            self.done[:self.size] = data['done']
            self.mask[:self.size] = data['mask']
            self.next_mask[:self.size] = data['next_mask']
            
            print(f"--> [ReplayBuffer] 成功恢复了 {self.size} 条数据记录。")
        except Exception as e:
            print(f"--> [ReplayBuffer] 读取存档失败 ({e})，将以空经验池开局。")

    def push(self, state, action, reward, next_state, done, mask, next_mask):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        self.mask[self.ptr] = mask
        self.next_mask[self.ptr] = next_mask
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        # 固定为均匀随机采样，保障训练过程中 Q 价值的平滑拟合
        ind = np.random.randint(0, self.size, size=batch_size)
        
        # 返回并直接分配到 GPU 显存上
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device),
            torch.BoolTensor(self.mask[ind]).to(self.device),
            torch.BoolTensor(self.next_mask[ind]).to(self.device)
        )

    def __len__(self):
        return self.size
