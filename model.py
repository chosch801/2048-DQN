import torch
import torch.nn as nn

class TransformerDQN(nn.Module):
    def __init__(self, action_dim=4, board_size=16, d_model=64, nhead=2, num_layers=2):
        super(TransformerDQN, self).__init__()
        
        self.action_dim = action_dim
        self.board_size = board_size
        
        # ==========================================
        # 1. 状态编码层 (Input Encoding)
        # ==========================================
        # 设置 num_embeddings=16 以覆盖对数处理后最大至 2^15 左右的数字规模
        self.val_embedding = nn.Embedding(num_embeddings=16, embedding_dim=d_model)
        
        # 2D Positional Encoding: 分别给行和列进行空间定位
        # 行列分离的 2D 绝对位置编码
        self.row_embed = nn.Embedding(4, d_model // 2)
        self.col_embed = nn.Embedding(4, d_model // 2)
        
        # ==========================================
        # 2. 特征提取主干 (Transformer Backbone)
        # ==========================================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 设定池化特征维度
        pool_dim = d_model
        
        # ==========================================
        # 3. 分支 A: DQN 决策头 (Q-Learning)
        # ==========================================
        self.q_head = nn.Sequential(
            nn.Linear(pool_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        # ==========================================
        # 4. 分支 B: 自监督预测头 (SSL Next-State)
        # ==========================================
        # 预测所有 action 产生状态（Next-State）
        # 预测尺寸等于: action_dim(4) * board_size(16) = 64
        self.ssl_head = nn.Sequential(
            nn.Linear(pool_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, action_dim * board_size)
        )

    def forward(self, x):
        """
        前向传播函数
        x: [batch, 16] - log2 处理后的 16 个格子的数组
        """
        # 转换类型适配 Embedding Lookup
        x = x.long()
        batch_size = x.size(0)
        
        # 
        # 1. 抽取数值 Embedding
        val_emb = self.val_embedding(x)
        
        # 2. 生成固定的 2D 坐标位置索引
        device = x.device
        row_indices = torch.arange(4, device=device).repeat_interleave(4).unsqueeze(0).expand(batch_size, -1)
        col_indices = torch.arange(4, device=device).repeat(4).unsqueeze(0).expand(batch_size, -1)
        
        r_emb = self.row_embed(row_indices) # [batch, 16, d_model // 2]
        c_emb = self.col_embed(col_indices) # [batch, 16, d_model // 2]
        pos_emb = torch.cat([r_emb, c_emb], dim=-1) # 拼接二维坐标特征
        
        # 合并特征与位置编码: (batch_size, 16, d_model)
        seq = val_emb + pos_emb
        
        # Transformer Encoder 获取全局注意力特征: (batch_size, 16, d_model)
        out = self.transformer(seq)
        
        # 应用 Global Average Pooling 降维提取核心表征
        features = out.mean(dim=1)
        
        # 分支输出预测结果
        # 1. DQN Q值
        q_values = self.q_head(features)
        
        # 2. SSL 分支预测
        ssl_preds = self.ssl_head(features).reshape(-1, self.action_dim, self.board_size)
        
        return q_values, ssl_preds

if __name__ == "__main__":
    print("=== PyTorch 网络结构形态编译测试 ===")
    
    # 实例化
    model = TransformerDQN()
    
    # 统计实际占用的参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"-> 模型构建成功！参数总量: {total_params} 个。 ")
    
    # 模拟从 env 传入的一个“已处理”的数据 batch (假设同时推断两个棋盘，batch_size=2)
    dummy_x = torch.tensor([
        [0, 0, 1, 0, 3, 2, 0, 0, 4, 0, 0, 0, 0, 0, 1, 0], 
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0]  
    ], dtype=torch.float32)
    
    q_vals, ssl_fut_states = model(dummy_x)
    
    print(f"\\n--> [输入层]: 棋盘张量尺寸: {dummy_x.shape}")
    print(f"--> [输出 A]: DQN Q-Values: {q_vals.shape} -> 代表 batch 中的 2 个状态下，各自 4 个动作的分数预期。")
    print(f"--> [输出 B]: SSL 预测未来状态: {ssl_fut_states.shape} -> 代表对 batch 中 2 个状态，预演了各自4个动作后，新产生的16个格子的变化期望。")

