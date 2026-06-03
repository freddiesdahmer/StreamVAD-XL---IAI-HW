import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

# --------------------------
# Transformer-XL 记忆缓存块
# --------------------------
class TransformerXLMemoryBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: Optional[torch.Tensor] = None, mem_len: int = 32):
        if memory is not None:
            detached_mem = memory.detach()
            kv = torch.cat([detached_mem, x], dim=1)
        else:
            kv = x

        attn_output, _ = self.self_attn(query=x, key=kv, value=kv)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        ffn_output = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)

        if memory is None:
            next_memory = x
        else:
            next_memory = torch.cat([memory, x], dim=1)[:, -mem_len:, :]

        return x, next_memory

# --------------------------
# 增强版PCI模块（融合Transformer-XL）
# --------------------------
class EnhancedPCI(nn.Module):
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256, nhead: int = 4, 
                 mem_len: int = 32, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.mem_len = mem_len
        
        self.layers = nn.ModuleList([
            TransformerXLMemoryBlock(hidden_dim, nhead, dim_feedforward=hidden_dim*4)
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x: torch.Tensor, memories: Optional[List[torch.Tensor]] = None):
        B, T, D = x.shape
        x = self.input_proj(x)

        if memories is None:
            memories = [None] * len(self.layers)
        
        new_memories = []
        for i, layer in enumerate(self.layers):
            x, next_mem = layer(x, memories[i], self.mem_len)
            new_memories.append(next_mem)

        x = self.output_proj(x)
        return x, new_memories

# --------------------------
# 完整StreamVAD+XL模型
# --------------------------
class StreamVADWithXL(nn.Module):
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256, 
                 mem_len: int = 32, num_layers: int = 2, nhead: int = 4):
        super().__init__()
        # 关键帧生成器（保留原始设计）
        self.kcg = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        
        # 增强版时序建模
        self.pci = EnhancedPCI(feature_dim, hidden_dim, nhead, mem_len, num_layers)

        self.feature_norm = nn.LayerNorm(feature_dim)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x, memories=None):
        # 外部加 sigmoid，保证 kcg 权重仍在 0~1
        kcg_weights = torch.sigmoid(self.kcg(x)).squeeze(-1)
        weighted_x = x * kcg_weights.unsqueeze(-1)
        
        enhanced_x, new_memories = self.pci(weighted_x, memories)
        
        # 新增：归一化，防止数值爆炸
        enhanced_x = self.feature_norm(enhanced_x)
        
        # 返回 logits（原始分数），不要 sigmoid
        frame_logits = self.classifier(enhanced_x).squeeze(-1)
        
        return frame_logits, new_memories, kcg_weights