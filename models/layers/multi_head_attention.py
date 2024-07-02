import torch.nn as nn
import os
import sys

from scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model):
        super().__init__()
        self.n_heads = n_heads
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        """
        输入的q, k, v的维度为[batch_size, seq_length, d_model]
        实际使用过程中，输入的q、k、v均为原始输入X
        """
        # 1. 获得Q、K、V三个矩阵
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. 将其进行分割为n_heads个头
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. 将q、k、v输入到ScaleDotProductAttention获得多头注意力输出
        x, attention = self.attention(q, k, v)

        # 4. 将多头注意力输出拼接起来并对维度进行修正
        x = self.concat(x)
        x = self.w_concat(x)

        return x
        

        

    def split(self, tensor):
        """
        将[batch_size, seq_length, d_model]的tensor分解成[batch_size, n_heads, seq_length, d_tensor]的tensor
        满足d_tensor * n_heads = d_model
        """
        batch_size, seq_length, d_model = tensor.size()
        n_heads = self.n_heads

        d_tensor = d_model // n_heads
        tensor = tensor.view(batch_size, seq_length, n_heads, d_tensor)
        
        return tensor.transpose(1, 2)



    def concat(self, tensor):
        """
        将[batch_size, n_heads, seq_length, d_tensor]的tensor合并成[batch_size, seq_length, d_model]
        """
        batch_size, n_heads, seq_length, d_tensor = tensor.size()  # tensor.size()和tensor.shape完全相同，没有性能上的差异，只是写法不同
        d_model = n_heads * d_tensor

        tensor = tensor.transpose(1, 2)
        tensor = tensor.view(batch_size, seq_length, d_model).continguous()


