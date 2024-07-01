import torch.nn as nn
import numpy as np

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax()

    def forward(self, q, k, v, mask=None):
        """
        q、k、v皆为[batch, n_heads, seq_length, d_model]维的向量
        """
        d = v.shape[-1]
        attention = q @ k.transpose(2, 3) / np.sqrt(d)
        attention = self.softmax(attention)
        if mask:
            attention = attention.masked_fill(mask, -1e5)

        v = attention @ v
        return x, attention

        