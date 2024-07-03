import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, max_length, d_model, device=torch.device('cpu')):
        self.encoding = torch.zeros(max_length, d_model, device=device)

        pos = torch.arange(0, max_length)
        pos = pos.float().unsqueeze(1)

        _2i = torch.arange(0, d_model, 2)

        self.encoding[:, 0::2] = torch.sin(pos / 10000 ** (_2i/d_model))
        self.encoding[:, 0::2] = torch.cos(pos / 10000 ** (_2i/d_model))

    def forward(self, x):
        """
        这里的x是输入的token（或者token_id），维度为[batch_size, seq_length]
        """
        batch_size, seq_length = x.shape
        return self.encoding[:seq_length]