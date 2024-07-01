import torch.nn as nn
import torch

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super().__init__()