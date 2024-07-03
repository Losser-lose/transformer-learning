from positional_encoding import PositionalEncoding
from token_embedding import TokenEmbedding
import torch.nn as nn


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_length, drop_prob):
        super().__init__()
        self.tok_emb = TokenEmbedding()
        self.pos_emb = PositionalEncoding()
        self.drop_out = nn.Dropout(drop_prob)
    
    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb+pos_emb)