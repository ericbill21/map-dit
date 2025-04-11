import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import normalize


class MPEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight)
        
    def forward(self, x):
        # Forced weight normalization
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(self.weight))

        # Traditional weight normalization
        w = normalize(self.weight)
        return F.embedding(x, w)