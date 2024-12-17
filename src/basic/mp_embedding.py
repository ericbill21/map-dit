import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import normalize


class MPEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, use_wn: bool, use_forced_wn: bool):
        super().__init__()

        self.use_wn = use_wn
        self.use_forced_wn = use_forced_wn

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

        if use_wn:
            nn.init.normal_(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight)

    def forward(self, x):
        # Forced weight normalization
        if self.training and self.use_forced_wn:
            with torch.no_grad():
                self.weight.copy_(normalize(self.weight))

        # Traditional weight normalization
        w = self.weight
        if self.use_wn:
            w = normalize(w)

        return F.embedding(x, w)
