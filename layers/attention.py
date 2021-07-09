import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.random.seed(0)

class Attention(nn.Module):
    def __init__(self, rel_size, in_ft):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(rel_size, in_ft, 1))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, emb):
        # emb (rel_size, batch_size, d) weight (rel_size, d, 1)
        emb_att =  torch.bmm(emb, self.weight)  # (rel_size, batch_size, 1)
        alphas = F.softmax(emb_att.squeeze(-1), 0) # (rel_size, batch_size)
        emb_sumamry = (emb * alphas.unsqueeze(-1)).sum(0)
        return emb_sumamry # (batch_size, d)
