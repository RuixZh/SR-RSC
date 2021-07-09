import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.random.seed(0)

class GCN(nn.Module):
    def __init__(self, rel_size, in_ft, out_ft, act=nn.PReLU(), drop_prob=0.5, isBias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(rel_size, in_ft, out_ft))
        nn.init.xavier_uniform_(self.weight)

        if isBias:
            self.bias = nn.Parameter(torch.empty(rel_size, out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        self.act = act
        self.drop_prob = drop_prob
        self.isBias = isBias

    def forward(self, emb):
        # emb (rel_size, batch_size, ft) weight (rel_size, ft, d)
        e_ = F.dropout(emb, self.drop_prob, training=self.training)
        e = torch.bmm(e_, self.weight)  #  (rel_size, batch_size, d)
        if self.isBias:
            e += self.bias.unsqueeze(1)
        e_out = self.act(e)
        return torch.transpose(e_out, 0, 1)
