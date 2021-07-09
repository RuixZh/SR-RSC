import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.random.seed(0)

class GCN2(nn.Module):
    def __init__(self, in_ft, out_ft, act=nn.Identity(), drop_prob=0.5, isBias=False):
        super().__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        nn.init.xavier_uniform_(self.fc.weight.data)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0.0)

        if isBias:
            self.bias = nn.Parameter(torch.empty(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        self.act = act
        self.drop_prob = drop_prob
        self.isBias = isBias

    def forward(self, emb):
        # emb (batch_size, ft)
        e_ = F.dropout(emb, self.drop_prob, training=self.training)
        e = self.fc(e_)  #  (batch_size, d)
        if self.isBias:
            e += self.bias
        return self.act(e)
