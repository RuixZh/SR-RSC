import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.random.seed(0)

class BiNN(nn.Module):
    def __init__(self, in_ft, out_ft, act=nn.PReLU(), drop_prob=0.5, isBias=False):
        super().__init__()
        self.fc_1 = nn.Linear(in_ft, out_ft, bias=False)

        if isBias:
            self.bias_1 = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias_1.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

        self.act = act
        self.drop_prob = drop_prob
        self.isBias = isBias

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, emb):
        e_ = F.dropout(emb, self.drop_prob, training=self.training)
        e = self.fc_1(e_)
        return self.act(e)
