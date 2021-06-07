import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.random.seed(0)

class Discriminator(nn.Module):
    def __init__(self, v_ft, u_ft):
        super().__init__()
        self.disc = nn.Linear(v_ft, u_ft, bias=False)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, v_h, u_h, act=nn.Sigmoid(), bias=None):
        sc_ = self.disc(v_h)
        sc = torch.mm(sc_, u_h.t())
        if bias is not None:
            sc += bias
        logits = act(sc)
        return logits
