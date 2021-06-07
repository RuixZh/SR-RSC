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
    def __init__(self, out_ft, nb_graphs):
        super().__init__()
        self.nb_graphs = nb_graphs
        self.A = nn.ModuleList([nn.Linear(out_ft, 1) for _ in range(nb_graphs)])
        self.weight_init()

    def weight_init(self):
        for i in range(self.nb_graphs):
            nn.init.xavier_normal_(self.A[i].weight)
            self.A[i].bias.data.fill_(0.0)

    def forward(self, features):
        features_attn = []
        for i in range(self.nb_graphs):
            features_attn.append(self.A[i](features[i]))
        features_attn = F.softmax(torch.cat(features_attn, 1), -1)
        features = torch.stack(features,1)
        features = (features * features_attn.unsqueeze(-1)).sum(1)
        return features
