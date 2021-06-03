import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BiNN(nn.Module):
    """
        GNN Module layer
    """
    def __init__(self, rel, rel_fea, hid_units, out_ft, act='prelu', drop_prob=0.5, isBias=False):
        super().__init__()
        v, u = rel.split("-")
        self.fc_v = nn.Linear(rel_fea[v][rel].shape[1], hid_units)
        self.fc_u = nn.Linear(rel_fea[u][rel].shape[1], hid_units)
        self.fc_v2 = nn.Linear(hid_units+rel_fea[v]['original'].shape[1], out_ft)
        self.fc_u2 = nn.Linear(hid_units+rel_fea[v]['original'].shape[1], out_ft)

        if act == 'prelu':
            self.act = nn.PReLU()
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'leakyrelu':
            self.act = nn.LeakyReLU()
        elif act == 'relu6':
            self.act = nn.ReLU6()
        elif act == 'rrelu':
            self.act = nn.RReLU()
        elif act == 'selu':
            self.act = nn.SELU()
        elif act == 'celu':
            self.act = nn.CELU()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'identity':
            self.act = nn.Identity()

        if isBias:
            self.bias_1 = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias_1.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

        self.drop_prob = drop_prob
        self.isBias = isBias

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, ):
        seq = F.dropout(seq, self.drop_prob, training=self.training)
        seq = self.fc_1(seq)
        if sparse:
            seq = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq, 0)), 0)
        else:
            seq = torch.bmm(adj, seq)

        if self.isBias:
            seq += self.bias_1

        return self.act(seq)
