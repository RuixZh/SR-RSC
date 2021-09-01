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
    def __init__(self, in_ft, out_ft, act=nn.Identity(), drop_prob=0.5, isBias=False):
        super(GCN, self).__init__()
         self.weight = nn.Parameter(torch.empty(in_ft, out_ft))
         nn.init.xavier_uniform_(self.weight)
        if isBias:
            self.bias = nn.Parameter(torch.empty(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)
        self.dropout = dropout
        self.act = nn.LeakyReLU(alpha)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return self.act(output)


class DGCN(nn.Module):
    def __init__(self, v_in_ft, u_in_ft, out_ft, act=nn.ReLU(), drop_prob=0.5, isBias=False, leaky=0.1):
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

        self.v_gc1 = GCN(nfeat=v_in_ft,
                        nhid=out_ft,
                        dropout=drop_prob,
                        alpha=leaky)
        self.v_gc2 = GCN(nfeat=out_ft,
                        nhid=out_ft,
                        dropout=drop_prob,
                        alpha=leaky)

        self.u_gc1 = GCN(nfeat=u_in_ft,
                        nhid=out_ft,
                        dropout=drop_prob,
                        alpha=leaky)
        self.u_gc2 = GCN(nfeat=out_ft,
                        nhid=out_ft,
                        dropout=drop_prob,
                        alpha=leaky)
        self.u_fc = nn.Linear(out_ft + u_in_ft, out_ft)
        nn.init.xavier_uniform_(self.u_fc.weight.data)
        self.v_fc = nn.Linear(out_ft + v_in_ft, out_ft)
        nn.init.xavier_uniform_(self.v_fc.weight.data)

        self.act = act
        self.drop_prob = drop_prob
        self.isBias = isBias

    def forward(self, uv_adj, vu_adj, ufea, vfea):
        # emb (batch_size, ft)
        u = F.dropout(ufea, self.drop_prob, training=self.training)
        v = F.dropout(vfea, self.drop_prob, training=self.training)

        vu = self.u_gc1(vu_adj, ufea)
        uv = self.v_gc1(uv_adj, vfea)

        uv2 = self.v_gc2(uv_adj, vu)
        vu2 = self.u_gc2(vu_adj, uv)

        Hv = torch.cat((vu2, vfea), dim=1)
        Hu = torch.cat((uv2, ufea), dim=1)

        Hv = self.v_fc(Hv)  #  (batch_size, d)
        Hu = self.u_fc(Hu)  #  (batch_size, d)

        return self.act(Hu), self.act(Hv)
