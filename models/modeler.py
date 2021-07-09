import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.random.seed(0)
from layers import *

class modeler(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.bnn1 = nn.ModuleList([GCN(args.nb_rel,
                                       args.ft_size,
                                       args.out_ft,
                                       nn.PReLU(),
                                       args.drop_prob,
                                       args.isBias)])
        self.att = nn.ModuleList([Attention(args.ft_size,args.out_ft) for _ in range(args.hopK)])
        self.bnn2 = GCN2(args.out_ft + self.args.ft_size,
                         args.out_ft,
                         nn.Identity(),
                         args.drop_prob,
                         args.isBias)
        for k in range(1, args.hopK):
            self.bnn1.append(GCN(args.nb_rel,
                                 args.out_ft,
                                 args.out_ft,
                                 nn.PReLU(),
                                 args.drop_prob,
                                 args.isBias))
        self.marginloss = nn.MarginRankingLoss(0.5)

    def forward(self, node_list, graph, features, k, act=nn.Sigmoid(), isConcat=False, isAtt=False):
        fbatch = []
        for n in node_list:
            rel_neighbors = graph[n]
            v = []
            for neighbors in rel_neighbors:
                v.append(torch.mean(features[neighbors],0))
            fbatch.append(torch.vstack(v).unsqueeze(1))
        v_in = torch.cat(fbatch, 1)  # (rel_size, batch_size, ft)
        v_out = self.bnn1[k](v)  # (batch_size, rel_size, d)

        if isAtt:
            v_out = self.att[k](v_out)
        else:
            v_out = torch.mean(v_out, 1)  # (batch_size, d)

        if isConcat:
            v = torch.hstack((v_out, features[node_list+1]))
            v_out = self.bnn2(v)

        return v_out

    def loss(self, embs, graph):
        sub_h = []
        for i, n in enumerate(graph):
            vn = torch.hstack(n+[torch.LongTensor([1+i]).to(self.args.device)])
            v_avg = torch.mean(embs[vn[vn>0]],0)
            sub_h.append(v_avg)
        h = torch.vstach(sub_h)
        shuf_idx = torch.randperm(self.args.nb_node)
        h2 = h[shuf_idx]
        embs = embs[1:]
        embs2 = embs[shuf_idx]
        logits_pos = torch.sigmoid(torch.sum(embs * h, dim=-1))
        logits_neg = torch.sigmoid(torch.sum(embs * h2, dim=-1))
        logits_pos2 = torch.sigmoid(torch.sum(embs2 * h2, dim=-1))
        logits_neg2 = torch.sigmoid(torch.sum(embs2 * h, dim=-1))
        totalLoss = 0.0
        ones = torch.ones(logits_pos.size(0)).to(self.args.device)
        totalLoss += self.marginloss(logits_pos, logits_neg2, ones)
        totalLoss += self.marginloss(logits_pos2, logits_neg, ones)
        return totalLoss
