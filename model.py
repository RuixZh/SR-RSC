import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.random.seed(0)
import pickle as pkl
from embedder import embedder
from layers import *

class BiHIN(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args

    def training(self):
        model = nn.Sequential(
            GCN(),
            GCN()
        )
        optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.l2_coef)
        cnt_wait = 0; best = 1e9
        for epoch in range(self.args.nb_epochs):



            # for bs in range(len(nodes) // self.args.batch_size+1):
            #     node_batch = nodes[bs * self.args.batch_size: (bs+1) * self.args.batch_size]
            #     neighbor_batch = [neighbors[n] for n in node_batch]
            #     neighbor_fea = [torch.mean([fea.to(self.args.device) for fea in self.features[nei]], 0) for nei in neighbor_batch]
            #     node_fea = [fea.to(self.args.device) for fea in self.features[node_batch]]

            if loss < best:
                best = loss
                cnt_wait = 0
                torch.save(model.state_dict(), 'saved_model/best_{}_{}.pkl'.format(self.args.dataset, self.args.embedder))
            else:
                cnt_wait += 1

            if cnt_wait == self.args.patience:
                break

            loss.backward()
            optimiser.step()


class modeler(nn.Module):
    def __init__(self, args, rel_fea):
        super().__init__()
        self.args = args
        self.gnn = nn.ModuleList([BiNN(rel, rel_fea, args.hid_units, args.out_ft, args.activation, args.drop_prob, args.isBias) for rel in self.args.bi_graphs])


    # def weights_init(self, m):
    #     if isinstance(m, nn.Linear):
    #         torch.nn.init.xavier_uniform_(m.weight.data)
    #         if m.bias is not None:
    #             m.bias.data.fill_(0.0)

    def forward(self, ):
        for rel in self.args.bi_graphs:
            st, tt = rel.split("-")
            self.rel_fea[st][rel]
            self.rel_fea[tt][rel]
