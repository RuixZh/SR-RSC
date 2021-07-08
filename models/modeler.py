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
        self.bnn1 = nn.ModuleList([BiNN(,
                                       args.out_ft,
                                       nn.PReLU(),
                                       args.drop_prob,
                                       args.isBias)])
        self.bnn2 = BiNN(args.out_ft + self.args.ft_size,
                         args.out_ft,
                         nn.Identity(),
                         args.drop_prob,
                         args.isBias)
        for k in range(1, args.hopK):
            self.bnn1.append(BiNN(,
                            args.out_ft,
                            nn.PReLU(),
                            args.drop_prob,
                            args.isBias))


    def forward(self, node_list, graph, features, k, act=nn.Sigmoid(), isConcat=False, isAtt=False):
        fbatch = []
        for n in node_list:
            rel_neighbors = graph[n]
            v = []
            for neighbors in rel_neighbors:
                v.append(torch.mean(features[neighbors],0))
            fbatch.append(torch.vstack(v).unsqueeze(0))
        v_in = torch.vstack(fbatch)  # (batch_size, rel_size, ft)
        v_out = self.bnn1[k](v)  # (batch_size, rel_size, d)

        if isAtt:
            v_out = self.att[k](v_out)
        else:
            v_out = torch.mean(v_out, 1)  # (batch_size, d)
        if isConcat:
            v = torch.hstack((v_out, features[node_list+1]))
            v_out = self.bnn2(v)

        return v_out

    def loss(self,)
