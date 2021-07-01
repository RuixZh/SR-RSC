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
        self.bnn1 = nn.ModuleList()
        self.bnn2 = nn.ModuleList()
        self.bnn = nn.ModuleList()

        for rel in self.args.bi_graphs:
            v, u = rel.split('-')
            if v == u:
                self.bnn1.append(BiNN(args.ft_size[v], args.hid_units, nn.PReLU(), args.drop_prob, args.isBias))
                self.bnn2.append(BiNN(args.hid_units, args.out_ft, nn.PReLU(), args.drop_prob, args.isBias))
                self.bnn.append(BiNN(args.out_ft+args.ft_size[v], args.out_ft, nn.PReLU(), args.drop_prob, args.isBias))

            else:
                self.bnn1.append(BiNN(args.ft_size[v], args.hid_units, nn.PReLU(), args.drop_prob, args.isBias))
                self.bnn1.append(BiNN(args.ft_size[u], args.hid_units, nn.PReLU(), args.drop_prob, args.isBias))
                self.bnn2.append(BiNN(args.hid_units, args.out_ft, nn.PReLU(), args.drop_prob, args.isBias))
                self.bnn2.append(BiNN(args.hid_units, args.out_ft, nn.PReLU(), args.drop_prob, args.isBias))
                self.bnn.append(BiNN(args.out_ft+args.ft_size[v], args.out_ft, nn.PReLU(), args.drop_prob, args.isBias))
                self.bnn.append(BiNN(args.out_ft+args.ft_size[u], args.out_ft, nn.PReLU(), args.drop_prob, args.isBias))

    def forward(self, t1, t2, subfeas, nt_cnt, hopK=None, act=nn.Sigmoid(), isAtt=False):
        if hopK == 1:
            e1 = self.bnn1[nt_cnt](subfeas)
            return e1
        elif hopK == 2:
            e2 = self.bnn2[nt_cnt](subfeas)
            return e2
        else:
            e = self.bnn[nt_cnt](subfeas)
            return e
