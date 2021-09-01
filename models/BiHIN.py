import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.random.seed(0)
from collections import defaultdict
from embedder import embedder
from layers import *
from evaluation import evaluation_metrics

class BiHIN(embedder):
    def __init__(self, args):

        embedder.__init__(self, args)
        self.args = args

    def training(self):
        # self.features = self.features.to(self.args.device)
        # self.graph = {t: [m.to(self.args.device) for m in ms] for t, ms in self.graph.items()}  # normalized adj
        # self.graph_nb_neighbor = {t: [m.to(self.args.device) for m in ms] for t, ms in self.graph_nb_neighbor.items()}

        model = modeler(self.args).to(self.args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.l2_coef)
        cnt_wait = 0; best = 1e9
        loader = DataLoader(list(self.args.node2id.values()), batch_size=self.args.batch_size)

        for epoch in range(self.args.nb_epochs):
            train_loss = 0.0
            for bidx, batch in enumerate(loader):
                model.train()
                optimizer.zero_grad()
                embs = model(self.graph, self.features)
                pn, nn = self.sample(batch)
                loss = model.loss(batch, embs, pn, nn)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss = train_loss/len(batch)
            if train_loss < best:
                best = train_loss
                cnt_wait = 0
                print("Epoch {},loss {:.5}".format(epoch, l))
                outs = embs.detach().cpu().numpy()
                evaluation_metrics(outs, self.args.labels)
                np.save("output_emb/"+self.args.dataset+".npy", outs)

            else:
                cnt_wait += 1
    def sample(self, batch):
        pos_neighbors = []
        neg_neighbors = []
        for i in batch:
            poslist = self.neigbor_list[i]
            neglist = set(range(self.args.node_size))-set([i])-nlist
            for n in range(self.args.n_samples):
                pos_neighbors.append(np.random.choice(poslist))
                neg_neighbors.append(np.random.choice(neglist))
        return torch.LongTensor(pos_neighbors), torch.LongTensor(neg_neighbors)


class modeler(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.isAtt = self.args.isAtt
        self.marginloss = nn.MarginRankingLoss(0.8)
        self.bnn = nn.ModuleDict()
        self.fc = nn.ModuleDict()
        self.att = nn.ModuleDict()
        self.semanticatt = nn.ModuleDict()

        for rel in self.args.rel_types:
            self.dgcn[rel] = DGCN(args.ft_size, args.ft_size, args.out_ft, act=nn.ReLU(), drop_prob=args.drop_prob, isBias=args.isBias, leaky=0.1)

        for t, rels in self.args.nt_rel.items():
            self.semanticatt[t] = SemanticAttention(args.out_ft, args.att_hid_units, len(rels))

    def forward(self, graph, features, graph_nb):
        rel_embs = defaultdict(list)
        for uv in self.args.rel_types:
            if (uv == 'citing'):
                s, t = 'p', 'p'
                vu = 'cited'
            elif (uv == 'cited'):
                s, t = 'p', 'p'
                vu = 'citing'
            else:
                s, t = rel.split('-')
                vu = t + '-' + s

            ufea = features[self.node_cnt[s]].to(self.args.device)
            vfea = features[self.node_cnt[t]].to(self.args.device)
            uv_adj = graph[s][uv]
            vu_adj = graph[t][vu]
            Hu, Hv = self.dgcn[rel](uv_adj, vu_adj, ufea, vfea)
            rel_embs[s].append(Hu)
            rel_embs[t].append(Hv)

        outputs = torch.zeros((self.args.node_size, self.args.out_units)).to(self.args.device)
        for nt, embs in rel_embs.items():
            outs = self.semanticatt[nt](torch.vstack(embs))
            outputs[self.node_cnt[nt]] = outs

        return outputs

    def loss(self, batch, embs, pos_neighbors, neg_neighbors):
        # using batch size for loss
        totalLoss = 0.0

        vec = embs[batch].repeat(self.args.n_samples, 1, 1)
        pvec = embs[pos_neighbors]
        nvec = embs[neg_neighbors]

        logits_pos = torch.sum(vec * pvec, dim=-1)
        logits_neg = torch.sum(vec * nvec, dim=-1)

        ones = torch.ones(logits_pos.size(0)).to(self.args.device)
        totalLoss += self.marginloss(torch.sigmoid(logits_pos), torch.sigmoid(logits_neg), ones)

        return totalLoss
