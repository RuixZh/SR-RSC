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

class SubHIN(embedder):
    def __init__(self, args):

        embedder.__init__(self, args)
        self.args = args

    def training(self):
        # self.features = self.features.to(self.args.device)
        # self.graph = {t: [m.to(self.args.device) for m in ms] for t, ms in self.graph.items()}
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
                embs, loss = model(self.graph, self.features, self.neigbor_list, batch)
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
                np.save("output_emb/"+self.args.dataset+".test.npy", outs)

            else:
                cnt_wait += 1

class modeler(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.isAtt = self.args.isAtt
        self.marginloss = nn.MarginRankingLoss(0.5)
        self.bnn = nn.ModuleDict()
        self.fc = nn.ModuleDict()
        self.att = nn.ModuleDict()
        self.localatt = nn.ModuleDict()
        self.semanticatt = nn.ModuleDict()
        self.disc = nn.ModuleDict()
#         self.Z = nn.Parameter(torch.empty(self.args.node_size, self.args.out_ft*self.args.nheads))
#         nn.init.xavier_uniform_(self.Z)
        self.fc['0'] = FullyConnect(args.hid_units* args.nheads,
                                          args.ft_size,
                                          nn.Identity(),
                                          args.drop_prob,
                                          args.isBias)
        self.fc['1'] = FullyConnect(args.out_ft* args.nheads,
                                      args.ft_size,
                                      nn.Identity(),
                                      args.drop_prob,
                                      args.isBias)
        for t, rels in self.args.nt_rel.items():  # {note_type: [rel1, rel2]}
            self.bnn['0'+t] = GCN(len(rels),
                                   args.ft_size,
                                   args.hid_units* args.nheads,
                                   nn.PReLU(),
                                   args.drop_prob,
                                   args.isBias)
            self.bnn['1'+t] = GCN(len(rels),
                                   args.hid_units* args.nheads,
                                   args.out_ft* args.nheads,
                                   nn.PReLU(),
                                   args.drop_prob,
                                   args.isBias)
            self.fc['0'+t] = FullyConnect(args.hid_units* args.nheads + args.ft_size,
                                          args.hid_units* args.nheads,
                                          nn.Identity(),
                                          args.drop_prob,
                                          args.isBias)
            self.fc['1'+t] = FullyConnect(args.out_ft* args.nheads + args.ft_size,
                                          args.out_ft* args.nheads,
                                          nn.Identity(),
                                          args.drop_prob,
                                          args.isBias)

            self.fc[t] = FullyConnect(args.out_ft* args.nheads,
                                          args.ft_size,
                                          nn.Identity(),
                                          args.drop_prob,
                                          args.isBias)

#             self.att['0'+t] = Attention(len(rels), args.hid_units)
#             self.att['1'+t] = Attention(len(rels), args.out_ft)
#             self.disc['0'+t] = Discriminator(args.hid_units, args.ft_size)
#             print(t, len(rels))
#             self.disc[t] = BiLinear(args.out_ft* args.nheads, len(rels))
#             for i in range(args.nheads):
#                 self.nodeatt['0'+t+str(i)] = NodeAttention(args.ft_size, args.hid_units, concat=True)
#                 self.nodeatt['1'+t+str(i)] = NodeAttention(args.hid_units, args.out_ft, concat=True)
            self.semanticatt['0'+t] = SemanticAttention(args.hid_units * args.nheads, args.att_hid_units, len(rels))
            self.semanticatt['1'+t] = SemanticAttention(args.out_ft * args.nheads, args.att_hid_units, len(rels))



    def forward(self, graph, features, neigbor_list, batch):
        totalLoss = 0.0
        reg_loss = 0.0
#         embs2 = torch.zeros((self.args.node_size, self.args.hid_units*self.args.nheads)).to(self.args.device)
#         for t, ns in self.args.node_cnt.items():
#             embs2[ns] = self.fc[t](features[self.args.node_cnt[t]])

        embs1 = torch.zeros((self.args.node_size, self.args.hid_units*self.args.nheads)).to(self.args.device)
        for n, rels in self.args.nt_rel.items():   # p: [[Nei(p-a), Nei(p-c)]]  (Np, Nprel)
            vec = []
            fake_vec = []
            for j, rel in enumerate(rels):
                if (rel == 'citing') or (rel == 'cited'):
                    s, t = 'p', 'p'
                else:
                    s, t = rel.split('-')
                # nb = graph_nb[n][j].to(self.args.device)
                g = graph[n][j].to(self.args.device)
                f = features[self.args.node_cnt[t]].to(self.args.device)
                mean_neighbor = torch.spmm(g, f)
                # mean_neighbor = torch.div(mean_neighbor, nb) # (Nt, ft)
                vec.append(mean_neighbor.unsqueeze(0))  # (1, Nt, ft)

            vec = torch.cat(vec, 0)  # (rel_size, Nt, ft_size)
            vec = self.bnn['0'+n](vec) # (rel_size, Nt, hd_units)
            ns = vec.size(1)

            if self.isAtt:
                v_summary = self.semanticatt['0'+n](vec.view(len(rels)*ns,-1))
            else:
                v_summary = torch.mean(vec, 0)  # (Nt, hd)
            f2 = features[self.args.node_cnt[n]].to(self.args.device)
            v_cat = torch.hstack((v_summary, f2))
            v_summary = self.fc['0'+n](v_cat)
            embs1[self.args.node_cnt[n]] = v_summary
#         fc1 = self.fc['0'](embs1)
#         reg_loss = ((fc1 - features) ** 2).sum()

        embs = torch.zeros((self.args.node_size, self.args.out_ft*self.args.nheads)).to(self.args.device)

        for n, rels in self.args.nt_rel.items():   # p: [[Nei(p-a), Nei(p-c)]]  (Np, Nprel)
            vec = []
            fake_vec = []
            for j, rel in enumerate(rels):
                if (rel == 'citing') or (rel == 'cited'):
                    s, t = 'p', 'p'
                else:
                    s, t = rel.split('-')
                g = graph[n][j].to(self.args.device)
                f = features[self.args.node_cnt[t]].to(self.args.device)
                mean_neighbor = torch.spmm(g, f)
                vec.append(mean_neighbor.unsqueeze(0))  # (1, Nt, ft)

            vec = torch.cat(vec, 0)  # (rel_size, Nt, ft_size)
            vec = self.bnn['1'+n](vec) # (rel_size, Nt, hd_units)
            ns = vec.size(1)

            if self.isAtt:
                v_summary = self.semanticatt['1'+n](vec.view(len(rels)*ns,-1))
            else:
                v_summary = torch.mean(vec, 0)  # (Nt, hd)
            f2 = features[self.args.node_cnt[n]].to(self.args.device)
            v_cat = torch.hstack((v_summary, f2))
            v_cat = torch.hstack((v_summary, f2))
            v_summary = self.fc['1'+n](v_cat)

            embs[self.args.node_cnt[n]] = v_summary

            fc2 = self.fc[t](v_summary)
            reg_loss += ((fc2 - features[self.args.node_cnt[n]]) ** 2).sum()

        totalLoss += self.args.reg_coef * reg_loss

        vec = embs[batch]
        pvec = torch.cat([embs[neigbor_list[i]].mean(0, keepdim=True) for i in batch], dim=0)
        shuf_index = torch.randperm(len(batch)).to(self.args.device)
        shuf_vec = pvec[shuf_index]
        logits_neg = torch.sum(vec * shuf_vec, dim=-1).squeeze()
        logits_pos = torch.sum(vec * pvec, dim=-1).squeeze()

        ones = torch.ones(logits_pos.size(0)).to(self.args.device)
        totalLoss += self.marginloss(torch.sigmoid(logits_pos), torch.sigmoid(logits_neg), ones)

        return embs, totalLoss
