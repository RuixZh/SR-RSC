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
import evaluation_metrics

class BiHIN(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args

    def training(self):
        model = modeler(self.args).to(self.args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.l2_coef)
        cnt_wait = 0; best = 1e9
        self.graph = {k:{t:[i.to(self.args.device) for i in n] for t, n in v.items()} for k,v in self.graph.items()}
        self.features = {t: f.to(self.args.device) for t, f in self.features.items()}
        for epoch in range(self.args.nb_epochs):
#             for rel in self.args.nb_graphs:
#                 st, tt = rel.split("-")

            model.train()
            optimizer.zero_grad()

            embs, loss = model(self.graph, self.features)
            if loss < best:
                best = loss
                cnt_wait = 0
                print("Epoch {},loss {:.2}".format(epoch, loss))
                torch.save(model.state_dict(), 'saved_model/best_{}_{}.pkl'.format(self.args.dataset, self.args.embedder))
                self.node_embs = {k:torch.squeeze(v).detach().cpu().numpy() for k,v in embs.items()}
                evaluation_metrics(self.node_embs[self.args.task_node], self.args.labels)
            else:
                cnt_wait += 1

            if cnt_wait == self.args.patience:
                break

            loss.backward()
            optimizer.step()


class modeler(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.marginloss = nn.MarginRankingLoss(0.5)

        self.bnn = nn.ModuleList([BiNN(rel,
                                       args.ft_size,
                                       args.hid_units,
                                       nn.PReLU(),
                                       args.drop_prob,
                                       args.isBias) for rel in args.bi_graphs])
        self.bnn2 = nn.ModuleList([BiNN(rel,
                                        {t:args.hid_units for t, fs in args.ft_size.items()},
                                        args.out_ft,
                                        nn.PReLU(),
                                        args.drop_prob,
                                        args.isBias) for rel in args.bi_graphs])
        self.bnn3 = nn.ModuleList([BiNN(rel,
                                        {t:args.out_ft+fs for t, fs in args.ft_size.items()},
                                        args.out_ft,
                                        nn.Identity(),
                                        args.drop_prob,
                                        args.isBias) for rel in args.bi_graphs])
        self.att = nn.ModuleDict({k:Attention(args.out_ft, v) for k,v in args.node_bigraphs.items()})
        # self.disc = nn.ModuleList([Discriminator(args.out_ft, args.out_ft) for _ in args.bi_graphs])

    def forward(self, graph, features, act=nn.Sigmoid(), isAtt=False):
        node_embs = defaultdict(list)
        total_loss = 0.0
        sigmoid = nn.Sigmoid()
        for i, rel in enumerate(self.args.bi_graphs):  # acm p-a p-s
            v, u = rel.split('-')
            vidx = torch.LongTensor(np.random.permutation(self.args.nb_nodes[v])).to(self.args.device)
            uidx = torch.LongTensor(np.random.permutation(self.args.nb_nodes[u])).to(self.args.device)
            vneighbor = torch.vstack([torch.mean(features[u][nid],0) for nid in graph[rel][v]])
            uneighbor = torch.vstack([torch.mean(features[v][nid],0) for nid in graph[rel][u]])
            ve1, ue1 = self.bnn[i](vneighbor, uneighbor)

            ve2 = torch.vstack([torch.mean(ue1[nid],0) for nid in graph[rel][v]])
            ue2 = torch.vstack([torch.mean(ve1[nid],0) for nid in graph[rel][u]])
            ve2, ue2 = self.bnn2[i](ve2, ue2)
            ve3, ue3 = torch.cat((ve2, features[v]), -1), torch.cat((ue2, features[u]), -1)
            ve3, ue3 = self.bnn3[i](ve3, ue3)

            sv = torch.vstack([torch.mean(torch.vstack([ue3[nid],ve3[cid].unsqueeze(0)]),0) for cid,nid in enumerate(graph[rel][v])])
            su = torch.vstack([torch.mean(torch.vstack([ve3[nid],ue3[cid].unsqueeze(0)]),0) for cid,nid in enumerate(graph[rel][u])])

            vpl = sigmoid(torch.sum(ve3 * sv, -1))
            upl = sigmoid(torch.sum(ue3 * su, -1))

            nsv = sv[vidx]
            nsu = su[uidx]
            vnl = sigmoid(torch.sum(ve3 * nsv, -1))
            unl = sigmoid(torch.sum(ue3 * nsu, -1))

            vt = torch.ones(self.args.nb_nodes[v]).to(self.args.device)
            ut = torch.ones(self.args.nb_nodes[u]).to(self.args.device)

            total_loss +=  self.marginloss(vpl, vnl, vt)
            total_loss +=  self.marginloss(upl, unl, ut)

            node_embs[v].append(ve2)
            node_embs[u].append(ue2)

        if isAtt:
            node_embs = {k: self.att[k](v) for k,v in node_embs.items()}
        else:
            node_embs = {k:torch.mean(torch.stack(v), 0) for k,v in node_embs.items()}

        bi_loss = None
        for i, rel in enumerate(self.args.bi_graphs):
            v, u = rel.split('-')
            target = edges[rel].to_dense()
            logit = self.disc[i](node_embs[v], node_embs[u])
            n = self.args.nb_nodes[v]*self.args.nb_nodes[u]
            norm = n/(n-target.sum())
            pos_weight = (n-target.sum())/target.sum()
            if bi_loss is None:
                bi_loss = nn.BCEWithLogitsLoss(weight=norm, pos_weight=pos_weight)(logit, target)
            else:
                bi_loss += nn.BCEWithLogitsLoss(weight=norm, pos_weight=pos_weight)(logit, target)
        # loss_total = bi_loss #+ disc_loss
        return node_embs, total_loss
