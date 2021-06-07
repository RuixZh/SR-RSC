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
        self.features = {k: v.to(self.args.device) for k, v in self.features.items()}
        self.shuff_features = {k: v.to(self.args.device) for k, v in self.shuff_features.items()}
        self.edges = {k: v.to(self.args.device) for k, v in self.edges.items()}
        for epoch in range(self.args.nb_epochs):
#             for rel in self.args.nb_graphs:
#                 st, tt = rel.split("-")
            model.train()
            optimizer.zero_grad()
            embs, loss = model(self.edges, self.features, self.shuff_features, self.args.sparse)
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
        self.bnn = nn.ModuleList([BiNN(rel,
                                       args.ft_size,
                                       args.hid_units,
                                       nn.PReLU(),
                                       args.drop_prob,
                                       args.isBias) for rel in args.bi_graphs])
        self.bnn2 = nn.ModuleList([BiNN(rel,
                                        {t:fs+args.hid_units for t, fs in args.ft_size.items()},
                                        args.out_ft,
                                        nn.Identity(),
                                        args.drop_prob,
                                        args.isBias) for rel in args.bi_graphs])

        self.att = nn.ModuleDict({k:Attention(args.out_ft, v) for k,v in args.node_bigraphs.items()})
        self.disc = nn.ModuleList([Discriminator(args.out_ft, args.out_ft) for _ in args.bi_graphs])

    def forward(self, edges, features, shuff_features, sparse, act=nn.Sigmoid(), isAtt=False):
        node_embs = defaultdict(list)
        for i, rel in enumerate(self.args.bi_graphs):
            v, u = rel.split('-')
            ve, ue = self.bnn[i](edges[rel], edges[u+'-'+v], features[v], features[u], sparse)
            ve = torch.cat((ve, features[v]), 1)
            ue = torch.cat((ue, features[u]), 1)
            ve2, ue2 = self.bnn2[i](edges[rel], edges[u+'-'+v], ve, ue, sparse)
#             emb = torch.cat((ve2, ue2), 1)
            sv = torch.mean(ve2, 0)
            su = torch.mean(ue2, 0)
            s = act(torch.cat((sv, su), -1))

#             shuff_ve, shuff_ue = self.bnn[i](edges[rel], edges[u+'-'+v], shuff_features[v], shuff_features[u], sparse)
#             shuff_ve = torch.cat((shuff_ve, shuff_features[v]), 1)
#             shuff_ue = torch.cat((shuff_ue, shuff_features[u]), 1)
#             shuff_ve2, shuff_ue2 = self.bnn2[i](edges[rel], edges[u+'-'+v], shuff_ve, shuff_ue, sparse)

#             shuff_emb = torch.cat((shuff_ve2, shuff_ue2), 1)

#             logit = self.disc[i](s, emb, shuff_emb)
#             vp_1 = torch.ones(features[v].shape[0])
#             vp_2 = torch.zeros(features[v].shape[0])
#             vp = torch.cat((lbl_1, lbl_2), 1).to(self.args.device)


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
        loss_total = bi_loss #+ disc_loss
        return node_embs, loss_total
