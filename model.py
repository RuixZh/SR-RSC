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
from models import *
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
        zeros = torch.zeros((1, self.args.hid_units)).to(self.args.device)
        for epoch in range(self.args.nb_epochs):
            model.train()
            nt_cnt = {1:0,2:0,3:0}
            node_embs = defaultdict()
            for i, rel in enumerate(self.args.bi_graphs):
                v, u = rel.split('-')
                if v == u:
                    dataloader = {rel: torch.utils.data.DataLoader(range(self.args.nb_nodes[v]),
                                                              batch_size=self.args.batch_size,
                                                              shuffle=True,
                                                              drop_last=False)}
                else:
                    dataloader = {rel: torch.utils.data.DataLoader(range(self.args.nb_nodes[v]),
                                                              batch_size=self.args.batch_size,
                                                              shuffle=True,
                                                              drop_last=False),
                                  u+'-'+v: torch.utils.data.DataLoader(range(self.args.nb_nodes[u]),
                                                             batch_size=self.args.batch_size,
                                                             shuffle=True,
                                                             drop_last=False)}

                for r in dataloader:
                    t1, t2 = r.split('-')
                    e = torch.zeros(args.nb_nodes[t1],args.hid_units).to(self.args.device)
                    for batch in dataloader[r]:
                        fbatch = self.search(self.graph[rel][t1], self.features[t2], batch)
                        embs = model(t1, t2, fbatch, nt_cnt[1], hopK=1)
                        e[batch,:] = embs
                    node_embs[t1] = torch.cat((zeros, e), 0)
                    nt_cnt[1] += 1
                for r in dataloader:
                    t1, t2 = r.split('-')
                    e = torch.zeros(args.nb_nodes[t1], args.out_ft).to(self.args.device)
                    for batch in dataloader[r]:
                        fbatch = self.search(self.graph[rel][t1], node_embs[t2], batch)
                        embs = model(t1, t2, fbatch, nt_cnt[2], hopK=2)
                        e[batch,:] = embs
                    node_embs[t1] = torch.cat((e[t1], self.features[t1][1:]), -1)
                    nt_cnt[2] += 1
                for r in dataloader:
                    t1, t2 = r.split('-')
                    e = torch.zeros(args.nb_nodes[t1], args.out_ft).to(self.args.device)
                    for batch in dataloader[r]:
                        fbatch = node_embs[t1][batch]
                        embs = model(t1, t2, fbatch, nt_cnt[3])
                        e[batch,:] = embs
                    node_embs[t1] = e
                    nt_cnt[3] += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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

    def search(self, graph, features, node_list):
        # nbatch = []
        fbatch = []
        for n in node_list:
            neighbors = graph[n]
            # nbatch.append(neighbors)
            fbatch.append(torch.mean(features[neighbors],0))
        return torch.vstack(fbatch)
