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
        self.graph = [[i.to(self.args.device) for i in n] for n in self.graph]  # [0: nb_node-1] neighbor_idx [0: nb_node] with 0 padding
        self.features = self.features.to(self.args.device)  # [0: nb_node] with 0 padding
        # zeros = torch.zeros((1, self.args.hid_units)).to(self.args.device)
        for epoch in range(self.args.nb_epochs):
            model.train()
            dataloader = torch.utils.data.DataLoader(range(self.args.nb_node),  # [0: nb_node-1]
                                                      batch_size=self.args.batch_size,
                                                      shuffle=True,
                                                      drop_last=False)
            features = self.features
            isConcat = False
            for k in range(self.args.hopK):
                if k == self.args.hopK-1:
                    isConcat = True
                new_fea = torch.zeros((1+self.args.nb_node, self.args.out_ft)).to(self.args.device)
                for batch in dataloader:
                    vec_1 = model(batch, self.graph, features, k, isConcat=isConcat)
                    new_fea[batch+1] = vec_1
                features = new_fea


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

    # def search(self, graph, features, node_list):
    #     fbatch = []
    #     for n in node_list:
    #         rel_neighbors = graph[n]
    #         v = []
    #         for neighbors in rel_neighbors:
    #             v.append(torch.mean(features[neighbors],0))
    #         fbatch.append(torch.vstack(v).unsqueeze(0))
    #     return torch.vstack(fbatch)  # (batch_size, rel_size, dim)
