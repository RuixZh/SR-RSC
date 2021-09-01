import os
import numpy as np
np.random.seed(0)
from collections import defaultdict, Counter
# import pickle5 as pickle
import pickle
import torch
import torch.nn as nn
import scipy.sparse as sp
from utils import process

class embedder:
    def __init__(self, args, rewrite=False, useMP2vec=False):
        args.sparse = True
        args.nheads = 1
        args.att_hid_units = 8
        args.batch_size = 500
        args.gpu_num_ = args.gpu_num
        if args.gpu_num_ == "cpu":
            args.device = "cpu"
        else:
            args.device = torch.device("cuda:"+ args.gpu_num_ if torch.cuda.is_available() else "cpu")


        path = "./dataset/"+args.dataset
        norm = True
#         if args.dataset in ['yelp']:
#             norm = True

        with open(path+'/meta_data.pkl', 'rb') as f:
            data = pickle.load(f)
        idx ={}
        for t in data['t_info'].keys():
            idx[t] = torch.LongTensor([i for p, i in data['node2gid'].items() if p.startswith(t)])
        node2id = data['node2gid']

        with open(path+'/labels.pkl', 'rb') as f:
            labels = pickle.load(f)

        with open(path+'/lp_edges.pkl', 'rb') as f:
            edges = pickle.load(f)

        node_rel = defaultdict(list)
        subgraph = {}
        subgraph_nb = {}
        rel_types = set()
        real_adj = {}
        neighbors = defaultdict(set)
        for rel in edges:
#             if (rel=='b-l') or (rel == 'l-b'): continue
            if (rel == 'citing'):
                s, t = 'p', 'p'
                vu = 'cited'
            elif (rel == 'cited'):
                s, t = 'p', 'p'
                vu = 'citing'
            else:
                s, t = rel.split('-')
                vu = t + '-' + s
            node_rel[s].append(rel)
            if vu not in rel_types:
                rel_types.add(rel)
        for nt, rels in node_rel.items():
            rel_list = []
            nb_neighbor = []
            for rel in rels:
                if (rel == 'citing') or (rel == 'cited'):
                    s, t = 'p', 'p'
                else:
                    s, t = rel.split('-')
                e = edges[rel][idx[s],:][:,idx[t]]
                nb = e.sum(1)
                nb_neighbor.append(torch.FloatTensor(nb))
                e = process.sparse_to_tuple(e)  # coords, values, shape
                for i,j in e[0].T:
                    neighbors[i].add(j)
                rel_list.append(torch.sparse_coo_tensor(torch.LongTensor(e[0]),torch.FloatTensor(e[1]), torch.Size(e[2])))
            subgraph[nt] = rel_list
            subgraph_nb[nt] = nb_neighbor
        neighbors_list = []
        for i in range(len(node2id)):
            neighbors_list.append(torch.LongTensor(neighbors[i]))
        if useMP2vec:
            with open("dataset/"+args.dataset+"mp_emb.pkl", "rb") as f:
                features = torch.FloatTensor(pickle.load(f))
            ft = features.shape[1]
#             self.features = torch.vstack((features, torch.zeros((1, ft))))
            self.features = features
        else:
            with open(path+"/node_features.pkl", "rb") as f:
                features = pickle.load(f)
            ft = features.shape[1]
            padding_idx = features.shape[0]
            self.features = process.preprocess_features(features, norm=norm)  # {node_type: [0 || node_features]}

        self.graph = subgraph
        self.neigbor_list = neighbors_list
        self.graph_nb_neighbor = subgraph_nb
        args.node2id = node2id
        args.labels = labels   # {node_type: labels} refer to the sequence of [n, node_cnt[node_type]]
        args.nt_rel = node_rel   # {note_type: [rel1, rel2]}
        args.node_cnt = idx  # {note_type: nb}
        args.ft_size = ft
        args.node_size = len(node2id)
        args.rel_types = rel_types
        self.args = args
        print("node_type num_node:")
        for t, num in self.args.node_cnt.items():
            print("\n%s\t %s"%(t, len(num)))
        print("Graph prepared!")


    def corrupt_structure(self, e, rate, nb):
        sl, tl = e[2]
        adj = [(i,j) for i,j in e[0].T]
        corruption_edges = int(sl * tl * rate) + 1
        idx = []
        for k in range(corruption_edges):
            i = np.random.randint(0, sl-1)
            j = np.random.randint(0, tl-1)
            if (i,j) in adj:
                adj.remove((i,j))
                nb[i] -= 1
            else:
                adj.append((i,j))
                nb[i] += 1

        adj = np.array(adj)
        cor_adj = sp.coo_matrix((np.ones(adj.shape[0]), (adj[:, 0], adj[:, 1])),
                           shape=(sl, tl),
                           dtype=np.float32)
        return cor_adj, nb
