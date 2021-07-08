import os
import numpy as np
np.random.seed(0)
from collections import defaultdict, Counter
import pickle
import torch
import torch.nn as nn

def preprocess_features(features, normalize=False):
    """Row-normalize feature matrix and convert to tuple representation"""
    if normalize:
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1.0).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
    else:
        features = features.toarray()
    return torch.FloatTensor(features)

class embedder:
    def __init__(self, args, rewrite=False):
        args.sparse = True
        args.gpu_num_ = args.gpu_num
        if args.gpu_num_ == "cpu":
            args.device = "cpu"
        else:
            args.device = torch.device("cuda:" + args.gpu_num_ if torch.cuda.is_available() else "cpu")

        self.args = args
        path = './subgraph/' + args.dataset + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        path += args.dataset
        if (rewrite == False) and os.path.isfile(path+'_info.pkl') and os.stat(path+'_info.pkl').st_size != 0:
            print ("Exists graph file")
            with open(path+"_info.pkl", "rb") as f:
                data = pickle.load(f)
            self.graph = data["graph"]  # {node: [[Np1], [Np2]]}
            self.features = data['features']  # [0, node_features]
            self.rel2id = data["rel2id"]
            self.id2rel = data["id2rel"]
            self.node2id = data["node2id"]
            self.id2node = data["id2node"]
            self.labels = data["labels"]
            self.node_cnt = data["node_type_cnt"]
            self.rel_cnt = data["edge_type_cnt"]
            self.args.nb_rel = len(self.rel2id)
            self.args.nb_node = len(self.node2id)
            self.args.ft_size = self.features.shape[1]

        else:
            print ("Construct graph")
            self.build(path)

        print("node_type num_node:")
        for t, num in self.args.node_cnt.items():
            print("\n%s\t %s"%(t, num))
        print("relation_type num_relation:")
        for t, num in self.args.rel_cnt.items():
            print("\n%s\t %s"%(t, num))

        print("Graph prepared!")

    def build(self, savepath):
        path = "./dataset/"+args.dataset + '/'
        with open(path+args.dataset+".content.pkl", "rb") as f:
            features = pickle.load(f)
        node2id = {}
        id2node = {}
        node_cnt = {}
        mft = 0
        for t, f in features.items():
            nt, ft = f.shape
            node_cnt[t] = nt
            for i in range(nt):
                cnt = len(node2id)+1
                node2id[t+str(i)] = cnt
                id2node[cnt] = t+str(i)
            mft = max(mft, ft)
        fea_pad = []
        for f in features.values():
            nt, ft = f.shape
            if mft != ft:
                zeros = np.zeros((nt, mft-ft))
                fea_pad.append(preprocess_features(sp.hstack((f, zeros)), normalize=False))
            else:
                fea_pad.append(preprocess_features(f, normalize=False))
        new_features = torch.vstack((torch.zeros((1,mft)), torch.vstack(fea_pad)))

        neighbors = defaultdict(lambda:defaultdict(list))
        rel2id = {}
        id2rel = {}
        rel_cnt = Counter()
        with open(path+args.dataset+".relation", "r") as f:
            for line in f.readlines():
                s, t, rel = line.strip().split('\t')
                rel_cnt[rel] += 1
                if rel not in rel2id:
                    i = len(id2rel)
                    rel2id[rel] = i
                    id2rel[i] = rel
                neighbors[node2id[s]][rel2id[rel]].append(node2id[t])
        graph = {}
        for s, rts in neighbors.items():
            rel_list = [[0] for _ in range(len(rel2id))]
            for r, ts in rts.items():
                if 0 in rel_list[r]:
                    rel_list[r] = ts
                else:
                    rel_list[r].extend(ts)
            graph[s] = [torch.LongTensor(l) for l in rel_list]
        new_graph = []
        for i in range(len(graph)):
            new_graph.append(graph[i])
        labels = []
        with open(path+args.dataset+".label", 'r') as f:
            for line in f.readlines():
                n, l = line.strip().split('\t')
                labels.append([node2id[n], int(l)])
        labels = np.array(labels)
        info = {"graph": new_graph,
                "features":new_features,
                "rel2id": rel2id,
                "id2rel": id2rel,
                "node2id": node2id,
                "id2node": id2node,
                "labels": labels,
                "node_type_cnt": node_cnt,
                "edge_type_cnt": rel_cnt}
        self.rel2id = rel2id
        self.id2rel = id2rel
        self.node2id = node2id
        self.id2node = id2node
        self.labels = labels
        self.node_cnt = node_cnt
        self.rel_cnt = rel_cnt
        self.graph = new_graph
        self.features = new_features
        self.args.nb_rel = len(self.rel2id)
        self.args.nb_node = len(self.node2id)
        self.args.ft_size = self.features.shape[1]
        with open(savepath+"_info.pkl", "wb") as handle:
            pickle.dump(info, handle, protocol=pickle.HIGHEST_PROTOCOL)
