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
            args.device = torch.device("cuda:" + str(args.gpu_num_) if torch.cuda.is_available() else "cpu")

        self.args = args
        path = './subgraph/' + args.dataset + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        path += args.dataset
        if (rewrite == False) and os.path.isfile(path+'_neighbors') and os.stat(path+'_neighbors').st_size != 0:
            print ("Exists graph file")
            self.args.neighbors = torch.load(path+'_neighbors')
            self.args.relations = torch.load(path+'_relations')
            self.args.features = torch.load(path+'_features')
            with open(path+"_idx.pkl", "rb") as f:
                data = pickle.load(f)
            self.args.rel2id = data['rel2id']
            self.args.id2rel = data["id2rel"]
            self.args.node2id = data["node2id"]
            self.args.id2node = data["id2node"]
            self.args.labels = data["labels"]
            self.args.node_cnt = data["node_type_cnt"]
            self.args.rel_cnt = data["edge_type_cnt"]
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

        neighbors = defaultdict(list)
        relations = defaultdict(list)
        rel2id = {}
        id2rel = {}
        rel_cnt = Counter()
        with open(path+args.dataset+".relation", "r") as f:
            for line in f.readlines():
                s, t, rel = line.strip().split('\t')
                rel_cnt[rel] += 1
                neighbors[node2id[s]].append(node2id[t])
                if rel in rel2id:
                    relations[node2id[s]].append(rel2id[rel])
                else:
                    i = len(id2rel)
                    rel2id[rel] = i
                    id2rel[i] = rel
        labels = []
        with open(path+args.dataset+".label", 'r') as f:
            for line in f.readlines():
                n, l = line.strip().split('\t')
                labels.append([node2id[n], int(l)])
        labels = np.array(labels)
        torch.save(neighbors, savepath+'_neighbors')
        torch.save(relations, savepath+'_relations')
        torch.save(new_features, savepath+'_features')
        idx = {"rel2id": rel2id,
               "id2rel": id2rel,
               "node2id": node2id,
               "id2node": id2node,
               "labels": labels,
               "node_type_cnt": node_cnt,
               "edge_type_cnt": rel_cnt}
        self.args.rel2id = rel2id
        self.args.id2rel = id2rel
        self.args.node2id = node2id
        self.args.id2node = id2node
        self.args.labels = labels
        self.args.node_cnt = node_cnt
        self.args.rel_cnt = rel_cnt
        self.args.neighbors = neighbors
        self.args.relations = relations
        self.args.features = new_features
        with open(savepath+"_idx.pkl", "wb") as handle:
            pickle.dump(idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
