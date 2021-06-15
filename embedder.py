import os
import numpy as np
np.random.seed(0)
from collections import defaultdict
import pickle
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

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
    def __init__(self, args):
        file = "./dataset/"+args.dataset+".pkl"
        with open(file, "rb") as f:
            data = pickle.load(f)

        args.task_node = data["task_node"]
        args.nb_task_node = data["label"][ : , 0]
        args.labels = data["label"][ : , 1]  # array (N, 1)
        args.nb_classes = len(set(args.labels))

        args.sparse = True
        args.gpu_num_ = args.gpu_num
        if args.gpu_num_ == "cpu":
            args.device = "cpu"
        else:
            args.device = torch.device("cuda:" + str(args.gpu_num_) if torch.cuda.is_available() else "cpu")
        args.nb_nodes = {t:ft.shape[0] for t,ft in data["feature"].items()}
        args.ft_size = {t:ft.shape[1] for t,ft in data["feature"].items()}

        args.bi_graphs = data["bigraphs"]
        args.nb_graphs = len(args.bi_graphs)
        args.node_bigraphs = {k:v for k,v in Counter('+'.join(data["bigraphs"])).items() if k in data["t_info"]}

        self.args = args
        print("Bi-graph:%s"%args.nb_graphs)
        print("node_type num_node ft_size")
        for t,ft in self.features.items():
            print("\n%s\t %s\t %s"%(t,ft.shape[0],ft.shape[1]))
        print("num_class:%s"%args.nb_classes)

        self.path = './subgraph/' + args.dataset

        if os.path.isfile(self.path+'_graph') and os.stat(self.path+'_graph').st_size != 0:
            print ("Exists graph file")
            self.graph = torch.load(self.path+'_graph')
        else:
            print ("Extract graph")
            self.build(data["adj_list"], data["feature"])
        print("Graph prepared!")

    def build(self, adj_list, features):
        self.graph = {}
        for rel in args.bi_graphs:
            s, t = rel.split('-')
            nb_s = args.nb_nodes[s]
            nb_t = args.nb_nodes[t]
            edge = {}
            for i in range(nb_s):
                nodes = adj_list[rel][i]
                x = preprocess_features(features[t][nodes], normalize=False)
                new_index = [[], []]
                new_index[0] += len(nodes) * [i]
                new_index[1] += nodes
                edge[s+str(i)] = Data(x, torch.LongTensor(new_index))
            for i in range(nb_t):
                nodes = adj_list[t+'-'+s][i]
                x = preprocess_features(features[s][nodes], normalize=False)
                new_index = [[], []]
                new_index[0] += len(nodes) * [i]
                new_index[1] += nodes
                edge[t+str(i)] = Data(x, torch.LongTensor(new_index))
            self.graph[rel] = edge
        torch.save(self.graph, self.path+'_graph')
