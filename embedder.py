import numpy as np
np.random.seed(0)
from collections import defaultdict
import pickle
import torch
import torch.nn as nn
import argparse

def parse_args():
    # input arguments
    parser = argparse.ArgumentParser(description='BiHIN')
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--embedder', nargs='?', default='BiHIN')
    parser.add_argument('--dataset', nargs='?', default='acm')
    return parser.parse_known_args()

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1.0).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class embedder:
    def __init__(self, args):
        file = "dataset/"+args.dataset+".pkl"
        with open(file, "rb") as f:
            data = pickle.load(f)
        self.features = {}
        self.shuff_features = {}
        for t, ft in data["feature"].items():
            f = preprocess_features(ft)
            self.features[t] = torch.FloatTensor(f.toarray())  # sparse (nb_nodes, ft_size)
            idx = np.random.permutation(ft.shape[0])
            self.shuff_features[t] = torch.FloatTensor(f[idx].toarray())
        args.task_node = data["task_node"]
        args.nb_task_node = data["label"][ : , 0]
        args.labels = data["label"][ : , 1]  # array (N, 1)
        self.edges = {r: sparse_mx_to_torch_sparse_tensor(v) for r,v in data["edges"].items()}

        print("Bipatite graph-based HIN constructed!")
        args.sparse = True
        args.gpu_num_ = args.gpu_num
        if args.gpu_num_ == "cpu":
            args.device = "cpu"
        else:
            args.device = torch.device("cuda:" + str(args.gpu_num_) if torch.cuda.is_available() else "cpu")
        args.nb_nodes = {t:ft.shape[0] for t,ft in self.features.items()}
        args.ft_size = {t:ft.shape[1] for t,ft in self.features.items()}
        args.nb_classes = data["n_class"]
        args.bi_graphs = data["bigraphs"]
        args.nb_graphs = len(args.bi_graphs)
        args.node_bigraphs = {k:v for k,v in Counter('+'.join(data["bigraphs"])).items() if k in data["t_info"]}

        print("Bi-graph:%s"%args.nb_graphs)
        print("node_type num_node ft_size")
        for t,ft in self.features.items():
            print("\n%s\t %s\t %s"%(t,ft.shape[0],ft.shape[1]))
        print("num_class:%s"%args.nb_classes)
        self.args = args
