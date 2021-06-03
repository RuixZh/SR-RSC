import numpy as np
from collections import defaultdict
import pickle
import torch
import torch.nn as nn

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
        args.sparse = True
        args.gpu_num_ = args.gpu_num
        if args.gpu_num_ == 'cpu':
            args.device = 'cpu'
        else:
            args.device = torch.device("cuda:" + str(args.gpu_num_) if torch.cuda.is_available() else "cpu")
        file = "dataset/"+args.dataset+".pkl"
        with open(file, "rb") as f:
            data = pickle.load(f)
        self.relation_dict = data["rel_dict"]
        # self.features = data["feature"]  # sparse (nb_nodes, ft_size)
        self.task_nodes = data["label"][ : , 0]
        self.labels = data["label"][ : , 1]  # array (N, 1)
        self.nb_classes = data["n_class"]
        self.rel_fea = defaultdict(defaultdict)
        features = {}
        for t, fea in data["feature"].items():
            features[t] = preprocess_features(fea)
            self.rel_fea[t]['original'] = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(features[t])).to(args.device)
        for rel, neighbor_dict in data['rel_dict'].items():
            st, tt = rel.split("-")
            node_num, ft_size = features[st].shape
            fea_matrix = np.zeros_like(features[st].toarray())
            for n, node in enumerate(data['t_info'][st]['ind']):
                neighbors = neighbor_dict[node]
                if len(neighbors)==0: continue
                nid = [data['node2lid'][data['id2node'][n]] for n in neighbors]
                f = features[tt][nid].mean(0)
                fea_matrix[n] = np.array(f)
            self.rel_fea[st][rel] = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(fea_matrix)).to(args.device)
        #     for node, neighbors in neighbor_dict.items():
        #         nid = [data['node2lid'][data['id2node'][n]] for n in neighbors]
        #         f = self.features[tt][nid].mean(0)
        #         self.rel_fea[rel][node] = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(f))
        # self.features = {t:sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(f)) for t, f in features.items()}
        print("Bipatite graph-based HIN constructed!")

        args.nb_nodes = self.task_nodes
        args.ft_size = {t:ft.shape[1] for t,ft in features.items()}
        args.nb_classes = self.nb_classes
        args.nb_graphs = len(data["bi_relations"])
        args.bi_graphs = data["bi_relations"]

        print("Bi-graph:%s"%args.nb_graphs)
        print("node_type num_node ft_size")
        for t,ft in features.items():
            print("\n%s\t %s\t %s"%(t,ft.shape[0],ft.shape[1]))
        print("num_class:%s"%args.nb_classes)
        self.args = args
