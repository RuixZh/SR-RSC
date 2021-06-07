from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn import metrics
from munkres import Munkres, print_matrix
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
import warnings

class evaluation_metrics():
    def __init__(self, embs, labels):

        self.evaluate_cluster(embs, labels, len(set(labels)))
        self.evaluate_clf(embs, labels)

    def evaluate_cluster(self, embs, labels, n_label):

        kmeans = KMeans(n_clusters=n_label, random_state=0).fit(embs)
        preds = kmeans.predict(embs)
        nmi = metrics.normalized_mutual_info_score( labels_true=labels, labels_pred=np.array(preds))
        adjscore = metrics.adjusted_rand_score(labels, np.array(preds))
        print('NMI: %.5f, ARI: %.5f'%(nmi, adjscore))

    def evaluate_clf(self, embs, labels):

        true_label = labels

        X_train, X_test, Y_train, Y_test = train_test_split(embs, true_label, test_size=0.8, random_state=0)
        lr = LogisticRegression(max_iter=500)
        lr.fit(X_train, Y_train)
        Y_pred = lr.predict(X_test)
        f1_micro = metrics.f1_score(Y_test, Y_pred, average='micro')
        f1_macro = metrics.f1_score(Y_test, Y_pred, average='macro')
        print('training 20%% f1_macro=%f, f1_micro=%f' % (f1_macro, f1_micro))

        X_train, X_test, Y_train, Y_test = train_test_split(embs, true_label, test_size=0.6, random_state=0)
        lr = LogisticRegression()
        lr.fit(X_train, Y_train)
        Y_pred = lr.predict(X_test)
        f1_micro = metrics.f1_score(Y_test, Y_pred, average='micro')
        f1_macro = metrics.f1_score(Y_test, Y_pred, average='macro')
        print('training 40%% f1_macro=%f, f1_micro=%f' % (f1_macro, f1_micro))

        X_train, X_test, Y_train, Y_test = train_test_split(embs, true_label, test_size=0.4, random_state=0)
        lr = LogisticRegression()
        lr.fit(X_train, Y_train)
        Y_pred = lr.predict(X_test)
        f1_micro = metrics.f1_score(Y_test, Y_pred, average='micro')
        f1_macro = metrics.f1_score(Y_test, Y_pred, average='macro')
        print('training 60%% f1_macro=%f, f1_micro=%f' % (f1_macro, f1_micro))

        X_train, X_test, Y_train, Y_test = train_test_split(embs, true_label, test_size=0.2, random_state=0)
        lr = LogisticRegression()
        lr.fit(X_train, Y_train)
        Y_pred = lr.predict(X_test)
        f1_micro = metrics.f1_score(Y_test, Y_pred, average='micro')
        f1_macro = metrics.f1_score(Y_test, Y_pred, average='macro')
        print('training 80%% f1_macro=%f, f1_micro=%f' % (f1_macro, f1_micro))
