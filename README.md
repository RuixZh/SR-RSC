# SR-RSC

The source code for the paper: "[A Simple Meta-path-free Framework for Heterogeneous Network Embedding](https://dl.acm.org/doi/abs/10.1145/3511808.3557223)" accepted in CIKM 2022.
```
@inproceedings{10.1145/3511808.3557223,
author = {Zhang, Rui and Zimek, Arthur and Schneider-Kamp, Peter},
title = {A Simple Meta-Path-Free Framework for Heterogeneous Network Embedding},
year = {2022},
isbn = {9781450392365},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3511808.3557223},
doi = {10.1145/3511808.3557223},
abstract = {Network embedding has recently attracted attention a lot since networks are widely used in various data mining applications. Attempting to break the limitations of pre-set meta-paths and non-global node learning in existing models, we propose a simple but effective framework for heterogeneous network embedding learning by encoding the original multi-type nodes and relations directly in a self-supervised way. To be more specific, we first learn the relation-based embeddings for global nodes from the neighbor properties under each relation type and exploit an attentive fusion module to combine them. Then we design a multi-hop contrast to optimize the regional structure information by utilizing the strong correlation between nodes and their neighbor-graphs, where we take multiple relationships into consideration by multi-hop message passing instead of pre-set meta-paths. Finally, we evaluate our proposed method on various downstream tasks such as node clustering, node classification, and link prediction between two types of nodes. The experimental results show that our proposed approach significantly outperforms state-of-the-art baselines on these tasks.},
booktitle = {Proceedings of the 31st ACM International Conference on Information &amp; Knowledge Management},
pages = {2600–2609},
numpages = {10},
keywords = {self-supervised learning, contrastive learning, meta-path-free, heterogeneous network embedding},
location = {Atlanta, GA, USA},
series = {CIKM '22}
}
```
