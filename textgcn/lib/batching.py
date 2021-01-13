from collections import Iterable
from typing import List
import torch_geometric as tg
import torch as th
import joblib as jl
import numpy as np


def _sample(doc_number):
    pass


class TextGCNBatcher(Iterable):
    def __init__(self, g: tg.data.Data, docs_per_batch: int = 400, n_hops: int = 2,
                 n_jobs: int = 6):
        self.n_jobs = n_jobs
        self.n_hops = n_hops
        self.docs_per_batch = docs_per_batch
        doc_mask = g.train_mask + g.test_mask
        if g.val_mask is not None:
            doc_mask += g.val_mask
        n_docs = th.sum(doc_mask).item()
        self.doc_idx = th.randperm(n_docs) + g.n_vocab
        self.i = 0
        self.g = g

    def __iter__(self):
        return self

    def __next__(self):
        low = self.i * self.docs_per_batch
        if low > len(self.doc_idx):
            raise StopIteration
        self.i += 1
        high = min(self.i * self.docs_per_batch, len(self.doc_idx))
        docs = self.doc_idx[low:high]
        result = jl.Parallel(n_jobs=self.n_jobs)(
            jl.delayed(_sample)(doc_number) for doc_number in docs
        )
        nodes, edges = zip(*result)
        nodes, ind_nodes = th.unique(th.cat(nodes), dim=0, return_inverse=True)

        edges, ind_edges = th.unique(th.cat(edges), dim=1, return_inverse=True)
        return tg.data.Data(
            x=nodes,
            y=self.g.y[ind_nodes],
            train_mask=self.g.val_mask[ind_nodes],
            test_mask=self.g.test_mask[ind_nodes],
            val_mask=None if self.g.val_mask is None else self.g.val_mask[ind_nodes],
            edge_index=edges,
            edge_attr=self.g.edge_attr[ind_edges]
        )

    def __len__(self):
        return int(len(self.doc_idx) / self.docs_per_batch)
