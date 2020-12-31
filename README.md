# PyTextGCN

A re-implementation of TextGCN by [Yao et al.: Graph Convolutional Networks for Text Classification](https://arxiv.org/abs/1809.05679).
This implementation uses Cython for the text-to-graph transformation, making it rather fast.
Graphs and GCN are based on the [pytorch-geometric](https://github.com/rusty1s/pytorch_geometric) library.

## Requirements

This project was built with:

- Python 3.8.5
- Cython 0.29.21
- CUDA 10.2 (optional for GPU support)
- scikit-learn 0.23.2
- pytorch 1.7.0
- torch-geometric 1.6.3

At least the Text2Graph-module should work with other versions of these libraries, too.