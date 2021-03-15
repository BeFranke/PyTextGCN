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
- gcc 9.3.0
- nltk 3.5
- scipy 1.5.2

At least the Text2Graph-module should work with other versions of these libraries, too.

## Installation
From the project root, the cython compilation can be done with:

`cd textgcn/lib/clib && python setup.py build_ext --inplace`

## Usage
To compute a graph from a list of strings (where each string contains the text of one document) called `X`, a list of labels called `y` and a list of test indices `test_idx`, simply run:

`from textgcn import Text2GraphTransformer`

`t2g = Text2GraphTransformer()`

`graph = t2g.fit_transform(X, y, test_idx=test_idx)`

The resulting object `graph` is a `torch_geometric.data.Data` object containing the resulting graph and can be processed by any torch-geometric-based network.
For more information on parameters of the `Text2GraphTransformer` and the resulting `Data`-object, consult the [documentation](textgcn/lib/text2graph.py) in the source files.

## Documentation
Currently resides in the source files.


## How to reproduce our experiments
Run one of the Python-files in this folder. Each script will produce the result for one experiment for one seed. Seeds can be changed by editing the `seed`-variable in the first part of the file.
