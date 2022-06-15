# GMCAP: Graph Multi-Convolution and Attention Pooling for Graph Classification

This is a PyTorch implementation of GMCAP algorithm, which adopts graph multi-convolution and attention pooling to learn the graph-level representation for the entire graph. 



## Requirements

- python
- pytorch
- pytorch_geometric (pyg)

Note:

This code repository is built on [pyg](https://github.com/pyg-team/pytorch_geometric), which is a Python package built for easy implementation of graph neural network model family. Please refer [here](https://github.com/pyg-team/pytorch_geometric) for how to install and utilize the library.



### Datasets

Graph classification benchmarks are publicly available at [here](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets).



### Run

To run GMCAP, just execute the following command for graph classification task:

```
python main.py
```