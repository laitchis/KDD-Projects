# KDD-Projects
Repo for CSC 466 projects, imported from Colab


Luke Aitchison

Running PageRank:

python3 pageRank.py [filename] [d] [useEdgeWeights]

Where filename is the path to a data file, d is a float between 0 and 1, and useEdgeWeights is either a 0 or a 1. 

This implementation of PageRank assumes a directed graph, so edges are only read in one way. If an undirected graph is the intention, the edges should be doubled in reverse in the input file.

Running Clustering:

python3 clustering.py [filename] [algorithm] [k/epsilon] [minPoints]

Where filename is the path to a data file, algorithm is a d for DBScan, k for K-Means, and h for hierarchical clustering. If K-Means is chosen, k is the number of means.
If DBScan is chosen, epsilon and minPoints are necessary hyperparameters for the algorithm.

