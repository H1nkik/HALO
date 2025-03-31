# HALO
![Static Badge](https://img.shields.io/badge/Deep_Graph_Clustering-blue)
![Static Badge](https://img.shields.io/badge/Code-PyTorch-8A2BE2)


## Preparation
**Dependency**

* Python == 3.8.8

* cuda == 11.0

* torch == 1.7.1

* numpy == 1.24.3

* scikit-learn == 1.3.0

* munkres == 1.1.4
  
**Data**

INPUT: attributes, adj, labels 

* STL-10 (HOG): This process is shown in [(STL-10)](https://github.com/mttk/STL10).

* Yale (HOG): This process is shown in [(Yale-FaceRecognition)](https://github.com/chenshen03/Yale-FaceRecognition). Raw data is uploaded as "yale_hog.npy".

* Others: Shown in `Data process.ipynb`.

## Usage
We provide a GPU&CPU version for all platforms (MacOS, Win, and Linux).

Just `python TDEC.py` . 

## Acknowledgement
We thank for their open sources of [(Awesome Deep Graph Clusteringv)](https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering) that contributes a lot to this community.
We also acknowledge the [(SwAV)](https://github.com/facebookresearch/swav).


