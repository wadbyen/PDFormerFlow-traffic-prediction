# PDFormerFlow-traffic-prediction
# Introduction
PDFormerFlow is an improved traffic flow prediction model that builds upon PDFormer and incorporates Flow-by-Flow modeling principles. PDFormerFlow inherits PDFormer's advantages in capturing dynamic spatial dependencies and long-range spatial dependencies and introduces new features to optimize traffic flow prediction performance. At the same time, PDFormerFlow draws inspiration from flow-by-flow models to better model the temporal dynamics of traffic flow. This implementation adapts the core strengths of PDFormer to describe the intricate relationships in flow-by-flow traffic matrices. The model can capture both spatial relationships (based on physical proximity of OD endpoints) and temporal relationships to effectively predict traffic flows. 
# Key Features

    🚀 Flow-centric architecture: Models traffic as origin-destination (OD) pairs rather than individual nodes

    🌐 Dual relationship modeling: Captures both geometric (spatial) and semantic (pattern-based) flow relationships

    ⏱️ Temporal attention: Learns complex time-dependent patterns in traffic flows

    📈 Curriculum learning: Gradually increases prediction horizon during training

    ⚡ Efficient implementation: Optimized for GPU acceleration

# The Extention  

  PDFormerFlow builds upon the original PDFormer architecture with these key enhancements:

    Flow-based representation:

        Treats each origin-destination pair as a separate flow

        Models flow-to-flow relationships instead of node-to-node

    Flow positional encoding:

        Combines origin and destination node features

        Creates unique embeddings for each OD pair

    Flow relationship masks:

        Geometric mask based on physical distance between flows

        Semantic mask based on traffic pattern similarity

    Efficient attention mechanisms:

        Factorized attention over flows

        Pattern-enhanced geometric attention
# Dataset
Using publicly available datasets  to validate the proposed prediction method, such as  the Abilene and GÉANT datasets. They provide the statistical traffic volume data of the real network traffic trace from the American Research and Education Network (Abilene)  and the Europe Research and Education Network (GÉANT) .

| **Topology** | **Nodes** | **Flows** | **Links** | **Interval** | **Horizon** | **Records** |
| ------------ | --------- | --------- | --------- | ------------ | ----------- | ----------- |
| **Abilene**  | 12        | 144       | 15        | 5 min        | 6 months    | 48046       |
| **GÉANT**    | 23        | 529       | 38        | 15 min       | 4 months    | 10772       |


## Installation

python=3.7.9
torch==1.7.0
tsai==0.3.0
numpy==1.19.2

# Training
for example for Abilene dataset 

      python SPtransf-train.py --model PDFormerFlow --dataset abilene --epochs 100 --batch_size 16 --pre_len 1 --rounds 10

# References

  [1]  Jiang, J., Han, C., Zhao, W. X., & Cao, Z. (2023). PDFormer: Propagation Delay-aware Dynamic Long-range Transformer for Traffic Flow Prediction. AAAI. https://github.com/BUAABIGSCity/PDFormer

  [2] Zhang, Y., et al. (2022). Bayesian Graph Convolutional Network for Traffic Prediction. IEEE Transactions on Intelligent Transportation Systems.

 [3] Li, Y., et al. (2021). Dynamic Graph Convolutional Network for Traffic Prediction. IEEE Transactions on Intelligent Transportation Systems.
 [4] Weiping Zheng, et al. (2022).  Flow-by-flow traffic matrix prediction methods: Achieving accurate, adaptable, low cost results, Computer Communications. https://github.com/FreeeBird/Flow-By-Flow-Prediction.
