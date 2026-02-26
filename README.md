<p float="left">
    <img src="icon_kit.png" width="10%" hspace="20"/>
</p>

[![Python](https://img.shields.io/badge/Python-3.12.8-blue?logo=python)](https://www.python.org/downloads/release/python-3918/)
[![License](https://img.shields.io/badge/License-MIT-green?logo=opensource)](./LICENSE)

<h1 align="center">Improving Spatial Allocation for Energy System
Coupling with Graph Neural Networks</h1>

**Note**: *Last update on 2026.02.24*

<div align="left"> This repository is the official code of the paper <strong>"Improving Spatial Allocation for Energy System
Coupling with Graph Neural Networks"</strong></div>

## 1. Introduction

**AllocateGNN** proposes a Graph Neural Network (GNN)-based approach to improve the spatial allocation of electricity demand in energy systems. Traditional methods such as Voronoi tessellation assign demand to the nearest substation using simple geometric proximity, ignoring structural and contextual information. This work formulates the spatial allocation task as an edge weight prediction problem on a heterogeneous graph and uses self-supervised learning to produce more accurate, context-aware allocations.

The framework models the spatial allocation problem as a heterogeneous graph with three node types:
- **Source nodes**: Represent regional administrative areas (e.g., ITL3 regions) with known aggregate demand.
- **Agent nodes**: Represent grid cells with land-use features that serve as intermediaries for demand distribution.
- **Target nodes**: Represent substations where demand is physically consumed.

A GNN encoder learns node embeddings via message passing, and a differentiable edge weighting layer predicts allocation weights, which are optimized using a combination of self-supervised and weakly-supervised loss functions including entropy regularization, feature similarity loss, and land-use prediction loss. The real substation demand $D_t$ is used only for evaluation (RMSE/MAE), not as a training signal.

## 2. Project Structure

```
AllocateGNN/
├── SpatialAllocation/               # Main GNN-based spatial allocation module
│   ├── GNN/
│   │   ├── core/
│   │   │   ├── EdgeWeightSolver.py  # Self-supervised training and inference pipeline
│   │   │   └── ModelConfig.py       # Model hyperparameter configuration
│   │   ├── Layer/
│   │   │   ├── EdgeWeightLayer.py   # Differentiable edge weight prediction layer
│   │   │   ├── GraphEncoder.py      # Heterogeneous graph encoder (GCN/SAGE/GAT/GIN/HGT)
│   │   │   └── LossFunction/
│   │   │       ├── CombinedLoss.py  # Multi-objective loss with learnable weights
│   │   │       ├── LossFunction.py  # Individual loss function implementations
│   │   │       └── LossRegistry.py  # Loss function registry
│   │   └── utils/
│   │       └── GraphBuilder.py      # Graph construction and feature preprocessing
│   ├── utils/
│   │   ├── CalcuLanduse.py          # Land-use data fetching and proportion calculation
│   │   ├── CalcuOverlapDict.py      # Overlap computation utilities
│   │   ├── GenerateGrid.py          # Grid generation within boundary polygons
│   │   ├── GetOsmData.py            # OpenStreetMap data retrieval
│   │   └── color.py                 # Color utilities for visualization
│   └── voronoi/
│       ├── clustering/              # Clustering algorithms (DBSCAN, HDBSCAN, K-Means, etc.)
│       └── core/
│           ├── NearestAssignment.py  # Nearest neighbor assignment
│           └── SimpleVoronoi.py      # Simple Voronoi tessellation baseline
│
├── ClusterBasedVoronoi/             # Baseline: Cluster-based Voronoi approach
│   ├── clustering/                  # Clustering algorithms
│   ├── voronoi/
│   │   ├── prepare_pyomo_parameter.py  # Grid generation and influence calculation
│   │   ├── pyomo_based_voronoi.py      # Pyomo optimization-based Voronoi
│   │   └── simple_voronoi.py           # Simple Voronoi baseline
│   └── utils/
│       ├── analyze_point_distribution.py  # Point distribution analysis
│       ├── create_weights.py              # Weight creation utilities
│       ├── get_osm_data.py                # OSM data retrieval
│       └── merge_region.py                # Hierarchical region merging
│
├── notebooks/                       # Jupyter notebooks for experiments
│   ├── 001_british_data_overview_and_prepare.ipynb   # Data exploration and preparation
│   ├── 002_british_voronoi_simulate.ipynb            # Voronoi baseline experiments
│   ├── 003_additional_node_features_osm.ipynb        # OSM feature engineering
│   ├── 101_British_GNN_data_prepare.ipynb            # GNN data preparation
│   └── 102_British_GPM.ipynb                         # GNN training and evaluation
│
├── README.md
├── license.md
└── icon_kit.png
```

## 3. Key Components

### 3.1 Heterogeneous Graph Encoder (`GraphEncoder`)
Supports multiple GNN convolution types (`GCN`, `GraphSAGE`, `GAT`, `GIN`, `HGT`) wrapped in `HeteroConv` for heterogeneous graph learning. Includes residual connections, layer normalization, and L2-normalized embeddings with learnable scaling.

### 3.2 Differentiable Edge Weighting (`DifferentiableEdgeWeighting`)
Predicts edge weights using embedding distances gated by a learned MLP, followed by temperature-scaled grouped softmax via `scatter_softmax`. Ensures that weights from each source node sum to 1.

### 3.3 Loss Functions

**Self-supervised / Weakly-supervised losses (recommended):**
| Loss | Type | Description |
|------|------|-------------|
| `entropy_regularization` | Self-supervised | Encourages uniform edge weight distribution per source node |
| `feature_similarity_loss` | Self-supervised | Minimizes feature variance within each source's allocation cluster |
| `feature_consistency_loss` | Self-supervised | Encourages feature-similar agents under the same source to have similar weights |
| `landuse_prediction_loss` | Weakly-supervised | Computes KL divergence between predicted and actual land-use ratios (from OSM) |

**Supervised loss (uses real substation demand $D_t$ — not recommended for self/weakly-supervised settings):**
| Loss | Type | Description |
|------|------|-------------|
| `supervised_substation_demand_loss` | Supervised | Minimizes MAE between predicted and actual substation demand |

All losses support learnable uncertainty-based weighting for multi-task optimization.

### 3.4 Cluster-Based Voronoi Baseline
An alternative approach using clustering (DBSCAN, HDBSCAN, K-Means, etc.) combined with Voronoi tessellation. Supports optimization-based allocation via Pyomo with CIVD/IVD influence methods.

## 4. Installation

### Prerequisites
- Python 3.12+
- CUDA-compatible GPU (recommended)

### Dependencies
```bash
pip install torch torch-geometric torch-scatter
pip install geopandas shapely osmnx
pip install scipy numpy pandas matplotlib
pip install pyomo   # For cluster-based Voronoi baseline
```

For PyTorch Geometric, follow the [official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) to ensure compatibility with your CUDA version.

## 5. Usage

### 5.1 Data Preparation
Use the provided notebooks to prepare data:
1. **`001_british_data_overview_and_prepare.ipynb`**: Load and explore UK geographic, population, and GVA data.
2. **`003_additional_node_features_osm.ipynb`**: Fetch and process OpenStreetMap land-use features.
3. **`101_British_GNN_data_prepare.ipynb`**: Build heterogeneous graph data objects for GNN training.

### 5.2 Training the GNN Model
```python
from SpatialAllocation.GNN.core.ModelConfig import ModelConfig
from SpatialAllocation.GNN.core.EdgeWeightSolver import EdgeWeightSolver
from torch_geometric.loader import DataLoader

# Configure model
config = ModelConfig(
    hidden_dim=128,
    embedding_dim=64,
    num_layers=3,
    conv_type='sage',
    epochs=300,
    learning_rate=0.001,
    learnable=True
)

# Create data loaders
train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)

# Train
solver = EdgeWeightSolver(config)
solver.train_multi_graph(
    train_loader,
    test_loader,
    objective_weights={
        'entropy_regularization': 1.0,
        'landuse_prediction_loss': 1.0
    }
)
```

### 5.3 Inference
```python
# Predict edge weights for a new graph
result_df = solver.predict_edge_weights(test_data)
# result_df contains: source_node_idx, agent_node_idx, agent_original_idx, predicted_weight
```

### 5.4 Voronoi Baseline
See **`002_british_voronoi_simulate.ipynb`** for running the cluster-based Voronoi baseline with different clustering methods and influence calculation strategies.

[//]: # (<h2>6. Citation &#128221;</h2>)

[//]: # (<p>)

[//]: # (If you use this framework in your research, please consider citing our paper &#128221; and giving the repository a star &#11088;:)

[//]: # (</p>)

[//]: # ()
[//]: # (```bibTeX)

[//]: # (@misc{mu2026improving,)

[//]: # (      title={Improving Spatial Allocation for Energy System Coupling with Graph Neural Networks}, )

[//]: # (      author={Xuanhao Mu and Gökhan Demirel and Yuzhe Zhang and Jianlei Liu and Thorsten Schlachter and Veit Hagenmeyer},)

[//]: # (      year={2025},)

[//]: # (      eprint={2508.10587},)

[//]: # (      archivePrefix={arXiv},)

[//]: # (      primaryClass={cs.LG},)

[//]: # (      url={https://arxiv.org/abs/2508.10587}, )

[//]: # (})

[//]: # (```)
## License
This code is licensed under the **[MIT License](LICENSE)**.
For any issues or any intention of cooperation, please feel free to contact me at **[xuanhao.mu@kit.edu](xuanhao.mu@kit.edu)**.
