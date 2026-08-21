<p float="left">
    <img src="icon_kit.png" width="10%" hspace="20"/>
</p>

[![Python](https://img.shields.io/badge/Python-3.12.8-blue?logo=python)](https://www.python.org/downloads/release/python-3918/)
[![License](https://img.shields.io/badge/License-MIT-green?logo=opensource)](./LICENSE)

<h1 align="center">AllocateGNN</h1>

**Note**: *Last update on 2026.08.21*

<div align="left"> This repository is the official code of the paper <strong>"Improving Spatial Allocation for Energy System
Coupling with Graph Neural Networks"</strong> and of its follow-up study on mechanism-dependent antagonism of auxiliary information.</div>

> [!NOTE]
> ⭐ **Now available:** This repository additionally hosts the extended code for our follow-up paper
> *"Mechanism-Dependent Antagonism of Auxiliary Information in Substation-Level Load Disaggregation for Distribution Network Planning"* ([arXiv:2605.24491](https://arxiv.org/abs/2605.24491)) — see the [`SpatialAllocation/`](./SpatialAllocation) library and the multi-country case studies under [`StudyCase/`](./StudyCase).
>
> The exact code that accompanied the first (EPSR) paper is preserved at the git tag [`paper1-epsr`](https://github.com/KIT-IAI/AllocateGNN/tree/paper1-epsr).

> [!NOTE]
> **Publication-preparation release:** `SpatialPlacement/`, `StudyCase/Placement/`, and
> `study_materials/placement/` support the forthcoming manuscript
> *"Task- and scale-matched evaluation of spatial allocation proxies for network planning"*.
> The package reproduces manuscript numbers from derived frozen CSVs; it does
> not claim end-to-end lu5 model retraining because the held-out fields and
> checkpoints are unavailable.

## 1. Introduction

**AllocateGNN** proposes a Graph Neural Network (GNN)-based approach to improve the spatial allocation of electricity demand in energy systems. Traditional methods such as Voronoi tessellation assign demand to the nearest substation using simple geometric proximity, ignoring structural and contextual information. This work formulates the spatial allocation task as an edge weight prediction problem on a heterogeneous graph and uses self-supervised learning to produce more accurate, context-aware allocations.

The framework models the spatial allocation problem as a heterogeneous graph with three node types:
- **Source nodes**: Represent regional administrative areas (e.g., ITL3 regions) with known aggregate demand.
- **Agent nodes**: Represent grid cells with land-use features that serve as intermediaries for demand distribution.
- **Target nodes**: Represent substations where demand is physically consumed.

A GNN encoder learns node embeddings via message passing, and a differentiable edge weighting layer predicts allocation weights, which are optimized using a combination of self-supervised and weakly-supervised loss functions including entropy regularization, feature similarity loss, and land-use prediction loss. The real substation demand $D_t$ is used only for evaluation (RMSE/MAE), not as a training signal.

### Follow-up study: mechanism-dependent antagonism of auxiliary information

The follow-up paper studies substation-level load disaggregation when **auxiliary information** — night-time lights (NTL) and proximity priors — is injected into the learned weighter. A central finding is that auxiliary signals are *not* unconditionally helpful: depending on the **correction mechanism** (multiplicative reweighting, additive correction, conservation renormalization) two otherwise useful cues can **antagonize** each other. The effect is examined across three independent grids (Great Britain, Australia, and the German Börde region). The corresponding modules (`FeatureExtractor`, the `Weighter`/`Allocator` registries, and the NTL/proximity prior losses and correctors) and the full experiment suite live in this repository.

## 2. Project Structure

```
AllocateGNN/
├── SpatialAllocation/               # Core GNN-based spatial allocation library
│   ├── GNN/
│   │   ├── core/                    # Training/inference orchestration + model config
│   │   ├── Layer/                   # Graph encoder, edge-weight layer, gating, losses
│   │   │   └── LossFunction/        # Modular losses incl. NTL / Proximity prior losses
│   │   ├── Allocation/              # End-to-end allocation graph/solver/losses
│   │   └── utils/                   # Graph construction and feature preprocessing
│   ├── Allocator/                   # Allocation methods (Voronoi, CIVD, …) via a registry
│   ├── Weighter/                    # Weighter methods (heterogeneous GNN, GPM, uniform, …)
│   ├── FeatureExtractor/            # Feature pipeline: fetchers / extractors / correctors
│   │   ├── fetchers/                # Sentinel-2, WorldCover, OSM, NTL data fetchers
│   │   ├── extractors/              # Spectral / land-use / WorldCover / NTL extractors
│   │   └── correctors/              # NTL and proximity feature correctors
│   └── utils/                       # Imagery, network distance, spectral indices, CNN features
│
├── ClusterBasedVoronoi/             # Baseline: cluster-based Voronoi approach
│
├── SpatialPlacement/                # Reusable planning-task evaluation library
│   ├── core/                        # Reconstruction, siting, sizing, connection bounds
│   └── pipeline/                    # Candidate generation, metrics and solvers
│
├── StudyCase/                       # Multi-country case studies (follow-up paper)
│   ├── British/                     # Great Britain case study (notebooks + training scripts)
│   ├── British_weighter_experiments/# Antagonism experiment suite (main experiments)
│   ├── Australia/                   # Australia (Ausgrid) case study
│   ├── Germany/                     # German Börde case study
│   └── Placement/                   # Task/scale-matched planning evaluation
│
├── study_materials/
│   └── placement/                   # Manuscript-scoped frozen numerical evidence
│
├── requirements.txt
├── README.md
├── license.md
└── icon_kit.png
```

> **Note on data:** raw and intermediate datasets, cached artifacts, and model
> products are not distributed. The placement release includes only the small
> derived result CSVs needed to reproduce the manuscript's numerical tables,
> statistics, and quantitative figure inputs.

## 3. Key Components

### 3.1 Heterogeneous Graph Encoder (`GraphEncoder`)
Supports multiple GNN convolution types (`GCN`, `GraphSAGE`, `GAT`, `GIN`, `HGT`) wrapped in `HeteroConv` for heterogeneous graph learning. Includes residual connections, layer normalization, and L2-normalized embeddings with learnable scaling.

### 3.2 Differentiable Edge Weighting
Predicts edge weights using embedding distances gated by a learned MLP, followed by temperature-scaled grouped softmax via `scatter_softmax`. Ensures that weights from each source node sum to 1.

### 3.3 Loss Functions
Modular loss system with learnable uncertainty-based weighting for multi-task optimization. Self-supervised / weakly-supervised objectives include entropy regularization, feature-similarity/consistency losses, and a land-use (KL) prediction loss. The follow-up study adds **NTL** and **proximity** prior losses that inject auxiliary spatial information into the weighter.

### 3.4 Allocator / Weighter / FeatureExtractor registries
The `Allocator`, `Weighter`, and `FeatureExtractor` subpackages expose registry-based interfaces so that allocation methods (Voronoi, CIVD, …), weighters (heterogeneous GNN, GPM, uniform, …), and feature fetchers/extractors/correctors can be selected and composed by name.

### 3.5 Cluster-Based Voronoi Baseline
An alternative approach using clustering (DBSCAN, HDBSCAN, K-Means, etc.) combined with Voronoi tessellation. Supports optimization-based allocation via Pyomo with CIVD/IVD influence methods.

### 3.6 Spatial placement evaluation
The top-level `SpatialPlacement` package provides reusable peak-reconstruction,
p-median siting, rule-based sizing, connection-proxy, uncertainty-bound, and
statistical utilities. Paper-specific case descriptions remain under
`StudyCase/Placement/`.

## 4. Case Studies (`StudyCase/`)

The follow-up paper is reproduced through four case-study folders. Each contains the analysis notebooks and the training/experiment scripts (numbered by execution order); raw-data ingestion and feature construction are excluded.

| Folder | Grid | Content |
|--------|------|---------|
| `British/` | Great Britain | Base case study: static allocation, degradation analysis, CIVD, GNN training with NTL/proximity priors. |
| `British_weighter_experiments/` | Great Britain | Main experiment suite for the antagonism study (main results, significance, strength sweeps, mechanism isolation, robustness, r-series ablations). |
| `Australia/` | Ausgrid (AU) | Static baselines, GNN training, statistical evaluation, feature-fusion training. |
| `Germany/` | Börde (DE) | Börde training, results tables, LOOCV, additive-correction matrix, pandapower downstream. |
| `Placement/` | Britain + Australia | Task- and scale-matched reconstruction, siting, sizing, connection, and conditional-bound evaluation. |

## 5. Installation

### Prerequisites
- Python 3.12+
- CUDA-compatible GPU (recommended)

### Dependencies
```bash
pip install -r requirements.txt
```

For PyTorch Geometric and its compiled extensions (`torch-scatter`, `torch-sparse`), follow the [official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) to match your PyTorch and CUDA version. Pyomo additionally requires an external solver (e.g. CBC or GLPK) for the cluster-based Voronoi baseline.

## 6. Usage

### 6.1 Training the GNN Model
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

### 6.2 Inference
```python
# Predict edge weights for a new graph
result_df = solver.predict_edge_weights(test_data)
# result_df contains: source_node_idx, agent_node_idx, agent_original_idx, predicted_weight
```

### 6.3 Reproducing the case studies
Each `StudyCase/<grid>/` folder holds numbered scripts and notebooks. The training scripts (e.g. `005_kfold_prior_training.py` in `British_weighter_experiments/`) and the downstream experiment scripts expect processed inputs to be available locally.

### 6.4 Reproducing the placement-paper numbers

```bash
python -m SpatialPlacement.reproduce_paper_numbers --verify
```

This verifies the manuscript-scoped hashes and regenerates the numerical
summary from `study_materials/placement/`.

## 7. Citation 📝

If you use this framework in your research, please consider citing our papers 📝 and giving the repository a star ⭐:

```bibtex
@article{Mu2026Improving,
      author={Mu, Xuanhao and Geiges, Jakob and Liu, Nan and Schlachter, Thorsten and Hagenmeyer, Veit},
      title={Improving spatial allocation for energy system coupling with graph neural networks},
      journal={Electric Power Systems Research},
      volume={262},
      pages={113519},
      year={2027},
      issn={0378-7796},
      doi={10.1016/j.epsr.2026.113519},
      url={https://www.sciencedirect.com/science/article/pii/S0378779626008126}
}

@article{Mu2026Antagonism,
      author={Mu, Xuanhao and Thota, Kundan and Liu, Nan and Schlachter, Thorsten and Hagenmeyer, Veit},
      title={Mechanism-Dependent Antagonism of Auxiliary Information in Substation-Level Load Disaggregation for Distribution Network Planning},
      journal={arXiv preprint arXiv:2605.24491},
      year={2026},
      url={https://arxiv.org/abs/2605.24491}
}
```

## License
This code is licensed under the **[MIT License](LICENSE)**.
For any issues or any intention of cooperation, please feel free to contact me at **[xuanhao.mu@kit.edu](xuanhao.mu@kit.edu)**.
