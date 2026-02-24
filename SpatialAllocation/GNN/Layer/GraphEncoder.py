import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, HeteroConv, SAGEConv, GATConv,
    GINConv, HGTConv
)
from SpatialAllocation.GNN.core.ModelConfig import ModelConfig
from typing import Dict, List, Tuple


class GraphEncoder(nn.Module):
    """
    Heterogeneous graph encoder: uses HeteroConv to learn low-dimensional representations for multi-type nodes.

    Purpose:
        Learn informative embeddings for 'source' and 'agent' node types separately.

    How it works:
        1. Uses nn.ModuleDict with linear layers to project input features of different dimensions
           into a unified hidden dimension.
        2. Uses multiple HeteroConv layers. In each layer, HeteroConv invokes a specified GNN layer
           (e.g., GCNConv) for each edge relation type (e.g., 'source' -> 'agent') to perform
           message passing and aggregation.
        3. After multiple propagation layers, each node's final embedding incorporates neighborhood
           information and implicitly encodes neighbor type information.

    Result:
        Returns a dictionary containing low-dimensional, dense embedding vectors for each node type.
    """

    def __init__(self, input_dims: Dict[str, int], config: ModelConfig, metadata: Tuple[List[str], List[Tuple[str, str, str]]]):
        super(GraphEncoder, self).__init__()
        self.config = config
        self.conv_type = self.config.conv_type.lower()

        # Validate conv_type
        supported_types = ['gcn', 'sage', 'gat', 'gin', 'hgt']
        if self.conv_type not in supported_types:
            raise ValueError(f"Unsupported conv_type: {self.conv_type}. Supported types: {supported_types}")

        # 1. Create an initial linear projection layer for each node type to unify dimensions
        self.lin_dict = nn.ModuleDict()
        for node_type, in_dim in input_dims.items():
            # If a node type has no features, create a learnable embedding to represent it
            if in_dim == 0:
                self.lin_dict[node_type] = nn.Embedding(1, config.hidden_dim)
            else:
                self.lin_dict[node_type] = nn.Linear(in_dim, config.hidden_dim)

        # 2. Create heterogeneous graph convolution layer list
        self.convs = nn.ModuleList()
        # HGTConv requires a different initialization and forward propagation path
        if self.conv_type == 'hgt':
            for _ in range(config.num_layers):
                # HGTConv needs graph metadata for initialization
                conv = HGTConv(-1, config.hidden_dim, metadata, heads=self.config.gat_heads)
                self.convs.append(conv)
        else:
            # For other convolution types, wrap with HeteroConv
            for _ in range(config.num_layers):
                # Create independent convolution instances for each edge type
                conv_dict = {}
                for edge_type in metadata[1]:
                    # edge_type is already a tuple, e.g., ('source', 'connects_to', 'agent')
                    if self.conv_type == 'gcn':
                        conv_dict[edge_type] = GCNConv(-1, config.hidden_dim, add_self_loops=False)
                    elif self.conv_type == 'sage':
                        conv_dict[edge_type] = SAGEConv(-1, config.hidden_dim)
                    elif self.conv_type == 'gat':
                        heads = self.config.gat_heads
                        if config.hidden_dim % heads != 0:
                            raise ValueError(
                                f"For GAT, hidden_dim ({config.hidden_dim}) must be divisible by heads ({heads}).")
                        out_channels = config.hidden_dim // heads
                        conv_dict[edge_type] = GATConv(-1, out_channels, heads=heads, add_self_loops=False)
                    elif self.conv_type == 'gin':
                        mlp = nn.Sequential(
                            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
                            nn.ReLU(),
                            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                        )
                        conv_dict[edge_type] = GINConv(nn=mlp, train_eps=True)

                # HeteroConv accepts a dict keyed by tuples and manages the modules automatically
                conv = HeteroConv(conv_dict, aggr='sum')
                self.convs.append(conv)

        # 3. Create a final linear layer for outputting the desired embedding_dim
        self.out_lin = nn.Linear(config.hidden_dim, config.embedding_dim)

        # 4. Learnable scaling factor g (same as before)
        self.g = nn.Parameter(torch.ones(config.embedding_dim))
        self.dropout = nn.Dropout(0.1)

        # 5. Create a LayerNorm layer for each node type
        self.norm_dict = nn.ModuleDict()
        for node_type in input_dims.keys():
            self.norm_dict[node_type] = nn.LayerNorm(config.hidden_dim)

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[str, torch.Tensor]) -> Dict[
        str, torch.Tensor]:
        """
        Forward propagation function; receives and returns dictionary-formatted data.
        """
        # 1. Apply initial linear transformation or embedding
        for node_type, x in x_dict.items():
            if x.size(1) == 0:  # If no input features
                # Create a zero index tensor to query the embedding
                idx = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
                x_dict[node_type] = self.lin_dict[node_type](idx)
            else:
                x_dict[node_type] = self.lin_dict[node_type](x)

        # 2. Pass through multiple heterogeneous convolution layers
        for i, conv in enumerate(self.convs):
            # Cache input for residual connection
            x_input_dict = x_dict
            x_dict = conv(x_dict, edge_index_dict)

            # Apply activation, normalization, and residual connection for each node type
            for node_type, x_out in x_dict.items():
                x = F.relu(x_out)
                x = self.norm_dict[node_type](x)

                # Add residual connection (ensure the input dict has the corresponding node type)
                if node_type in x_input_dict:
                    x = x + x_input_dict[node_type]

                if i < len(self.convs) - 1:
                    x = self.dropout(x)
                x_dict[node_type] = x


        # 3. Apply the final output linear layer
        for node_type in x_dict.keys():
            x_dict[node_type] = self.out_lin(x_dict[node_type])

        # 4. L2 normalization and scaling (same as before, but operates on each tensor in the dict)
        for node_type in x_dict.keys():
            x_dict[node_type] = F.normalize(x_dict[node_type], p=2, dim=1)
            x_dict[node_type] = self.g * x_dict[node_type]

        return x_dict