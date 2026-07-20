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
    Heterogeneous graph encoder: uses HeteroConv to learn low-dimensional
    representations for multiple node types.
    Purpose: learn information-rich embeddings separately for the 'source'
        and 'agent' node types.
    How it works:
        1. A set of linear layers (contained in an nn.ModuleDict) unifies
           input features of different dimensions into the same hidden dimension.
        2. Multiple layers of HeteroConv are applied. At each layer, HeteroConv
           calls a specified GNN layer (e.g. GCNConv) for each edge relation in
           the graph (e.g. 'source' -> 'agent') to perform message passing and
           aggregation.
        3. After multiple rounds of propagation, each node's final embedding
           contains not only neighborhood information but also implicitly
           encodes the types of its neighbors.
    Result: returns a dictionary containing a low-dimensional, dense embedding
        vector for each node type.
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

        # 2. Create the list of heterogeneous graph convolution layers
        self.convs = nn.ModuleList()
        # HGTConv requires a different initialization and forward pass path
        if self.conv_type == 'hgt':
            for _ in range(config.num_layers):
                # HGTConv needs the graph metadata for initialization
                conv = HGTConv(-1, config.hidden_dim, metadata, heads=self.config.gat_heads)
                self.convs.append(conv)
        else:
            # For other convolution types, wrap with HeteroConv
            for _ in range(config.num_layers):
                # MODIFICATION 1: create an independent conv instance for each edge type
                conv_dict = {}
                for edge_type in metadata[1]:
                    # edge_type is itself a tuple, e.g. ('source', 'connects_to', 'agent')
                    if self.conv_type == 'gcn':
                        # Use the tuple edge_type directly as the key
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

                # HeteroConv accepts a dict keyed by tuples and automatically manages the contained modules
                conv = HeteroConv(conv_dict, aggr='sum')
                self.convs.append(conv)

        # 3. Create a final linear layer to output the desired embedding_dim
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
        Forward pass: takes and returns dictionary-formatted data.
        """
        # 1. Apply the initial linear transform or embedding
        for node_type, x in x_dict.items():
            if x.size(1) == 0:  # If there are no input features
                # Create an all-zero index tensor to look up the embedding
                idx = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
                x_dict[node_type] = self.lin_dict[node_type](idx)
            else:
                x_dict[node_type] = self.lin_dict[node_type](x)

        # 2. Pass through multiple layers of heterogeneous convolution
        for i, conv in enumerate(self.convs):
            # Cache the input for the residual connection
            x_input_dict = x_dict
            x_dict = conv(x_dict, edge_index_dict)

            # Preserve node types that did not participate in message passing
            # (e.g. 'target' when it is not covered by the encoder's edges)
            for node_type in x_input_dict:
                if node_type not in x_dict:
                    x_dict[node_type] = x_input_dict[node_type]

            # Apply activation, normalization, and residual connection to each node type
            for node_type, x_out in x_dict.items():
                x = F.relu(x_out)
                x = self.norm_dict[node_type](x)

                # Add the residual connection (ensure the input dict has a matching node type)
                if node_type in x_input_dict:
                    x = x + x_input_dict[node_type]

                if i < len(self.convs) - 1:
                    x = self.dropout(x)
                x_dict[node_type] = x


        # 3. Apply the final output linear layer
        for node_type in x_dict.keys():
            x_dict[node_type] = self.out_lin(x_dict[node_type])

        # 4. L2 normalization and scaling (same as before, but applied to each tensor in the dict)
        for node_type in x_dict.keys():
            x_dict[node_type] = F.normalize(x_dict[node_type], p=2, dim=1)
            x_dict[node_type] = self.g * x_dict[node_type]

        return x_dict