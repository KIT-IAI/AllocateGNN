import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, SAGEConv
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from SpatialAllocation.GNN.core.ModelConfig import ModelConfig
from SpatialAllocation.GNN.Layer.GraphEncoder import GraphEncoder
from SpatialAllocation.GNN.Layer.EdgeWeightLayer import DifferentiableEdgeWeighting
from SpatialAllocation.GNN.Layer.LossFunction.CombinedLoss import CombinedLoss
from torch_geometric.loader import DataLoader


class EdgeWeightSolver:
    """
    Self-supervised point allocation solver.
    """

    def __init__(self, config: ModelConfig):
        self.config = config

        # Set device
        if config.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config.device)

        self.encoder: Optional[GraphEncoder] = None
        self.edge_weighting_layer: Optional[DifferentiableEdgeWeighting] = None

    def init_model(self, train_dataloader: DataLoader):
        """
        Initialize model parameters.
        """
        # Get input dimensions from the training data loader
        first_batch = next(iter(train_dataloader))
        input_dims = {
            node_type: first_batch[node_type].x.shape[1]
            for node_type in first_batch.node_types
        }
        metadata = first_batch.metadata()
        self.encoder = GraphEncoder(input_dims, self.config, metadata).to(self.device)
        self.edge_weighting_layer = DifferentiableEdgeWeighting(self.config).to(self.device)

    def train_multi_graph(self, train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                          objective_weights: Optional[Dict[str, float]] = None):
        """
        Self-supervised training for multi-graph edge weight prediction (fully adapted to native
        heterogeneous models), including test/validation logic.
        """
        if objective_weights is None:
            objective_weights = {'entropy_regularization': 1.0}
        criterion = CombinedLoss(objective_weights, learnable=self.config.learnable).to(self.device)

        if self.config.debug:
            torch.autograd.set_detect_anomaly(True)

        if test_dataloader is None:
            print("No test data loader provided; training only.")

        self.init_model(train_dataloader)

        params = list(self.encoder.parameters()) + list(self.edge_weighting_layer.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        scheduler = None
        if self.config.use_scheduler:
            T_max = self.config.cosine_epochs if self.config.cosine_epochs is not None else self.config.epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=self.config.cosine_eta_min
            )

        # --- 1. Initialize loss dictionaries for training and testing ---
        train_losses = {key: [] for key in objective_weights.keys()}
        train_losses['total'] = []
        train_losses['learning_rate'] = []

        test_losses = {key: [] for key in objective_weights.keys()}
        test_losses['total'] = []

        best_test_loss = float('inf')  # Track best test loss

        print("Starting native heterogeneous graph model training and testing...")
        for epoch in range(self.config.epochs):
            start_time = time.time()
            epoch_grad_norms = []

            # ================= TRAINING PHASE =================
            self.encoder.train()
            self.edge_weighting_layer.train()

            epoch_train_losses = {key: 0.0 for key in train_losses.keys() if key != 'learning_rate'}
            num_train_batches = 0

            for batch_idx, batch_data in enumerate(train_dataloader):
                optimizer.zero_grad()
                batch_data = batch_data.to(self.device)
                embeddings_dict = self.encoder(batch_data.x_dict, batch_data.edge_index_dict)
                embeddings_s = embeddings_dict['source']
                embeddings_a = embeddings_dict['agent']
                edge_index_sa = batch_data['source', 'connects_to', 'agent'].edge_index
                edge_weights, edge_costs = self.edge_weighting_layer(embeddings_s, embeddings_a, edge_index_sa)

                metadata_for_loss = {
                    'num_s': batch_data['source'].num_nodes,
                    'num_a': batch_data['agent'].num_nodes,
                    'agent_features': batch_data['agent'].x,
                }
                # Only add data when it exists, for backward compatibility
                if hasattr(batch_data, 'agent'):
                    if hasattr(batch_data['agent'], 'demand'):
                        metadata_for_loss['agent_demand'] = batch_data['agent'].demand
                    if hasattr(batch_data['agent'], 'substation_idx'):
                        metadata_for_loss['agent_substation_map'] = batch_data['agent'].substation_idx
                if hasattr(batch_data, 'substation_y'):
                    metadata_for_loss['substation_y_true'] = batch_data.substation_y
                if hasattr(batch_data, 'num_substations'):
                    metadata_for_loss['num_substations'] = batch_data.num_substations
                if hasattr(batch_data, 'landuse_mapping_matrix'):
                    metadata_for_loss['landuse_mapping_matrix'] = batch_data.landuse_mapping_matrix
                if hasattr(batch_data, 'landuse_ratio'):
                    metadata_for_loss['landuse_ratio'] = batch_data.landuse_ratio


                total_loss, objectives = criterion(edge_weights, edge_index_sa, metadata_for_loss)

                total_loss.backward()
                total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=float('inf'))
                epoch_grad_norms.append(total_norm.item())
                if self.config.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(params, max_norm=self.config.clip_grad_norm)

                optimizer.step()

                epoch_train_losses['total'] += total_loss.item()
                for key, value in objectives.items():
                    if key in epoch_train_losses:
                        epoch_train_losses[key] += value.item()
                num_train_batches += 1

            # Compute and record training losses and learning rate
            for key in epoch_train_losses:
                train_losses[key].append(epoch_train_losses[key] / num_train_batches)
            train_losses['learning_rate'].append(optimizer.param_groups[0]['lr'])
            avg_epoch_train_loss = train_losses['total'][-1]
            avg_grad_norm = np.mean(epoch_grad_norms)

            if test_dataloader is not None:
                # ================= TESTING PHASE =================
                self.encoder.eval()
                self.edge_weighting_layer.eval()

                epoch_test_losses = {key: 0.0 for key in test_losses.keys()}
                num_test_batches = 0

                with torch.no_grad():  # No gradient computation during testing
                    for batch_data in test_dataloader:
                        batch_data = batch_data.to(self.device)
                        embeddings_dict = self.encoder(batch_data.x_dict, batch_data.edge_index_dict)
                        embeddings_s = embeddings_dict['source']
                        embeddings_a = embeddings_dict['agent']
                        edge_index_sa = batch_data['source', 'connects_to', 'agent'].edge_index
                        edge_weights, edge_costs = self.edge_weighting_layer(embeddings_s, embeddings_a, edge_index_sa)

                        # Also add landuse matrices for the testing phase
                        metadata_for_loss = {
                            'num_s': batch_data['source'].num_nodes,
                            'num_a': batch_data['agent'].num_nodes,
                            'agent_features': batch_data['agent'].x,
                        }
                        # Only add data when it exists, for backward compatibility
                        if hasattr(batch_data, 'agent'):
                            if hasattr(batch_data['agent'], 'demand'):
                                metadata_for_loss['agent_demand'] = batch_data['agent'].demand
                            if hasattr(batch_data['agent'], 'substation_idx'):
                                metadata_for_loss['agent_substation_map'] = batch_data['agent'].substation_idx
                        if hasattr(batch_data, 'substation_y'):
                            metadata_for_loss['substation_y_true'] = batch_data.substation_y
                        if hasattr(batch_data, 'num_substations'):
                            metadata_for_loss['num_substations'] = batch_data.num_substations
                        if hasattr(batch_data, 'landuse_mapping_matrix'):
                            metadata_for_loss['landuse_mapping_matrix'] = batch_data.landuse_mapping_matrix
                        if hasattr(batch_data, 'landuse_ratio'):
                            metadata_for_loss['landuse_ratio'] = batch_data.landuse_ratio

                        total_loss, objectives = criterion(edge_weights, edge_index_sa, metadata_for_loss)

                        epoch_test_losses['total'] += total_loss.item()
                        for key, value in objectives.items():
                            if key in epoch_test_losses:
                                epoch_test_losses[key] += value.item()
                        num_test_batches += 1

                # Compute and record test losses
                for key in epoch_test_losses:
                    test_losses[key].append(epoch_test_losses[key] / num_test_batches)
                avg_epoch_test_loss = test_losses['total'][-1]
            else:
                avg_epoch_test_loss = float('inf')

            # --- Post-epoch: scheduling, saving, and printing ---
            if scheduler is not None:
                scheduler.step()

            end_time = time.time()
            print(
                f"Epoch {epoch + 1}/{self.config.epochs} completed in {end_time - start_time:.2f}s | "
                f"Avg Train Loss: {avg_epoch_train_loss:.6f} | "
                f"Avg Test Loss: {avg_epoch_test_loss:.6f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                f"Avg Grad Norm: {avg_grad_norm:.6f}"
            )

            # Save best model based on test loss
            if test_dataloader is not None:
                if avg_epoch_test_loss < best_test_loss:
                    best_test_loss = avg_epoch_test_loss
                    checkpoint = {
                        'epoch': epoch,
                        'encoder_state_dict': self.encoder.state_dict(),
                        'edge_weighting_layer_state_dict': self.edge_weighting_layer.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_test_loss,
                    }
                    if scheduler:
                        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                    torch.save(checkpoint, self.config.save_path)
                    print(f'  -> Epoch {epoch + 1}, ** New best model saved (Test Loss: {best_test_loss:.6f}) **')
            else:
                if avg_epoch_train_loss < best_test_loss:
                    best_test_loss = avg_epoch_train_loss
                    checkpoint = {
                        'epoch': epoch,
                        'encoder_state_dict': self.encoder.state_dict(),
                        'edge_weighting_layer_state_dict': self.edge_weighting_layer.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_test_loss,
                    }
                    if scheduler:
                        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                    torch.save(checkpoint, self.config.save_path)
                    print(f'  -> Epoch {epoch + 1}, ** New best model saved (Test Loss: {best_test_loss:.6f}) **')

        # --- 3. Call the plotting function ---
        self._plot_training_curves({'train': train_losses, 'test': test_losses})
        print("Native heterogeneous graph training and testing complete!")

    def _plot_training_curves(self, all_losses: Dict[str, Dict[str, list]]):
        """
        Dynamically plot all recorded training loss curves.
        """
        # Get all loss key names (excluding learning rate)
        train_losses = all_losses.get('train', {})
        test_losses = all_losses.get('test', {})
        loss_keys = [key for key in train_losses.keys() if key != 'learning_rate' and len(train_losses[key]) > 0]

        # Calculate number of subplots needed
        num_plots = len(loss_keys)
        if num_plots == 0:
            print("No loss data to plot.")
            return

        plt.figure(figsize=(6 * num_plots, 5))

        # Dynamically create subplots
        for i, key in enumerate(loss_keys):
            plt.subplot(1, num_plots, i + 1)
            plt.plot(train_losses[key], label=f'Train {key.replace("_", " ").title()} Loss')
            # Check if the test set has records for this loss
            if key in test_losses and test_losses[key]:
                 plt.plot(test_losses[key], label=f'Test {key.replace("_", " ").title()} Loss')
            plt.title(f'{key.replace("_", " ").title()} Loss Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()

        plt.tight_layout()
        plt.show()

    def predict_edge_weights(self, data: HeteroData) -> pd.DataFrame:
        """
        Use the trained model to predict edge weights for the given heterogeneous graph data,
        and reorder results based on original indices.

        Args:
            data (HeteroData): A graph data object containing 'source' and 'agent' nodes and
                               edges between them. This object should contain an 'agent_index_map'
                               attribute that maps internal agent indices back to original GeoDataFrame indices.

        Returns:
            pd.DataFrame: A DataFrame containing predicted edge weights, ordered consistently
                          with the original agent GeoDataFrame indices.
        """
        if self.encoder is None or self.edge_weighting_layer is None:
            raise RuntimeError("Model has not been trained or loaded. Please call train_multi_graph or load a model first.")

        # Check that necessary metadata exists
        if not hasattr(data, 'agent_index_map'):
            raise ValueError("The input HeteroData object must contain an 'agent_index_map' attribute for weight reordering.")

        # Load best model state
        try:
            checkpoint = torch.load(self.config.save_path, map_location=self.device)
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.edge_weighting_layer.load_state_dict(checkpoint['edge_weighting_layer_state_dict'])
            print(f"Successfully loaded model from '{self.config.save_path}'.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file '{self.config.save_path}' not found. Please ensure the model has been trained and saved.")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

        self.encoder.eval()
        self.edge_weighting_layer.eval()

        with torch.no_grad():
            data = data.to(self.device)
            embeddings_dict = self.encoder(data.x_dict, data.edge_index_dict)
            embeddings_s = embeddings_dict['source']
            embeddings_a = embeddings_dict['agent']

            try:
                edge_index_sa = data['source', 'connects_to', 'agent'].edge_index
            except KeyError:
                raise ValueError("The input HeteroData object must contain edges of type ('source', 'connects_to', 'agent').")

            edge_weights, edge_costs = self.edge_weighting_layer(
                embeddings_s, embeddings_a, edge_index_sa
            )

            # --- Validation logic ---
            print("\n=== Edge Weight Prediction Statistics ===")
            num_s = data['source'].num_nodes

            # 1. Compute theoretical uniform distribution weights
            uniform_weights = torch.zeros_like(edge_weights)
            s_indices = edge_index_sa[0]
            # Use scatter_add to efficiently count the out-degree of each source
            ones = torch.ones_like(s_indices, dtype=torch.float)
            out_degree_s = torch.zeros(num_s, device=self.device, dtype=torch.float).scatter_add_(0, s_indices, ones)
            # Avoid division by zero
            uniform_weight_values = 1.0 / out_degree_s[s_indices].clamp(min=1)
            uniform_weights = uniform_weight_values

            # 2. Compute MAE (Mean Absolute Error)
            mae = torch.mean(torch.abs(edge_weights - uniform_weights))
            print(f"Overall MAE (vs. uniform distribution): {mae.item():.6f}")

            # 3. Verify that weight sums per source node equal 1
            weight_sums_per_s = torch.zeros(num_s, device=self.device).scatter_add_(0, s_indices, edge_weights)
            avg_weight_sum_error = torch.mean(torch.abs(weight_sums_per_s[out_degree_s > 0] - 1.0))
            print(f"Average weight sum error (should be close to 0): {avg_weight_sum_error.item():.6f}")
            print(f"Weight range: [{edge_weights.min().item():.4f}, {edge_weights.max().item():.4f}]")
            print("=" * 35 + "\n")
            # --- End of validation logic ---

            # --- Reordering logic ---
            # Get the local agent indices from graph edges
            agent_local_indices = edge_index_sa[1].cpu().numpy()

            # Use agent_index_map to convert local indices to original GeoDataFrame indices
            # agent_index_map should be a Series or dict where keys are local indices and values are original indices
            agent_index_map = data.agent_index_map
            original_gdf_indices = pd.Series(agent_index_map).iloc[agent_local_indices].values

            # Each agent belongs to exactly one source
            source_indices = edge_index_sa[0].cpu().numpy()

            # Return a DataFrame containing source_id, agent_id, and weight
            result_df = pd.DataFrame({
                'source_node_idx': source_indices,
                'agent_node_idx': agent_local_indices,
                'agent_original_idx': original_gdf_indices,
                'predicted_weight': edge_weights.cpu().numpy()
            })

            print("Generated DataFrame with original indices and weights.")

            return result_df