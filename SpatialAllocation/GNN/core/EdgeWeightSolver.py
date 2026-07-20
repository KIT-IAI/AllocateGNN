import time
import json
import dataclasses
import datetime
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
from SpatialAllocation.GNN.Layer.ProjectionHead import SpectralProjectionHead, MacroReconstructionHead
from SpatialAllocation.GNN.Layer.AgentGating import AgentGating
from torch_geometric.loader import DataLoader


class EdgeWeightSolver:
    """
    Self-supervised point allocation solver (Stage 1: source -> agent weight prediction)
    """

    def __init__(self, config: ModelConfig):
        self.config = config

        # Set the device
        if config.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config.device)

        self.encoder: Optional[GraphEncoder] = None
        self.edge_weighting_layer: Optional[DifferentiableEdgeWeighting] = None
        self.projection_head: Optional[SpectralProjectionHead] = None
        self.recon_head: Optional[MacroReconstructionHead] = None
        self.agent_gating: Optional[AgentGating] = None

    def init_model(self, train_dataloader: DataLoader, objective_weights: Optional[Dict[str, float]] = None):
        """
        Initialize model parameters. Whether the spectral/gating modules are
        created is inferred from objective_weights:
        - Contains one of 'reconstruction'/'diversity'/'gate' -> initializes the projection head and reconstruction head
        - Contains 'gate' -> additionally initializes AgentGating
        """
        # Get the input dimensions from the training data loader
        first_batch = next(iter(train_dataloader))
        input_dims = {
            node_type: first_batch[node_type].x.shape[1]
            for node_type in first_batch.node_types
            if hasattr(first_batch[node_type], 'x')
        }
        metadata = first_batch.metadata()

        self.encoder = GraphEncoder(input_dims, self.config, metadata).to(self.device)
        self.edge_weighting_layer = DifferentiableEdgeWeighting(self.config).to(self.device)

        # Infer module enablement from weights
        _SPECTRAL_LOSSES = {'reconstruction', 'diversity', 'gate'}
        _use_spectral = bool(objective_weights and _SPECTRAL_LOSSES & set(objective_weights))
        _use_gating = bool(objective_weights and 'gate' in objective_weights)

        # Phase A: spectral feature projection head + reconstruction head
        if _use_spectral:
            self.projection_head = SpectralProjectionHead(
                d_spec=self.config.spectral_feature_dim,
                d_z=self.config.projection_bottleneck_dim,
                K=self.config.macro_attribute_dim,
            ).to(self.device)
            self.recon_head = MacroReconstructionHead(
                d_z=self.config.projection_bottleneck_dim,
                M=self.config.macro_attribute_dim,
                head_type=self.config.recon_head_type,
            ).to(self.device)

        # Phase A: agent gating (depends on spectral features, so _use_spectral must also be True)
        if _use_gating:
            self.agent_gating = AgentGating(
                d_z=self.config.projection_bottleneck_dim,
                bias_init=self.config.gate_bias_init,
            ).to(self.device)

    def train_multi_graph(self, train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                          objective_weights: Optional[Dict[str, float]] = None,
                          eval_every: int = 1):
        """
        Self-supervised training for multi-graph edge weight prediction
        (fully adapted to the native heterogeneous model), including the
        test/validation logic.
        """
        if objective_weights is None:
            objective_weights = {'entropy_regularization': 1.0}
        criterion = CombinedLoss(objective_weights, learnable=self.config.learnable).to(self.device)

        if self.config.debug:
            torch.autograd.set_detect_anomaly(True)

        if test_dataloader is None:
            print("No test data loader provided; training only.")

        self.init_model(train_dataloader, objective_weights)

        params = list(self.encoder.parameters()) + list(self.edge_weighting_layer.parameters())
        if self.projection_head is not None:
            params += list(self.projection_head.parameters())
        if self.recon_head is not None:
            params += list(self.recon_head.parameters())
        if self.agent_gating is not None:
            params += list(self.agent_gating.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        scheduler = None
        if self.config.use_scheduler:
            T_max = self.config.cosine_epochs if self.config.cosine_epochs is not None else self.config.epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=self.config.cosine_eta_min
            )

        # --- 1. Initialize the loss dictionaries for training and testing separately ---
        train_losses = {key: [] for key in objective_weights.keys()}
        train_losses['total'] = []
        train_losses['learning_rate'] = []

        test_losses = {key: [] for key in objective_weights.keys()}
        test_losses['total'] = []

        best_test_loss = float('inf')  # Used to track the best test loss
        best_epoch = -1

        train_start_time = datetime.datetime.now()
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

                # Phase A: spectral features -> projection head -> gating
                gate_values = None
                z_a = None
                T_hat = None
                recon_pred = None
                if self.projection_head is not None:
                    # Extract the spectral portion from the agent features (last spectral_feature_dim columns)
                    spectral_feat = batch_data['agent'].x[:, -self.config.spectral_feature_dim:]
                    z_a, T_hat = self.projection_head(spectral_feat)
                    recon_pred = self.recon_head(z_a)

                    # Gating
                    if self.agent_gating is not None:
                        gate_values = self.agent_gating(z_a)

                edge_weights, edge_costs = self.edge_weighting_layer(
                    embeddings_s, embeddings_a, edge_index_sa, gate_values=gate_values
                )

                metadata_for_loss = {
                    'num_s': batch_data['source'].num_nodes,
                    'num_a': batch_data['agent'].num_nodes,
                    'agent_features': batch_data['agent'].x,
                }
                # Only add these when the data is present, to preserve backward compatibility
                if hasattr(batch_data, 'agent'):
                    if hasattr(batch_data['agent'], 'demand'):
                        metadata_for_loss['agent_demand'] = batch_data['agent'].demand
                if hasattr(batch_data, 'landuse_mapping_matrix'):
                    metadata_for_loss['landuse_mapping_matrix'] = batch_data.landuse_mapping_matrix
                if hasattr(batch_data, 'landuse_ratio'):
                    metadata_for_loss['landuse_ratio'] = batch_data.landuse_ratio

                # Phase A: add metadata related to spectral features
                if z_a is not None:
                    metadata_for_loss['T_hat'] = T_hat
                    metadata_for_loss['recon_predictions'] = recon_pred
                    metadata_for_loss['gate_lambda'] = self.config.gate_lambda
                    if gate_values is not None:
                        metadata_for_loss['gate_values'] = gate_values
                    if hasattr(batch_data, 'macro_attributes'):
                        metadata_for_loss['macro_attributes'] = batch_data.macro_attributes
                    if hasattr(batch_data, 'attribute_stds'):
                        metadata_for_loss['attribute_stds'] = batch_data.attribute_stds

                # agent_adj_pairs is used by FeatureConsistencyLoss
                if hasattr(batch_data, 'agent_adj_pairs'):
                    metadata_for_loss['agent_adj_pairs'] = batch_data.agent_adj_pairs

                # NTL / Proximity prior data
                if hasattr(batch_data['agent'], 'ntl_values'):
                    metadata_for_loss['agent_ntl'] = batch_data['agent'].ntl_values
                if hasattr(batch_data['agent'], 'proximity_scores'):
                    metadata_for_loss['agent_proximity'] = batch_data['agent'].proximity_scores
                if hasattr(batch_data['agent'], 'rci_mask'):
                    metadata_for_loss['agent_rci_mask'] = batch_data['agent'].rci_mask

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

            # Compute and record the training loss and learning rate
            for key in epoch_train_losses:
                train_losses[key].append(epoch_train_losses[key] / num_train_batches)
            train_losses['learning_rate'].append(optimizer.param_groups[0]['lr'])
            avg_epoch_train_loss = train_losses['total'][-1]
            avg_grad_norm = np.mean(epoch_grad_norms)

            # eval_every control: only run test evaluation at the specified interval or on the final epoch
            is_eval_epoch = (epoch % eval_every == 0) or (epoch == self.config.epochs - 1)

            if test_dataloader is not None and is_eval_epoch:
                # ================= TESTING PHASE =================
                self.encoder.eval()
                self.edge_weighting_layer.eval()

                epoch_test_losses = {key: 0.0 for key in test_losses.keys()}
                num_test_batches = 0

                with torch.no_grad():  # No gradients are computed during the testing phase
                    for batch_data in test_dataloader:
                        batch_data = batch_data.to(self.device)
                        embeddings_dict = self.encoder(batch_data.x_dict, batch_data.edge_index_dict)
                        embeddings_s = embeddings_dict['source']
                        embeddings_a = embeddings_dict['agent']
                        edge_index_sa = batch_data['source', 'connects_to', 'agent'].edge_index

                        # Phase A: spectral features -> projection head -> gating (testing phase)
                        gate_values_test = None
                        z_a_test = None
                        T_hat_test = None
                        recon_pred_test = None
                        if self.projection_head is not None:
                            spectral_feat = batch_data['agent'].x[:, -self.config.spectral_feature_dim:]
                            z_a_test, T_hat_test = self.projection_head(spectral_feat)
                            recon_pred_test = self.recon_head(z_a_test)
                            if self.agent_gating is not None:
                                gate_values_test = self.agent_gating(z_a_test)

                        edge_weights, edge_costs = self.edge_weighting_layer(
                            embeddings_s, embeddings_a, edge_index_sa, gate_values=gate_values_test
                        )

                        metadata_for_loss = {
                            'num_s': batch_data['source'].num_nodes,
                            'num_a': batch_data['agent'].num_nodes,
                            'agent_features': batch_data['agent'].x,
                        }
                        if hasattr(batch_data, 'agent'):
                            if hasattr(batch_data['agent'], 'demand'):
                                metadata_for_loss['agent_demand'] = batch_data['agent'].demand
                        if hasattr(batch_data, 'landuse_mapping_matrix'):
                            metadata_for_loss['landuse_mapping_matrix'] = batch_data.landuse_mapping_matrix
                        if hasattr(batch_data, 'landuse_ratio'):
                            metadata_for_loss['landuse_ratio'] = batch_data.landuse_ratio

                        if z_a_test is not None:
                            metadata_for_loss['T_hat'] = T_hat_test
                            metadata_for_loss['recon_predictions'] = recon_pred_test
                            metadata_for_loss['gate_lambda'] = self.config.gate_lambda
                            if gate_values_test is not None:
                                metadata_for_loss['gate_values'] = gate_values_test
                            if hasattr(batch_data, 'macro_attributes'):
                                metadata_for_loss['macro_attributes'] = batch_data.macro_attributes
                            if hasattr(batch_data, 'attribute_stds'):
                                metadata_for_loss['attribute_stds'] = batch_data.attribute_stds

                        # agent_adj_pairs is used by FeatureConsistencyLoss
                        if hasattr(batch_data, 'agent_adj_pairs'):
                            metadata_for_loss['agent_adj_pairs'] = batch_data.agent_adj_pairs

                        # NTL / Proximity prior data
                        if hasattr(batch_data['agent'], 'ntl_values'):
                            metadata_for_loss['agent_ntl'] = batch_data['agent'].ntl_values
                        if hasattr(batch_data['agent'], 'proximity_scores'):
                            metadata_for_loss['agent_proximity'] = batch_data['agent'].proximity_scores
                        if hasattr(batch_data['agent'], 'rci_mask'):
                            metadata_for_loss['agent_rci_mask'] = batch_data['agent'].rci_mask

                        total_loss, objectives = criterion(edge_weights, edge_index_sa, metadata_for_loss)

                        epoch_test_losses['total'] += total_loss.item()
                        for key, value in objectives.items():
                            if key in epoch_test_losses:
                                epoch_test_losses[key] += value.item()
                        num_test_batches += 1

                # Compute and record the test loss
                for key in epoch_test_losses:
                    test_losses[key].append(epoch_test_losses[key] / num_test_batches)
                avg_epoch_test_loss = test_losses['total'][-1]
            elif test_dataloader is not None:
                # Non-evaluation epoch: reuse the previous test loss
                for key in test_losses:
                    if test_losses[key]:
                        test_losses[key].append(test_losses[key][-1])
                    else:
                        test_losses[key].append(float('inf'))
                avg_epoch_test_loss = test_losses['total'][-1]
            else:
                avg_epoch_test_loss = float('inf')

            # --- Scheduling, saving, and printing after the epoch ends ---
            if scheduler is not None:
                scheduler.step()

            end_time = time.time()
            log_parts = [
                f"Epoch {epoch + 1}/{self.config.epochs} completed in {end_time - start_time:.2f}s",
                f"Train Loss: {avg_epoch_train_loss:.6f}",
            ]
            if test_dataloader is not None:
                log_parts.append(f"Test Loss: {avg_epoch_test_loss:.6f}")
            log_parts.append(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            log_parts.append(f"Grad Norm: {avg_grad_norm:.6f}")
            print(" | ".join(log_parts))

            # Save the best model based on loss: use test loss when test_dl is present, otherwise use train loss
            ref_loss = avg_epoch_test_loss if test_dataloader is not None else avg_epoch_train_loss
            ref_label = "Test" if test_dataloader is not None else "Train"
            if ref_loss < best_test_loss:
                best_test_loss = ref_loss
                best_epoch = epoch + 1
                checkpoint = self._build_checkpoint(epoch, best_test_loss, optimizer, scheduler)
                torch.save(checkpoint, self.config.save_path)
                print(f'  -> Epoch {epoch + 1}, ** best model saved ({ref_label} Loss: {best_test_loss:.6f}) **')

        # --- 3. Save the training log as JSON ---
        self._save_training_log(
            train_start_time=train_start_time,
            objective_weights=objective_weights,
            train_losses=train_losses,
            test_losses=test_losses,
            best_epoch=best_epoch,
            best_loss=best_test_loss,
        )

        # --- 4. Call the plotting function ---
        self._plot_training_curves({'train': train_losses, 'test': test_losses})
        print("Native heterogeneous graph training and testing complete!")

    def _save_training_log(
        self,
        train_start_time: datetime.datetime,
        objective_weights: Dict[str, float],
        train_losses: Dict[str, list],
        test_losses: Dict[str, list],
        best_epoch: int,
        best_loss: float,
    ):
        """Save the training parameters and per-epoch losses as a JSON log file."""
        train_end_time = datetime.datetime.now()
        duration = (train_end_time - train_start_time).total_seconds()

        log = {
            "run_id": train_start_time.strftime("%Y%m%d_%H%M%S"),
            "start_time": train_start_time.isoformat(),
            "end_time": train_end_time.isoformat(),
            "duration_seconds": round(duration, 2),
            "config": dataclasses.asdict(self.config),
            "objective_weights": objective_weights,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "train_losses": train_losses,
            "test_losses": test_losses,
        }

        # The log path is in the same directory as the model file, with suffix changed to _training_log.json
        save_path = self.config.save_path
        if save_path.endswith('.pth'):
            log_path = save_path[:-4] + '_training_log.json'
        else:
            log_path = save_path + '_training_log.json'

        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        print(f"Training log saved to: {log_path}")

    def _build_checkpoint(self, epoch, loss, optimizer, scheduler=None):
        """Build a checkpoint dictionary containing the state of all modules."""
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        if self.edge_weighting_layer is not None:
            checkpoint['edge_weighting_layer_state_dict'] = self.edge_weighting_layer.state_dict()
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if self.projection_head is not None:
            checkpoint['projection_head_state_dict'] = self.projection_head.state_dict()
        if self.recon_head is not None:
            checkpoint['recon_head_state_dict'] = self.recon_head.state_dict()
        if self.agent_gating is not None:
            checkpoint['agent_gating_state_dict'] = self.agent_gating.state_dict()
        return checkpoint

    def _plot_training_curves(self, all_losses: Dict[str, Dict[str, list]]):
        """
        Dynamically plot all recorded training loss curves.
        """
        # Get all loss key names (except learning rate)
        train_losses = all_losses.get('train', {})
        test_losses = all_losses.get('test', {})
        loss_keys = [key for key in train_losses.keys() if key != 'learning_rate' and len(train_losses[key]) > 0]

        # Compute the number of subplots needed
        num_plots = len(loss_keys)
        if num_plots == 0:
            print("No loss data available to plot.")
            return

        plt.figure(figsize=(6 * num_plots, 5))

        # Dynamically create subplots
        for i, key in enumerate(loss_keys):
            plt.subplot(1, num_plots, i + 1)
            plt.plot(train_losses[key], label=f'Train {key.replace("_", " ").title()} Loss')
            # Check whether the test set has a record for this loss
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
        Predict source -> agent edge weights W_sa using the trained model.
        """
        if self.encoder is None or self.edge_weighting_layer is None:
            raise RuntimeError("The model has not been trained or loaded. Call train_multi_graph first, or load a model.")

        # Check that the required metadata is present
        if not hasattr(data, 'agent_index_map'):
            raise ValueError("The input HeteroData object must contain an 'agent_index_map' attribute for weight reordering.")

        # Load the best model's state
        self._load_checkpoint()

        self.encoder.eval()
        self.edge_weighting_layer.eval()
        if self.projection_head is not None:
            self.projection_head.eval()
        if self.recon_head is not None:
            self.recon_head.eval()
        if self.agent_gating is not None:
            self.agent_gating.eval()

        with torch.no_grad():
            data = data.to(self.device)
            embeddings_dict = self.encoder(data.x_dict, data.edge_index_dict)
            embeddings_s = embeddings_dict['source']
            embeddings_a = embeddings_dict['agent']

            try:
                edge_index_sa = data['source', 'connects_to', 'agent'].edge_index
            except KeyError:
                raise ValueError("The input HeteroData object must contain edges of type ('source', 'connects_to', 'agent').")

            # Phase A: also run the projection head and gating during inference
            gate_values_pred = None
            T_hat_pred = None
            if self.projection_head is not None:
                spectral_feat = data['agent'].x[:, -self.config.spectral_feature_dim:]
                z_a_pred, T_hat_pred = self.projection_head(spectral_feat)
                if self.agent_gating is not None:
                    gate_values_pred = self.agent_gating(z_a_pred)

            edge_weights, edge_costs = self.edge_weighting_layer(
                embeddings_s, embeddings_a, edge_index_sa, gate_values=gate_values_pred
            )

            # --- Validation logic ---
            print("\n=== Edge weight prediction statistics ===")
            num_s = data['source'].num_nodes

            # 1. Compute the theoretical uniform-distribution weights
            s_indices = edge_index_sa[0]
            ones = torch.ones_like(s_indices, dtype=torch.float)
            out_degree_s = torch.zeros(num_s, device=self.device, dtype=torch.float).scatter_add_(0, s_indices, ones)
            uniform_weight_values = 1.0 / out_degree_s[s_indices].clamp(min=1)

            # 2. Compute the MAE (Mean Absolute Error)
            mae = torch.mean(torch.abs(edge_weights - uniform_weight_values))
            print(f"Overall MAE (vs. uniform distribution): {mae.item():.6f}")

            # 3. Verify that each source node's weight sum is 1
            weight_sums_per_s = torch.zeros(num_s, device=self.device).scatter_add_(0, s_indices, edge_weights)
            avg_weight_sum_error = torch.mean(torch.abs(weight_sums_per_s[out_degree_s > 0] - 1.0))
            print(f"Average weight sum error (should be close to 0): {avg_weight_sum_error.item():.6f}")
            print(f"Weight range: [{edge_weights.min().item():.4f}, {edge_weights.max().item():.4f}]")
            print("=" * 35 + "\n")

            # --- Build the result DataFrame ---
            agent_local_indices = edge_index_sa[1].cpu().numpy()
            agent_index_map = data.agent_index_map
            original_gdf_indices = pd.Series(agent_index_map).iloc[agent_local_indices].values
            source_indices = edge_index_sa[0].cpu().numpy()

            result_df = pd.DataFrame({
                'source_node_idx': source_indices,
                'agent_node_idx': agent_local_indices,
                'agent_original_idx': original_gdf_indices,
                'predicted_weight': edge_weights.cpu().numpy()
            })

            # Phase A: add the spectral cluster label
            if T_hat_pred is not None:
                spectral_cluster_all = torch.argmax(T_hat_pred, dim=1).cpu().numpy()
                result_df['spectral_cluster'] = spectral_cluster_all[agent_local_indices]

            print("Generated a DataFrame containing the original indices and weights.")
            return result_df

    def _load_checkpoint(self):
        """Load the state_dict of every available module from save_path."""
        try:
            checkpoint = torch.load(self.config.save_path, map_location=self.device)
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            if self.edge_weighting_layer is not None and 'edge_weighting_layer_state_dict' in checkpoint:
                self.edge_weighting_layer.load_state_dict(checkpoint['edge_weighting_layer_state_dict'])
            if self.projection_head is not None and 'projection_head_state_dict' in checkpoint:
                self.projection_head.load_state_dict(checkpoint['projection_head_state_dict'])
            if self.recon_head is not None and 'recon_head_state_dict' in checkpoint:
                self.recon_head.load_state_dict(checkpoint['recon_head_state_dict'])
            if self.agent_gating is not None and 'agent_gating_state_dict' in checkpoint:
                self.agent_gating.load_state_dict(checkpoint['agent_gating_state_dict'])
            print(f"Successfully loaded the model from '{self.config.save_path}'.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file '{self.config.save_path}' not found. Make sure the model has been trained and saved.")
        except Exception as e:
            raise RuntimeError(f"Error loading the model: {e}")
