import time
import json
import dataclasses
import datetime
from typing import Optional, Dict
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from SpatialAllocation.GNN.Allocation.AllocationConfig import AllocationConfig
from SpatialAllocation.GNN.Layer.GraphEncoder import GraphEncoder
from SpatialAllocation.GNN.Allocation.AllocationEdgeWeightLayer import AllocationEdgeWeighting
from SpatialAllocation.GNN.Allocation.LossFunction.AllocationCombinedLoss import AllocationCombinedLoss


class AllocationSolver:
    """
    Stage 2 allocation solver: trains the agent->target edge weight prediction model.

    Uses an independent GraphEncoder + AllocationEdgeWeighting,
    trained on the agent-target bipartite graph, with the loss combined via AllocationCombinedLoss.
    """

    def __init__(self, config: AllocationConfig):
        self.config = config

        # Set device
        if config.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config.device)

        self.encoder: Optional[GraphEncoder] = None
        self.edge_weight_layer: Optional[AllocationEdgeWeighting] = None

    def init_model(self, train_dataloader: DataLoader):
        """
        Infers the graph structure from the training data and initializes the encoder and edge weight layer.
        """
        first_batch = next(iter(train_dataloader))
        input_dims = {
            node_type: first_batch[node_type].x.shape[1]
            for node_type in first_batch.node_types
            if hasattr(first_batch[node_type], 'x')
        }
        metadata = first_batch.metadata()

        # Reuse GraphEncoder (supports arbitrary heterogeneous graphs)
        # Construct a temporary object compatible with ModelConfig to pass parameters
        encoder_config = _make_encoder_config(self.config)
        self.encoder = GraphEncoder(input_dims, encoder_config, metadata).to(self.device)
        self.edge_weight_layer = AllocationEdgeWeighting(self.config).to(self.device)

    def train(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader = None,
        objective_weights: Optional[Dict[str, float]] = None,
        eval_every: int = 1,
    ):
        """
        Trains the agent->target allocation model.

        Args:
            train_dataloader: training data loader
            test_dataloader: test data loader (optional)
            objective_weights: loss weight dictionary (keys are AllocationLossRegistry registered names)
            eval_every: number of epochs between test set evaluations
        """
        if objective_weights is None:
            objective_weights = {
                'allocation_distance': 1.0,
                'allocation_feature_homogeneity': 0.5,
            }

        criterion = AllocationCombinedLoss(
            objective_weights, learnable=self.config.learnable
        ).to(self.device)

        if self.config.debug:
            torch.autograd.set_detect_anomaly(True)

        if test_dataloader is None:
            print("No test data loader provided; training only.")

        self.init_model(train_dataloader)

        params = list(self.encoder.parameters()) + list(self.edge_weight_layer.parameters())
        optimizer = torch.optim.AdamW(
            params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )

        scheduler = None
        if self.config.use_scheduler:
            T_max = self.config.cosine_epochs if self.config.cosine_epochs is not None else self.config.epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=self.config.cosine_eta_min
            )

        # Initialize loss tracking
        train_losses = {key: [] for key in objective_weights.keys()}
        train_losses['total'] = []
        train_losses['learning_rate'] = []

        test_losses = {key: [] for key in objective_weights.keys()}
        test_losses['total'] = []

        best_test_loss = float('inf')
        best_epoch = -1

        train_start_time = datetime.datetime.now()
        print("Starting Stage 2 allocation model training...")
        for epoch in range(self.config.epochs):
            start_time = time.time()
            epoch_grad_norms = []

            # ================= TRAINING PHASE =================
            self.encoder.train()
            self.edge_weight_layer.train()

            epoch_train_losses = {key: 0.0 for key in train_losses.keys() if key != 'learning_rate'}
            num_train_batches = 0

            for batch_data in train_dataloader:
                optimizer.zero_grad()
                batch_data = batch_data.to(self.device)

                # Encoder performs message passing over agent<->target edges
                embeddings_dict = self.encoder(batch_data.x_dict, batch_data.edge_index_dict)
                embeddings_a = embeddings_dict['agent']
                embeddings_t = embeddings_dict['target']

                edge_index_at = batch_data['agent', 'connects_to', 'target'].edge_index
                w_at, edge_costs = self.edge_weight_layer(
                    embeddings_a, embeddings_t, edge_index_at
                )

                # Build the metadata required for the loss
                metadata_for_loss = {
                    'edge_index_at': edge_index_at,
                    'num_targets': batch_data['target'].num_nodes,
                    'num_a': batch_data['agent'].num_nodes,
                }
                if hasattr(batch_data['agent'], 'demand'):
                    metadata_for_loss['agent_demand'] = batch_data['agent'].demand
                if hasattr(batch_data['agent'], 'coords'):
                    metadata_for_loss['agent_coords'] = batch_data['agent'].coords
                if hasattr(batch_data['target'], 'coords'):
                    metadata_for_loss['target_coords'] = batch_data['target'].coords
                if hasattr(batch_data['agent'], 'x'):
                    metadata_for_loss['agent_features'] = batch_data['agent'].x

                total_loss, objectives = criterion(w_at, edge_index_at, metadata_for_loss)

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

            for key in epoch_train_losses:
                train_losses[key].append(epoch_train_losses[key] / max(num_train_batches, 1))
            train_losses['learning_rate'].append(optimizer.param_groups[0]['lr'])
            avg_epoch_train_loss = train_losses['total'][-1]
            avg_grad_norm = np.mean(epoch_grad_norms) if epoch_grad_norms else 0.0

            # ================= TESTING PHASE =================
            is_eval_epoch = (epoch % eval_every == 0) or (epoch == self.config.epochs - 1)

            if test_dataloader is not None and is_eval_epoch:
                self.encoder.eval()
                self.edge_weight_layer.eval()

                epoch_test_losses = {key: 0.0 for key in test_losses.keys()}
                num_test_batches = 0

                with torch.no_grad():
                    for batch_data in test_dataloader:
                        batch_data = batch_data.to(self.device)
                        embeddings_dict = self.encoder(batch_data.x_dict, batch_data.edge_index_dict)
                        embeddings_a = embeddings_dict['agent']
                        embeddings_t = embeddings_dict['target']

                        edge_index_at = batch_data['agent', 'connects_to', 'target'].edge_index
                        w_at_test, _ = self.edge_weight_layer(
                            embeddings_a, embeddings_t, edge_index_at
                        )

                        metadata_for_loss = {
                            'edge_index_at': edge_index_at,
                            'num_targets': batch_data['target'].num_nodes,
                            'num_a': batch_data['agent'].num_nodes,
                        }
                        if hasattr(batch_data['agent'], 'demand'):
                            metadata_for_loss['agent_demand'] = batch_data['agent'].demand
                        if hasattr(batch_data['agent'], 'coords'):
                            metadata_for_loss['agent_coords'] = batch_data['agent'].coords
                        if hasattr(batch_data['target'], 'coords'):
                            metadata_for_loss['target_coords'] = batch_data['target'].coords
                        if hasattr(batch_data['agent'], 'x'):
                            metadata_for_loss['agent_features'] = batch_data['agent'].x

                        total_loss, objectives = criterion(w_at_test, edge_index_at, metadata_for_loss)

                        epoch_test_losses['total'] += total_loss.item()
                        for key, value in objectives.items():
                            if key in epoch_test_losses:
                                epoch_test_losses[key] += value.item()
                        num_test_batches += 1

                for key in epoch_test_losses:
                    test_losses[key].append(epoch_test_losses[key] / max(num_test_batches, 1))
                avg_epoch_test_loss = test_losses['total'][-1]
            elif test_dataloader is not None:
                for key in test_losses:
                    if test_losses[key]:
                        test_losses[key].append(test_losses[key][-1])
                    else:
                        test_losses[key].append(float('inf'))
                avg_epoch_test_loss = test_losses['total'][-1]
            else:
                avg_epoch_test_loss = float('inf')

            if scheduler is not None:
                scheduler.step()

            end_time = time.time()
            log_parts = [
                f"Epoch {epoch + 1}/{self.config.epochs} in {end_time - start_time:.2f}s",
                f"Train: {avg_epoch_train_loss:.6f}",
            ]
            if test_dataloader is not None:
                log_parts.append(f"Test: {avg_epoch_test_loss:.6f}")
            log_parts.append(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            log_parts.append(f"Grad: {avg_grad_norm:.4f}")
            print(" | ".join(log_parts))

            # Save the best model
            ref_loss = avg_epoch_test_loss if test_dataloader is not None else avg_epoch_train_loss
            ref_label = "Test" if test_dataloader is not None else "Train"
            if ref_loss < best_test_loss:
                best_test_loss = ref_loss
                best_epoch = epoch + 1
                checkpoint = {
                    'epoch': epoch,
                    'encoder_state_dict': self.encoder.state_dict(),
                    'edge_weight_layer_state_dict': self.edge_weight_layer.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_test_loss,
                }
                if scheduler:
                    checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                torch.save(checkpoint, self.config.save_path)
                print(f'  -> Epoch {epoch + 1}, ** Best model saved ({ref_label} Loss: {best_test_loss:.6f}) **')

        # Save the training log
        self._save_training_log(
            train_start_time=train_start_time,
            objective_weights=objective_weights,
            train_losses=train_losses,
            test_losses=test_losses,
            best_epoch=best_epoch,
            best_loss=best_test_loss,
        )
        self._plot_training_curves({'train': train_losses, 'test': test_losses})
        print("Stage 2 allocation model training complete!")

    def predict(self, data: HeteroData) -> pd.DataFrame:
        """
        Inference: returns the agent->target edge weight DataFrame.
        """
        if self.encoder is None or self.edge_weight_layer is None:
            raise RuntimeError("Model has not been trained or loaded.")

        self._load_checkpoint()
        self.encoder.eval()
        self.edge_weight_layer.eval()

        with torch.no_grad():
            data = data.to(self.device)
            embeddings_dict = self.encoder(data.x_dict, data.edge_index_dict)
            embeddings_a = embeddings_dict['agent']
            embeddings_t = embeddings_dict['target']

            edge_index_at = data['agent', 'connects_to', 'target'].edge_index
            w_at, _ = self.edge_weight_layer(embeddings_a, embeddings_t, edge_index_at)

            # Build DataFrame
            at_agent_idx = edge_index_at[0].cpu().numpy()
            at_target_idx = edge_index_at[1].cpu().numpy()

            result_df = pd.DataFrame({
                'agent_node_idx': at_agent_idx,
                'target_node_idx': at_target_idx,
                'weight': w_at.cpu().numpy(),
            })

            print(f"\n=== Stage 2 Allocation Inference ===")
            print(f"W_at range: [{w_at.min().item():.6f}, {w_at.max().item():.6f}]")
            max_per_agent = result_df.groupby('agent_node_idx')['weight'].max()
            print(f"max(W_at)/agent: mean={max_per_agent.mean():.4f}, "
                  f">0.5: {(max_per_agent > 0.5).sum()}/{len(max_per_agent)}")

            # Compute predicted target demand (if agent demand is available)
            if hasattr(data['agent'], 'demand'):
                agent_demand = data['agent'].demand
                edge_demand = w_at * agent_demand[edge_index_at[0]]
                num_t = data['target'].num_nodes
                predicted_demand = torch.zeros(num_t, device=self.device)
                predicted_demand.scatter_add_(0, edge_index_at[1], edge_demand)

                result_df_demand = pd.DataFrame({
                    'target_node_idx': np.arange(num_t),
                    'predicted_demand': predicted_demand.cpu().numpy(),
                })

                if hasattr(data['target'], 'demand'):
                    actual_demand = data['target'].demand.cpu().numpy()
                    result_df_demand['actual_demand'] = actual_demand
                    rmse = np.sqrt(np.mean((predicted_demand.cpu().numpy() - actual_demand) ** 2))
                    print(f"Target demand RMSE: {rmse:.4f}")

                print("=" * 35 + "\n")
                return result_df, result_df_demand

            print("=" * 35 + "\n")
            return result_df

    def load_model(self, save_path: str = None):
        """External-facing model loading method (does not call init_model automatically; call it manually first)."""
        if save_path is not None:
            self.config.save_path = save_path
        self._load_checkpoint()

    def _load_checkpoint(self):
        """Loads module state_dicts from save_path."""
        try:
            checkpoint = torch.load(self.config.save_path, map_location=self.device)
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.edge_weight_layer.load_state_dict(checkpoint['edge_weight_layer_state_dict'])
            print(f"Successfully loaded Stage 2 model from '{self.config.save_path}'.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: '{self.config.save_path}'.")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

    def _save_training_log(
        self,
        train_start_time: datetime.datetime,
        objective_weights: Dict[str, float],
        train_losses: Dict[str, list],
        test_losses: Dict[str, list],
        best_epoch: int,
        best_loss: float,
    ):
        """Saves the training parameters and losses as a JSON log."""
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

        save_path = self.config.save_path
        if save_path.endswith('.pth'):
            log_path = save_path[:-4] + '_training_log.json'
        else:
            log_path = save_path + '_training_log.json'

        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
        print(f"Training log saved to: {log_path}")

    def _plot_training_curves(self, all_losses: Dict[str, Dict[str, list]]):
        """Plots the training/test loss curves."""
        train_losses = all_losses.get('train', {})
        test_losses = all_losses.get('test', {})
        loss_keys = [
            key for key in train_losses.keys()
            if key != 'learning_rate' and len(train_losses[key]) > 0
        ]

        num_plots = len(loss_keys)
        if num_plots == 0:
            print("No loss data available to plot.")
            return

        plt.figure(figsize=(6 * num_plots, 5))

        for i, key in enumerate(loss_keys):
            plt.subplot(1, num_plots, i + 1)
            plt.plot(train_losses[key], label=f'Train {key.replace("_", " ").title()}')
            if key in test_losses and test_losses[key]:
                plt.plot(test_losses[key], label=f'Test {key.replace("_", " ").title()}')
            plt.title(f'{key.replace("_", " ").title()} Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()

        plt.tight_layout()
        plt.show()


def _make_encoder_config(alloc_config: AllocationConfig):
    """
    Converts AllocationConfig into a ModelConfig-compatible object expected by GraphEncoder.
    GraphEncoder only uses hidden_dim, embedding_dim, num_layers, conv_type, gat_heads.
    """
    from SpatialAllocation.GNN.core.ModelConfig import ModelConfig
    return ModelConfig(
        hidden_dim=alloc_config.hidden_dim,
        embedding_dim=alloc_config.embedding_dim,
        num_layers=alloc_config.num_layers,
        conv_type=alloc_config.conv_type,
        gat_heads=alloc_config.gat_heads,
    )
