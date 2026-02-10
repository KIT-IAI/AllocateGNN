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
    自监督学习的点分配求解器
    """

    def __init__(self, config: ModelConfig):
        self.config = config

        # 设置设备
        if config.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(config.device)

        self.encoder: Optional[GraphEncoder] = None
        self.edge_weighting_layer: Optional[DifferentiableEdgeWeighting] = None

    def init_model(self, train_dataloader: DataLoader):
        """
        初始化模型参数
        """
        # 从训练数据加载器中获取输入维度
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
        多图边权重预测的自监督训练 (已完全适配原生异构模型)，并包含测试/验证逻辑。
        """
        if objective_weights is None:
            objective_weights = {'entropy_regularization': 1.0}
        criterion = CombinedLoss(objective_weights, learnable=self.config.learnable).to(self.device)

        if self.config.debug:
            torch.autograd.set_detect_anomaly(True)

        if test_dataloader is None:
            print("没有提供测试数据加载器，将仅进行训练。")

        self.init_model(train_dataloader)

        params = list(self.encoder.parameters()) + list(self.edge_weighting_layer.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        scheduler = None
        if self.config.use_scheduler:
            T_max = self.config.cosine_epochs if self.config.cosine_epochs is not None else self.config.epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=self.config.cosine_eta_min
            )

        # --- 1. 为训练和测试分别初始化 losses 字典 ---
        train_losses = {key: [] for key in objective_weights.keys()}
        train_losses['total'] = []
        train_losses['learning_rate'] = []

        test_losses = {key: [] for key in objective_weights.keys()}
        test_losses['total'] = []

        best_test_loss = float('inf')  # 用于跟踪最佳测试损失

        print("开始原生异构图模型训练与测试...")
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
                # 只有当数据存在时才添加，以保持向后兼容性
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

                # --- 仅在调试时打印每个批次的损失 ---
                # print(f"  [Train] Batch {batch_idx + 1}/{len(train_dataloader)}, Total Loss: {total_loss.item():.6f}")

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

            # 计算并记录训练损失和学习率
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

                with torch.no_grad():  # 在测试阶段不计算梯度
                    for batch_data in test_dataloader:
                        batch_data = batch_data.to(self.device)
                        embeddings_dict = self.encoder(batch_data.x_dict, batch_data.edge_index_dict)
                        embeddings_s = embeddings_dict['source']
                        embeddings_a = embeddings_dict['agent']
                        edge_index_sa = batch_data['source', 'connects_to', 'agent'].edge_index
                        edge_weights, edge_costs = self.edge_weighting_layer(embeddings_s, embeddings_a, edge_index_sa)

                        # !! 修改点: 同样为测试阶段添加 landuse 矩阵 !!
                        metadata_for_loss = {
                            'num_s': batch_data['source'].num_nodes,
                            'num_a': batch_data['agent'].num_nodes,
                            'agent_features': batch_data['agent'].x,
                        }
                        # 只有当数据存在时才添加，以保持向后兼容性
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

                # 计算并记录测试损失
                for key in epoch_test_losses:
                    test_losses[key].append(epoch_test_losses[key] / num_test_batches)
                avg_epoch_test_loss = test_losses['total'][-1]
            else:
                avg_epoch_test_loss = float('inf')

            # --- Epoch 结束后的调度、保存和打印 ---
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

            # 根据测试损失保存最佳模型
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
                    print(f'  -> Epoch {epoch + 1}, ** 新的最佳模型已保存 (Test Loss: {best_test_loss:.6f}) **')
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
                    print(f'  -> Epoch {epoch + 1}, ** 新的最佳模型已保存 (Test Loss: {best_test_loss:.6f}) **')

        # --- 3. 调用修改后的绘图函数 ---
        # 假设 _plot_training_curves 函数可以接收一个包含 'train' 和 'test'键的字典
        self._plot_training_curves({'train': train_losses, 'test': test_losses})
        print("原生异构图训练和测试完成!")

    def _plot_training_curves(self, all_losses: Dict[str, Dict[str, list]]):
        """
        动态绘制所有记录的训练损失曲线。
        """
        # 获取所有损失的键名（除了学习率）
        train_losses = all_losses.get('train', {})
        test_losses = all_losses.get('test', {})
        loss_keys = [key for key in train_losses.keys() if key != 'learning_rate' and len(train_losses[key]) > 0]

        # 计算需要的子图数量
        num_plots = len(loss_keys)
        if num_plots == 0:
            print("没有可供绘制的损失数据。")
            return

        plt.figure(figsize=(6 * num_plots, 5))

        # 动态创建子图
        for i, key in enumerate(loss_keys):
            plt.subplot(1, num_plots, i + 1)
            plt.plot(train_losses[key], label=f'Train {key.replace("_", " ").title()} Loss')
            # 检查测试集中是否有该损失的记录
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
        使用训练好的模型，为给定的异构图数据预测边权重，并根据原始索引重新排序。

        Args:
            data (HeteroData): 一个包含'source'和'agent'节点及它们之间边的图数据对象。
                               这个对象应包含一个名为 'agent_index_map' 的属性，
                               它是一个能将图内部agent索引映射回原始gdf索引的Series或数组。

        Returns:
            np.ndarray: 一个包含预测边权重的NumPy数组，其顺序与原始agent的GeoDataFrame索引一致。
        """
        if self.encoder is None or self.edge_weighting_layer is None:
            raise RuntimeError("模型尚未训练或加载。请先调用 train_multi_graph 或加载一个模型。")

        # 检查必要的元数据是否存在
        if not hasattr(data, 'agent_index_map'):
            raise ValueError("输入的 HeteroData 对象必须包含 'agent_index_map' 属性用于权重重新排序。")

        # 加载最佳模型的状态
        try:
            checkpoint = torch.load(self.config.save_path, map_location=self.device)
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.edge_weighting_layer.load_state_dict(checkpoint['edge_weighting_layer_state_dict'])
            print(f"成功从 '{self.config.save_path}' 加载模型。")
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到模型文件 '{self.config.save_path}'。请确保模型已训练并保存。")
        except Exception as e:
            raise RuntimeError(f"加载模型时出错: {e}")

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
                raise ValueError("输入的 HeteroData 对象中必须包含 ('source', 'connects_to', 'agent') 类型的边。")

            edge_weights, edge_costs = self.edge_weighting_layer(
                embeddings_s, embeddings_a, edge_index_sa
            )

            # --- 重新引入验证逻辑 ---
            print("\n=== 边权重预测结果统计 ===")
            num_s = data['source'].num_nodes

            # 1. 计算理论上的均匀分布权重
            uniform_weights = torch.zeros_like(edge_weights)
            s_indices = edge_index_sa[0]
            # 使用 scatter_add 高效计算每个 source 的出边数量
            ones = torch.ones_like(s_indices, dtype=torch.float)
            out_degree_s = torch.zeros(num_s, device=self.device, dtype=torch.float).scatter_add_(0, s_indices, ones)
            # 避免除以零
            uniform_weight_values = 1.0 / out_degree_s[s_indices].clamp(min=1)
            uniform_weights = uniform_weight_values

            # 2. 计算 MAE (Mean Absolute Error)
            mae = torch.mean(torch.abs(edge_weights - uniform_weights))
            print(f"总体 MAE (对比均匀分布): {mae.item():.6f}")

            # 3. 验证每个 source 节点的权重和是否为1
            weight_sums_per_s = torch.zeros(num_s, device=self.device).scatter_add_(0, s_indices, edge_weights)
            avg_weight_sum_error = torch.mean(torch.abs(weight_sums_per_s[out_degree_s > 0] - 1.0))
            print(f"平均权重和误差 (应接近0): {avg_weight_sum_error.item():.6f}")
            print(f"权重范围: [{edge_weights.min().item():.4f}, {edge_weights.max().item():.4f}]")
            print("=" * 35 + "\n")
            # --- 验证逻辑结束 ---

            # --- 重新引入二次调整（重新排序）逻辑 ---
            # 获取图中边的 agent 局部索引
            agent_local_indices = edge_index_sa[1].cpu().numpy()

            # 使用 agent_index_map 将局部索引转换为原始的 GeoDataFrame 索引
            # agent_index_map 应该是一个 Series 或者 dict，键是局部索引，值是原始索引
            agent_index_map = data.agent_index_map
            original_gdf_indices = pd.Series(agent_index_map).iloc[agent_local_indices].values

            # 一个agent只属于一个source
            source_indices = edge_index_sa[0].cpu().numpy()

            # 返回一个包含 source_id, agent_id, weight 的 DataFrame，这是最清晰的
            result_df = pd.DataFrame({
                'source_node_idx': source_indices,
                'agent_node_idx': agent_local_indices,
                'agent_original_idx': original_gdf_indices,
                'predicted_weight': edge_weights.cpu().numpy()
            })

            print("已生成包含原始索引和权重的 DataFrame。")

            # 为了完全模拟原代码的返回类型，我们返回与边一一对应的权重值 numpy 数组
            # 调用者需要自己处理索引映射
            return result_df