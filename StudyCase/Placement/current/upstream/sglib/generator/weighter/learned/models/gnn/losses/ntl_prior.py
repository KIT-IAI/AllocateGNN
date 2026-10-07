import torch
from torch import nn

from .base import loss_registry, BaseLoss


@loss_registry.register(
    "ntl_prior",
    default_weight=0.1,
    description="NTL 夜间灯光先验: Forward KL(target || w)，软约束 w 覆盖 NTL 高值区域。"
)
class NTLPriorLoss(BaseLoss):
    """
    夜间灯光先验损失 L_ntl_prior。

    对每个 source 节点，将 agent 的 NTL 值 log 变换后归一化为 target 分布，
    然后用 Forward KL 散度约束 edge_weights 覆盖 NTL 高值区域。

    公式:
        target(s,a) = log(1 + ntl(a)) / Σ_{a'∈s} log(1 + ntl(a'))   (仅 RCI agent)
        L = mean_s [ Σ_a target(s,a) × log( target(s,a) / w(s,a) ) ]

    设计要点:
        - Forward KL(target || w): mode-covering，w 必须覆盖 target 高值区，
          但不强制 w 在 target 低值区趋零，避免与 temperature softmax 叠加导致权重过度集中
        - RCI mask: 非 RCI agent（lu_res+com+ind ≤ threshold）的 NTL 贡献归零，不参与 target
        - log(1+x): 压缩 NTL 原始值右偏（range [0.34, 267]），避免少数高 NTL agent 主导
        - Per-source 均值: 不偏向 agent 多的 source

    metadata 需要:
        - 'agent_ntl': (num_agents,) 每个 agent 的 NTL 值
        - 'agent_rci_mask': (num_agents,) bool, RCI agent 标识（可选）
        - 'num_s': int, source 节点数量
    """

    def forward(self, edge_weights, edge_index, metadata):
        device = edge_weights.device

        agent_ntl = metadata.get('agent_ntl')
        rci_mask = metadata.get('agent_rci_mask')
        num_s = metadata.get('num_s')

        if agent_ntl is None or num_s is None or num_s == 0:
            return torch.tensor(0.0, device=device)

        s_indices = edge_index[0]  # (num_edges,)
        a_indices = edge_index[1]  # (num_edges,)

        # 1. edge-level NTL, RCI 过滤 + log 变换
        edge_ntl = agent_ntl[a_indices]
        if rci_mask is not None:
            edge_ntl = edge_ntl * rci_mask[a_indices].float()
        edge_ntl = torch.log(1.0 + torch.clamp(edge_ntl, min=0.0))
        edge_ntl = torch.clamp(edge_ntl, min=1e-8)

        # 2. per-source 归一化: target = log(1+ntl(a)) / Σ_{a'∈s} log(1+ntl(a'))
        ntl_sum = torch.zeros(num_s, device=device)
        ntl_sum.scatter_add_(0, s_indices, edge_ntl)
        target = edge_ntl / (ntl_sum[s_indices] + 1e-8)

        # 3. Forward KL(target || w) = Σ target * log(target / w)
        t_safe = torch.clamp(target, min=1e-8)
        w_safe = torch.clamp(edge_weights, min=1e-8)
        kl_terms = t_safe * (torch.log(t_safe) - torch.log(w_safe))

        # 4. per-source 求和后取均值
        kl_per_s = torch.zeros(num_s, device=device)
        kl_per_s.scatter_add_(0, s_indices, kl_terms)
        valid = ntl_sum > 1e-6
        if valid.sum() > 0:
            return kl_per_s[valid].mean()
        else:
            return torch.tensor(0.0, device=device)


