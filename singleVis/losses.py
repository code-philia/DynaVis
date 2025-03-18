import torch
from torch import nn
from .backend import convert_distance_to_probability, compute_cross_entropy
import numpy as np
from tqdm import tqdm

"""
class UmapLoss(torch.nn.Module):
    def __init__(self, negative_sample_rate, device, a=1.0, b=1.0, repulsion_strength=1.0):
        super(UmapLoss, self).__init__()
        self._negative_sample_rate = negative_sample_rate
        self._a = a
        self._b = b
        self._repulsion_strength = repulsion_strength
        self.DEVICE = device

    def forward(self, embedding_to, embedding_from):
        batch_size = embedding_to.shape[0]
        embedding_neg_to = embedding_to.repeat_interleave(self._negative_sample_rate, dim=0)
        embedding_neg_from = embedding_from.repeat_interleave(self._negative_sample_rate, dim=0)
        randperm = torch.randperm(embedding_neg_from.shape[0])
        embedding_neg_from = embedding_neg_from[randperm]

        distance_embedding = torch.cat(
            (
                torch.norm(embedding_to - embedding_from, dim=1),
                torch.norm(embedding_neg_to - embedding_neg_from, dim=1)
            ),
            dim=0
        )

        probabilities_distance = 1.0 / (1.0 + self._a * distance_embedding ** (2 * self._b))
        probabilities_graph = torch.cat(
            (torch.ones(batch_size), torch.zeros(batch_size * self._negative_sample_rate)), dim=0
        ).to(self.DEVICE)

        attraction_loss = -torch.mean(probabilities_graph * torch.log(torch.clamp(probabilities_distance, 1e-12, 1.0)))
        repulsion_loss = -torch.mean((1.0 - probabilities_graph) * torch.log(torch.clamp(1.0 - probabilities_distance, 1e-12, 1.0)))
        loss = attraction_loss + self._repulsion_strength * repulsion_loss
        return loss
"""

class UmapLoss(nn.Module):
    def __init__(self, negative_sample_rate, device, a=1.0, b=1.0, repulsion_strength=1.0):
        super(UmapLoss, self).__init__()

        self._negative_sample_rate = negative_sample_rate
        self._a = a,
        self._b = b,
        self._repulsion_strength = repulsion_strength
        self.DEVICE = torch.device(device)

    @property
    def a(self):
        return self._a[0]

    @property
    def b(self):
        return self._b[0]

    def forward(self, embedding_to, embedding_from):
        batch_size = embedding_to.shape[0]
        # get negative samples
        embedding_neg_to = torch.repeat_interleave(embedding_to, self._negative_sample_rate, dim=0)
        repeat_neg = torch.repeat_interleave(embedding_from, self._negative_sample_rate, dim=0)
        randperm = torch.randperm(repeat_neg.shape[0])
        embedding_neg_from = repeat_neg[randperm]

        #  distances between samples (and negative samples)
        distance_embedding = torch.cat(
            (
                torch.norm(embedding_to - embedding_from, dim=1),
                torch.norm(embedding_neg_to - embedding_neg_from, dim=1),
            ),
            dim=0,
        )
        probabilities_distance = convert_distance_to_probability(
            distance_embedding, self.a, self.b
        )
        probabilities_distance = probabilities_distance.to(self.DEVICE)

        # set true probabilities based on negative sampling
        probabilities_graph = torch.cat(
            (torch.ones(batch_size), torch.zeros(batch_size * self._negative_sample_rate)), dim=0,
        )
        probabilities_graph = probabilities_graph.to(device=self.DEVICE)

        # compute cross entropy
        (_, _, ce_loss) = compute_cross_entropy(
            probabilities_graph,
            probabilities_distance,
            repulsion_strength=self._repulsion_strength,
        )

        return torch.mean(ce_loss)

class ReconLoss(torch.nn.Module):
    def __init__(self, beta=1.0):
        super(ReconLoss, self).__init__()
        self._beta = beta

    def forward(self, edge_to, edge_from, recon_to, recon_from):
        loss1 = torch.mean(torch.mean(torch.pow(edge_to - recon_to, 2), 1))
        loss2 = torch.mean(torch.mean(torch.pow(edge_from - recon_from, 2), 1))
        return (loss1 + loss2) / 2

class TemporalRankingLoss(nn.Module):
    def __init__(self, data_provider, temporal_edges):
        """
        :param data_provider: 数据提供者
        :param temporal_edges: (t_edge_from, t_edge_to) 所有时序边
        """
        super(TemporalRankingLoss, self).__init__()
        self.t_edge_from, self.t_edge_to = temporal_edges
        
    def forward(self, edge_to, edge_from, embedding_to, embedding_from, is_temporal):
        """
        :param edge_to: 当前batch中的目标节点特征
        :param edge_from: 当前batch中的源节点特征
        :param embedding_to: 目标节点的低维嵌入
        :param embedding_from: 源节点的低维嵌入
        :param is_temporal: 是否为时序边的标记
        """
        if not torch.any(is_temporal):
            return torch.tensor(0.0, device=edge_from.device, requires_grad=True)
        
        # 只保留时序边
        temporal_mask = is_temporal.bool()
        edge_to = edge_to[temporal_mask]
        edge_from = edge_from[temporal_mask]
        embedding_to = embedding_to[temporal_mask]
        embedding_from = embedding_from[temporal_mask]
        
        loss = torch.tensor(0.0, device=edge_from.device, requires_grad=True)
        valid_pairs = 0
        
        # 对当前batch中的每个from节点
        unique_from = torch.unique(edge_from, dim=0)
        # print(len(unique_from))
        for from_feat in unique_from:
            # 找到当前batch中这个from对应的所有to
            mask = torch.all(edge_from == from_feat, dim=1)
            curr_to_feats = edge_to[mask]
            curr_to_embeds = embedding_to[mask]
            curr_from_embed = embedding_from[mask][0]
            # print(len(curr_to_feats))
            
            # if len(curr_to_feats) < 2:
            #     continue
                
            # 计算高维空间中的距离
            D_high = torch.norm(curr_to_feats - from_feat.unsqueeze(0), dim=1)
            
            # 计算低维空间中的距离
            D_low = torch.norm(curr_to_embeds - curr_from_embed, dim=1)
            
            # 对所有可能的邻居对计算ranking loss
            n = len(curr_to_feats)
            for i in range(n):
                for k in range(n):
                    if i != k:
                        # 如果在高维空间中xi到xj的距离小于xi到xk的距离
                        if D_high[i] < D_high[k]:
                            # 那么在低维空间中也应该保持这种关系
                            # 如果不满足，就产生loss
                            if D_low[i] >= D_low[k]:
                                loss = loss + (D_low[i] - D_low[k])
                                valid_pairs += 1
        
        if valid_pairs == 0:
            return torch.tensor(0.0, device=edge_from.device, requires_grad=True)
        
        return loss / valid_pairs


class SingleVisLoss(nn.Module):
    def __init__(self, umap_loss, recon_loss, temporal_loss=None, lambd=1.0, gamma=1.0):
        super(SingleVisLoss, self).__init__()
        self.umap_loss = umap_loss
        self.recon_loss = recon_loss
        self.temporal_loss = temporal_loss
        self.lambd = lambd
        self.gamma = gamma

    def forward(self, edge_to, edge_from, outputs, is_temporal):
        """
        :param edge_to: 高维特征
        :param edge_from: 高维特征
        :param outputs: 模型输出的字典，包含umap和重构结果
        :param is_temporal: 布尔张量，标识哪些边是时序边
        """
        embedding_to, embedding_from = outputs["umap"]
        recon_to, recon_from = outputs["recon"]
        
        # UMAP loss
        umap_loss = self.umap_loss(embedding_to, embedding_from)
        
        # Reconstruction loss
        recon_loss = self.recon_loss(edge_to, edge_from, recon_to, recon_from)
        
        # Temporal ranking loss
        temporal_loss = 0.0
        if self.temporal_loss is not None:
            temporal_loss = self.temporal_loss(
                edge_to, 
                edge_from, 
                embedding_to, 
                embedding_from,
                is_temporal
            )
        
        total_loss = umap_loss + self.lambd * recon_loss + self.gamma * temporal_loss
        
        return umap_loss, recon_loss, temporal_loss, total_loss

import torch.nn.functional as F

def batch_ranking_loss(edge_from, edge_to, embedding_from, embedding_to):
    """
    计算 batch 内所有重复出现的 edge_from，其对应 edge_to 之间的 ranking loss
    """
    batch_size = edge_from.shape[0]

    # 统计 batch 内 `edge_from` 出现的次数
    unique_from, counts = torch.unique(edge_from, return_counts=True, dim=0)

    # 选择那些在 batch 里 **重复出现的 edge_from**
    repeated_from_mask = counts > 1
    repeated_from = unique_from[repeated_from_mask]  # 只保留重复出现的 edge_from
    # print(len(unique_from), len(repeated_from))

    loss = 0.0
    valid_count = 0  # 统计有多少个有效的 ranking loss 计算

    for i in range(repeated_from.shape[0]):  # 遍历所有重复的 edge_from
        current_from = repeated_from[i]  # 当前 `edge_from`

        # 找到所有 `edge_to`，也就是 batch 里 `edge_from == current_from` 的索引
        indices = torch.all(edge_from == current_from, dim=1).nonzero(as_tuple=True)[0]

        if len(indices) < 2:  # 只有一个 `edge_to`，无法计算 ranking loss，跳过
            continue

        selected_to = edge_to[indices]  # 这些 `edge_to` 是 high-dim 里当前 `edge_from` 连接的点
        selected_embedding_to = embedding_to[indices]  # 这些 `embedding_to` 是 low-dim 里对应的点

        # 计算 pairwise 距离
        D_high = torch.norm(selected_to[:, None, :] - selected_to[None, :, :], dim=-1)  # 高维 pairwise 距离
        D_low = torch.norm(selected_embedding_to[:, None, :] - selected_embedding_to[None, :, :], dim=-1)  # 低维 pairwise 距离

        # 计算高维邻居排序
        high_indices = torch.argsort(D_high, dim=-1)

        # 按照高维的排序索引，对低维距离进行排序
        sorted_low_distances, _ = torch.sort(D_low.gather(1, high_indices), dim=-1)

        # 计算 pairwise ranking loss，确保高维顺序与低维顺序一致
        loss += torch.mean(F.relu(sorted_low_distances[:, 1:] - sorted_low_distances[:, :-1]))
        valid_count += 1

    if valid_count == 0:
        return torch.tensor(0.0, device=edge_from.device)  # 避免除零错误

    return loss / valid_count  # 归一化

# 重新定义 listwise ranking loss
def listwise_ranking_loss(D_high, D_low, k=3):
    """
    计算 Listwise 排序损失，使用 KL 散度匹配邻居概率分布。
    """
    P_high = F.softmax(-D_high[:, :k], dim=1)  # 高维最近邻的 softmax 距离分布
    P_low = F.softmax(-D_low[:, :k], dim=1)    # 低维最近邻的 softmax 距离分布

    loss = F.kl_div(P_low.log(), P_high, reduction='batchmean')  # KL 散度损失
    return loss

