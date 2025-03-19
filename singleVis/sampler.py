from torch.utils.data import WeightedRandomSampler
import torch
import numpy as np
from typing import List, Dict

class CustomWeightedRandomSampler(WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())

from torch.utils.data import Sampler
from typing import Sequence, Iterator

class TemporalPreservingSampler(Sampler[int]):
    r"""采样器对空间边进行加权随机采样，对时序边按照edge_to分组采样。

    Args:
        weights (sequence): 所有边的权重序列
        num_spatial_samples (int): 要采样的空间边数量
        is_temporal (sequence): 布尔序列，标识每条边是否为时序边
        edge_from (sequence): 所有边的源节点索引
        edge_to (sequence): 所有边的目标节点索引
        generator (Generator): 用于采样的随机数生成器
    """

    def __init__(
        self,
        weights: Sequence[float],
        num_spatial_samples: int,
        is_temporal: Sequence[bool],
        edge_from: Sequence[int],
        edge_to: Sequence[int],
        generator=None,
    ) -> None:
        if not isinstance(num_spatial_samples, int) or num_spatial_samples <= 0:
            raise ValueError(
                f"num_spatial_samples should be a positive integer value, but got num_spatial_samples={num_spatial_samples}"
            )

        # 转换输入为tensor
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.is_temporal = torch.as_tensor(is_temporal, dtype=torch.bool)
        self.edge_from = torch.as_tensor(edge_from)
        self.edge_to = torch.as_tensor(edge_to)
        
        # 确保所有输入长度相同
        if not (len(self.weights) == len(self.is_temporal) == len(self.edge_from) == len(self.edge_to)):
            raise ValueError("weights, is_temporal, edge_from, and edge_to should have the same length")

        self.num_spatial_samples = num_spatial_samples
        self.generator = generator

        # 获取空间边和时序边的索引
        self.spatial_indices = torch.where(~self.is_temporal)[0]
        self.temporal_indices = torch.where(self.is_temporal)[0]

        if len(self.spatial_indices) == 0:
            raise ValueError("No spatial edges found in the dataset.")

        # 只保留空间边的权重
        self.spatial_weights = self.weights[self.spatial_indices]

        # 对时序边按照edge_to分组
        self.temporal_groups = self._group_temporal_edges()

    def _group_temporal_edges(self) -> Dict[int, List[int]]:
        """将时序边按照edge_to分组"""
        temporal_edge_to = self.edge_to[self.temporal_indices]
        unique_to = torch.unique(temporal_edge_to)
        
        # 创建分组字典
        groups = {}
        for to_idx in unique_to:
            # 找到所有到to_idx的时序边
            mask = temporal_edge_to == to_idx
            group_indices = self.temporal_indices[mask]
            groups[to_idx.item()] = group_indices.tolist()
            
        return groups

    def __iter__(self) -> Iterator[int]:
        # 对空间边进行加权随机采样
        sampled_spatial_indices = torch.multinomial(
            self.spatial_weights,
            self.num_spatial_samples,
            replacement=True,
            generator=self.generator
        )
        sampled_spatial_indices = self.spatial_indices[sampled_spatial_indices]
        
        # 对时序边分组采样
        temporal_indices = []
        
        # 确保每个to节点的所有from节点都被采样
        for to_idx, group_indices in self.temporal_groups.items():
            temporal_indices.extend(group_indices)

        # 打印unique edge_from和对应的edge_to数量
        unique_from = torch.unique(self.edge_from[temporal_indices])
        print(f"Found {len(unique_from)} unique edge_from.")
        for from_idx in unique_from:
            count_to = (self.edge_from[temporal_indices] == from_idx).sum().item()
            print(f"Edge from {from_idx.item()} has {count_to} edge_to.")

        # 将采样的空间边索引和分组后的时序边索引连接
        all_indices = torch.cat([
            sampled_spatial_indices,
            torch.tensor(temporal_indices, dtype=torch.long)
        ])
        
        # 将所有索引按edge_from分组
        grouped_indices = self._group_indices_by_edge_from(all_indices.tolist())
        
        for group in grouped_indices:
            yield from group

    def _group_indices_by_edge_from(self, indices: List[int]) -> List[List[int]]:
        """将索引按edge_from分组"""
        grouped = {}
        for idx in indices:
            from_node = self.edge_from[idx].item()
            if from_node not in grouped:
                grouped[from_node] = []
            grouped[from_node].append(idx)
        return list(grouped.values())

    def __len__(self) -> int:
        return self.num_spatial_samples + len(self.temporal_indices)