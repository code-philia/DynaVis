from torch.utils.data import WeightedRandomSampler
import torch
import numpy as np
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
    r"""采样器对空间边进行加权随机采样，对时序边按照edge_from分组采样。

    Args:
        weights (sequence): 所有边的权重序列
        num_spatial_samples (int): 要采样的空间边数量
        is_temporal (sequence): 布尔序列，标识每条边是否为时序边
        edge_from (sequence): 所有边的源节点索引
        generator (Generator): 用于采样的随机数生成器
    """

    def __init__(
        self,
        weights: Sequence[float],
        num_spatial_samples: int,
        is_temporal: Sequence[bool],
        edge_from: Sequence[int],
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
        
        # 确保所有输入长度相同
        if not (len(self.weights) == len(self.is_temporal) == len(self.edge_from)):
            raise ValueError("weights, is_temporal, and edge_from should have the same length")

        self.num_spatial_samples = num_spatial_samples
        self.generator = generator

        # 获取空间边和时序边的索引
        self.spatial_indices = torch.where(~self.is_temporal)[0]
        self.temporal_indices = torch.where(self.is_temporal)[0]

        if len(self.spatial_indices) == 0:
            raise ValueError("No spatial edges found in the dataset.")

        # 只保留空间边的权重
        self.spatial_weights = self.weights[self.spatial_indices]

        # 对时序边按照edge_from分组
        self.temporal_groups = self._group_temporal_edges()

    def _group_temporal_edges(self):
        """将时序边按照edge_from分组"""
        temporal_edge_from = self.edge_from[self.temporal_indices]
        unique_from = torch.unique(temporal_edge_from)
        
        # 创建分组字典
        groups = {}
        for from_idx in unique_from:
            # 找到所有从from_idx出发的时序边
            mask = temporal_edge_from == from_idx
            group_indices = self.temporal_indices[mask]
            groups[from_idx.item()] = group_indices
            
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
        
        # 随机打乱源节点的顺序
        from_nodes = list(self.temporal_groups.keys())
        np.random.shuffle(from_nodes)
        
        # 按照分组顺序添加时序边
        for from_idx in from_nodes:
            group_indices = self.temporal_groups[from_idx]
            temporal_indices.extend(group_indices.tolist())
        
        # 将采样的空间边索引和分组后的时序边索引连接
        all_indices = torch.cat([
            sampled_spatial_indices,
            torch.tensor(temporal_indices, dtype=torch.long)
        ])
        
        yield from iter(all_indices.tolist())

    def __len__(self) -> int:
        return self.num_spatial_samples + len(self.temporal_indices)