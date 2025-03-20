import os
import numpy as np
import scipy

from umap.umap_ import compute_membership_strengths
from singleVis.backend import get_graph_elements

class TemporalEdgeConstructor:
    def __init__(self, X, time_step_nums, n_neighbors, n_epochs):
        self.features = X
        self.time_step_nums = time_step_nums
        self.time_steps = len(time_step_nums)
        self.n_neighbors = n_neighbors
        self.n_epochs = n_epochs
    
    def temporal_simplicial_set(
        self,
        rows,
        cols,
        vals,
        n_vertice,
        set_op_mix_ratio=1.0,
        apply_set_operations=True):
        result = scipy.sparse.coo_matrix(
            (vals, (rows, cols)), shape=(n_vertice, n_vertice)
        )
        result.eliminate_zeros()

        if apply_set_operations:
            transpose = result.transpose()
            prod_matrix = result.multiply(transpose)
            result = (
                    set_op_mix_ratio * (result + transpose - prod_matrix)
                    + (1.0 - set_op_mix_ratio) * prod_matrix
            )
        result.eliminate_zeros()
        return result
    
    def construct(self):
        rows = np.zeros(1, dtype=np.int32)
        cols = np.zeros(1, dtype=np.int32)
        vals = np.zeros(1, dtype=np.float32)

        base_idx = 0
        base_idx_list = []
        for i in self.time_step_nums:
            base_idx_list.append(base_idx)
            base_idx = base_idx + i[0]
        base_idx_list = np.array(base_idx_list, dtype=int)

        num = len(self.features)

        # placeholder for knn_indices and knn_dists
        indices = - np.ones((num, self.n_neighbors), dtype=int)
        dists = np.zeros((num, self.n_neighbors), dtype=np.float32)

        for time_step in range(self.time_steps):
            start_idx = base_idx_list[time_step]
            end_idx = start_idx + self.time_step_nums[time_step][0] - 1

            for i in range(start_idx, end_idx + 1):
                if time_step < self.time_steps - 1:
                    next_start = base_idx_list[time_step + 1]
                    next_end = next_start + self.time_step_nums[time_step + 1][0] - 1
                    candidate_idxs = np.arange(start_idx, end_idx + 1)
                    candidate_idxs = np.concatenate((candidate_idxs, np.arange(next_start, next_end + 1)))
                else:
                    candidate_idxs = np.arange(start_idx, end_idx + 1)
                
                candidate_idxs = candidate_idxs[candidate_idxs != i]
                nn_dist = np.linalg.norm(self.features[i] - self.features[candidate_idxs], axis=1)
                if len(candidate_idxs) > self.n_neighbors:
                    sorted_indices = np.argsort(nn_dist)[:self.n_neighbors]
                    candidate_idxs = candidate_idxs[sorted_indices]
                    nn_dist = nn_dist[sorted_indices]
                else:
                    sorted_indices = np.argsort(nn_dist)
                    candidate_idxs = candidate_idxs[sorted_indices]
                    nn_dist = nn_dist[sorted_indices]
                
                indices[i] = candidate_idxs
                dists[i] = nn_dist

        # Ensure sigmas and rhos are float32
        sigmas = np.ones(num, dtype=np.float32)
        rhos = np.zeros(num, dtype=np.float32)

        rows, cols, vals, _ = compute_membership_strengths(indices, dists, sigmas, rhos, return_dists=False)
        time_complex = self.temporal_simplicial_set(rows=rows, cols=cols, vals=vals, n_vertice=num)
        _, heads, tails, weights, _ = get_graph_elements(time_complex, n_epochs=self.n_epochs)

        return heads, tails, weights

class BaselineTemporalEdgeConstructor:
    def __init__(self, X, time_step_nums, n_neighbors, n_epochs):
        self.features = X
        self.time_step_nums = time_step_nums
        self.time_steps = len(time_step_nums)
        self.n_neighbors = n_neighbors
        self.n_epochs = n_epochs

    def temporal_simplicial_set(
        self,
        rows,
        cols,
        vals,
        n_vertice,
        set_op_mix_ratio=1.0,
        apply_set_operations=True):
        result = scipy.sparse.coo_matrix(
            (vals, (rows, cols)), shape=(n_vertice, n_vertice)
        )
        result.eliminate_zeros()

        if apply_set_operations:
            transpose = result.transpose()
            prod_matrix = result.multiply(transpose)
            result = (
                    set_op_mix_ratio * (result + transpose - prod_matrix)
                    + (1.0 - set_op_mix_ratio) * prod_matrix
            )
        result.eliminate_zeros()
        return result
    
    def construct(self):
        rows = np.zeros(1, dtype=np.int32)
        cols = np.zeros(1, dtype=np.int32)
        vals = np.zeros(1, dtype=np.float32)

        base_idx = 0
        base_idx_list = []
        for i in self.time_step_nums:
            base_idx_list.append(base_idx)
            base_idx = base_idx + i[0]
        base_idx_list = np.array(base_idx_list, dtype=int)
        
        valid_idx_list = []
        for i in range(len(self.time_step_nums)):
            valid_idx_list.append(base_idx_list[i] + self.time_step_nums[i][0])
        valid_idx_list = np.array(valid_idx_list, dtype=int)
        
        num = len(self.features)
        
        indices = -np.ones((num, self.n_neighbors), dtype=int)
        dists = np.zeros((num, self.n_neighbors), dtype=np.float32)

        for time_step in range(self.time_steps):
            start_idx = base_idx_list[time_step]
            end_idx = start_idx + self.time_step_nums[time_step][0] - 1
            
            move_positions = base_idx_list - start_idx
            for sample_idx in range(start_idx, end_idx + 1):
                candidate_idxs = sample_idx + move_positions
                
                candidate_idxs = candidate_idxs[np.logical_and(
                    candidate_idxs >= base_idx_list, 
                    candidate_idxs < valid_idx_list
                )]
                
                candidate_idxs = candidate_idxs[candidate_idxs != sample_idx]
                
                nn_dist = np.linalg.norm(self.features[sample_idx] - self.features[candidate_idxs], axis=1)
                
                if len(candidate_idxs) > self.n_neighbors:
                    sorted_indices = np.argsort(nn_dist)[:self.n_neighbors]
                    top_k_idxs = candidate_idxs[sorted_indices]
                    top_k_dists = nn_dist[sorted_indices]
                else:
                    print("finding no enough neighbors")
                    assert 1==2
                    """
                    sorted_indices = np.argsort(nn_dist)
                    top_k_idxs = candidate_idxs[sorted_indices]
                    top_k_dists = nn_dist[sorted_indices]
                    top_k_idxs = np.pad(top_k_idxs, (0, self.n_neighbors - len(top_k_idxs)), 
                                         'constant', constant_values=-1)
                    top_k_dists = np.pad(top_k_dists, (0, self.n_neighbors - len(top_k_dists)), 
                                          'constant', constant_values=0.0)"
                    """
                
                indices[sample_idx] = top_k_idxs
                dists[sample_idx] = top_k_dists

        sigmas = np.ones(num, dtype=np.float32)
        rhos = np.zeros(num, dtype=np.float32)
        
        rows, cols, vals, _ = compute_membership_strengths(indices, dists, sigmas, rhos, return_dists=False)
        time_complex = self.temporal_simplicial_set(rows=rows, cols=cols, vals=vals, n_vertice=num)
        _, heads, tails, weights, _ = get_graph_elements(time_complex, n_epochs=self.n_epochs)
        
        return heads, tails, weights
    
    