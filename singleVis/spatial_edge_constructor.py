import os
import time
import math
import json
from umap.umap_ import fuzzy_simplicial_set
from pynndescent import NNDescent
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
import numpy as np
from .backend import get_graph_elements

class SpatialEdgeConstructor:
    def __init__(self, data_provider, init_num, s_n_epochs, b_n_epochs, n_neighbors):
        self.data_provider = data_provider
        self.init_num = init_num
        self.s_n_epochs = s_n_epochs
        self.b_n_epochs = b_n_epochs
        self.n_neighbors = n_neighbors

    def _construct_fuzzy_complex(self, train_data):
        n_trees = min(64, 5 + int(round(train_data.shape[0] ** 0.5 / 20.0)))
        n_iters = max(5, int(round(np.log2(train_data.shape[0]))))
        nnd = NNDescent(
            train_data,
            n_neighbors=self.n_neighbors,
            metric="euclidean",
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=60,
            verbose=False
        )
        knn_indices, knn_dists = nnd.neighbor_graph
        
        random_state = check_random_state(None)
        complex, sigmas, rhos = fuzzy_simplicial_set(
            X=train_data,
            n_neighbors=self.n_neighbors,
            metric="euclidean",
            random_state=random_state,
            knn_indices=knn_indices,
            knn_dists=knn_dists,
        )
        return complex, sigmas, rhos, knn_indices
    
    def _construct_step_edge_dataset(self, vr_complex):
        if vr_complex is None:
            return None, None, None
        
        _, vr_head, vr_tail, vr_weight, _ = get_graph_elements(vr_complex, self.s_n_epochs)
        return vr_head, vr_tail, vr_weight

    def construct(self):
        edge_to = []
        edge_from = []
        feature_vectors = []
        time_step_nums = []
        time_step_idxs_list = []
        all_probs = []
        
        for t in range(self.data_provider.s, self.data_provider.e + 1, self.data_provider.p):
            train_data = self.data_provider.train_representation(t)
            
            if train_data is None:
                continue
            
            complex, sigmas, rhos, knn_indices = self._construct_fuzzy_complex(train_data)
            head, tail, weight = self._construct_step_edge_dataset(complex)
            
            edge_to.append(head)
            edge_from.append(tail)
            feature_vectors.append(train_data)
            
            probs = np.ones_like(weight) / len(weight)  
            all_probs.append(probs)
            
            time_step_nums.append((train_data.shape[0], 0))
            time_step_idxs_list.append(np.arange(train_data.shape[0]).tolist())
        
        edge_to = np.concatenate(edge_to, axis=0)
        edge_from = np.concatenate(edge_from, axis=0)
        feature_vectors = np.vstack(feature_vectors)
        time_step_nums = np.array(time_step_nums)
        all_probs = np.concatenate(all_probs, axis=0)
        
        probs = all_probs / all_probs.max()
        
        return edge_to, edge_from, probs, feature_vectors, time_step_nums, time_step_idxs_list

class IncrSpatialEdgeConstructor:
    def __init__(self, data_provider, init_num, s_n_epochs, b_n_epochs, n_neighbors):
        self.data_provider = data_provider
        self.init_num = init_num
        self.s_n_epochs = s_n_epochs
        self.b_n_epochs = b_n_epochs
        self.n_neighbors = n_neighbors

    def _construct_fuzzy_complex(self, train_data):
        n_trees = min(64, 5 + int(round(train_data.shape[0] ** 0.5 / 20.0)))
        n_iters = max(5, int(round(np.log2(train_data.shape[0]))))
        nnd = NNDescent(
            train_data,
            n_neighbors=self.n_neighbors,
            metric="euclidean",
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=60,
            verbose=False
        )
        knn_indices, knn_dists = nnd.neighbor_graph
        
        random_state = check_random_state(None)
        complex, sigmas, rhos = fuzzy_simplicial_set(
            X=train_data,
            n_neighbors=self.n_neighbors,
            metric="euclidean",
            random_state=random_state,
            knn_indices=knn_indices,
            knn_dists=knn_dists,
        )
        return complex, sigmas, rhos, knn_indices
    
    def _construct_step_edge_dataset(self, vr_complex):
        if vr_complex is None:
            return None, None, None
        
        _, vr_head, vr_tail, vr_weight, _ = get_graph_elements(vr_complex, self.s_n_epochs)
        return vr_head, vr_tail, vr_weight

    def construct(self):
        edge_to = []
        edge_from = []
        feature_vectors = []
        groups_list = []
        time_step_nums = []
        time_step_idxs_list = []
        all_probs = []
        
        for t in range(self.data_provider.s, self.data_provider.e + 1, self.data_provider.p):
            train_data, groups = self.data_provider.train_representation(t)
            if train_data is None:
                continue
            
            complex, sigmas, rhos, knn_indices = self._construct_fuzzy_complex(train_data)
            head, tail, weight = self._construct_step_edge_dataset(complex)
            
            edge_to.append(head)
            edge_from.append(tail)
            feature_vectors.append(train_data)
            groups_list.append(groups)
            
            probs = np.ones_like(weight) / len(weight)  
            all_probs.append(probs)
            
            time_step_nums.append((train_data.shape[0], 0))
            time_step_idxs_list.append(np.arange(train_data.shape[0]).tolist())
        
        edge_to = np.concatenate(edge_to, axis=0)
        edge_from = np.concatenate(edge_from, axis=0)
        feature_vectors = np.vstack(feature_vectors)
        groups_list = np.concatenate(groups_list)
        time_step_nums = np.array(time_step_nums)
        all_probs = np.concatenate(all_probs, axis=0)
        
        probs = all_probs / all_probs.max()
        
        return edge_to, edge_from, probs, feature_vectors, time_step_nums, time_step_idxs_list, groups_list