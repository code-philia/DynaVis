import os
import time
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.neighbors import NearestNeighbors
from singleVis.data_provider import DataProvider, NewDataProvider
from singleVis.spatial_edge_constructor import SpatialEdgeConstructor
from singleVis.temporal_edge_constructor import TemporalEdgeConstructor
from singleVis.visualization_model import SingleVisualizationModel
from singleVis.trainer import SingleVisTrainer
from singleVis.backend import find_ab_params
from singleVis.visualizer import DataVisualizer
from singleVis.sampler import WeightedRandomSampler

# Parameters
# content_path = "/home/zicong/data/Code_Retrieval_Samples/merged_train_data/"
content_path = "/home/zicong/data/multi_epochs_cls/"
epoch_start = 1
epoch_end = 6
epoch_period = 1
split = 0
input_dims = 768  # Adjust according to your data
output_dims = 2
units = 256
hidden_layer = 3
n_neighbors = 15
s_n_epochs = 5
b_n_epochs = 5
t_n_epochs = 100
final_k = 0
persistence = 0
INIT_NUM = 100
ALPHA = 0.0
BETA = 0.1
MAX_EPOCH = 20
PATIENT = 5
S_N_EPOCHS = 5
B_N_EPOCHS = 5
T_N_EPOCHS = 100
VARIANTS = "SVis"
TEMP_TYPE = "local"
SCHEDULE = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_samples = 500
# selected_groups = [2]

# Data Provider
# data_provider = DataProvider(content_path, epoch_start, epoch_end, epoch_period, split, selected_groups=selected_groups)
data_provider = NewDataProvider(content_path, epoch_start, epoch_end, epoch_period, split, max_samples=max_samples)


# Construct Spatial-Temporal Complex
spatial_cons = SpatialEdgeConstructor(
    data_provider=data_provider,
    init_num=INIT_NUM,
    s_n_epochs=S_N_EPOCHS,
    b_n_epochs=B_N_EPOCHS,
    n_neighbors=n_neighbors,
)
s_edge_to, s_edge_from, s_probs, feature_vectors, time_step_nums, time_step_idxs_list = spatial_cons.construct()

# Construct Temporal Complex
temporal_cons = TemporalEdgeConstructor(
    X=feature_vectors,
    time_step_nums=time_step_nums,
    n_neighbors=n_neighbors,
    n_epochs=T_N_EPOCHS
)
t_edge_to, t_edge_from, t_probs = temporal_cons.construct()

# Merge edges
edge_to = np.concatenate((s_edge_to, t_edge_to), axis=0)
edge_from = np.concatenate((s_edge_from, t_edge_from), axis=0)
probs = np.concatenate((s_probs, t_probs), axis=0)
probs = probs / (probs.max() + 1e-3)
eliminate_zeros = probs > 1e-3
edge_to = edge_to[eliminate_zeros]
edge_from = edge_from[eliminate_zeros]
probs = probs[eliminate_zeros]

# Define the model
model = SingleVisualizationModel(
    input_dims=input_dims,
    output_dims=output_dims,
    units=units,
    hidden_layer=hidden_layer,
    device=DEVICE
)
model = model.to(DEVICE)

# Define loss
from singleVis.losses import UmapLoss, ReconLoss, SingleVisLoss
a, b = find_ab_params(1.0, 0.1)
umap_loss = UmapLoss(
    negative_sample_rate=5,
    device="cuda",
    a=a,
    b=b,
    repulsion_strength=1.0
)
recon_loss = ReconLoss(beta=1.0)
criterion = SingleVisLoss(umap_loss, recon_loss, lambd=1.0)

# Define optimizer and lr_scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)

#print(edge_to.shape, edge_from.shape, feature_vectors.shape)
#assert 1==2

# Define DataLoader
from singleVis.data_handler import DataHandler
dataset = DataHandler(edge_to, edge_from, feature_vectors)
# n_samples = edge_to.shape[0]
n_samples = int(np.sum(S_N_EPOCHS * probs) // 1)
sampler = WeightedRandomSampler(probs, n_samples, replacement=True)
edge_loader = DataLoader(dataset, batch_size=1000, sampler=sampler)

# Trainer
trainer = SingleVisTrainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    edge_loader=edge_loader
)

# Train the model
trainer.train(PATIENT=PATIENT, max_epochs=MAX_EPOCH)

"""
# Visualization
model.eval()
with torch.no_grad():
    embedding = model.encoder(torch.tensor(feature_vectors, dtype=torch.float32, device="cuda")).cpu().numpy()
vis_dir = os.path.join(content_path, "vis")
if not os.path.exists(vis_dir):
    os.mkdir(vis_dir)
np.save(os.path.join(vis_dir, "TimeVis_embedding.npy"), embedding)

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 10))
plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5)
plt.title("TimeVis Visualization")
plt.show()
"""

from pynndescent import NNDescent
from scipy.stats import spearmanr
from pynndescent import NNDescent
from scipy.stats import spearmanr

def evaluate_temporal_neighbor_preservation(data_provider, model, n_neighbors_list, device, epochs=None, metric="euclidean"):
    """
    评估不同时间步骤之间的邻居保留情况
    
    Parameters:
    :param data_provider: 数据提供者
    :param model: 可视化模型
    :param n_neighbors_list: 需要测试的k邻居数列表
    :param device: 运算设备
    :param epochs: 指定要评估的轮次列表，如果为None则使用所有可用轮次
    :param metric: 距离度量，默认为"euclidean"
    
    Returns:
    :return temporal_preservation_results: 字典，包含不同时间步骤和不同k值的邻居保留率
    """
    if epochs is None:
        epochs = list(range(data_provider.s_epoch, data_provider.e_epoch + 1, data_provider.p_epoch))
    
    results = {}
    
    epoch_data = {}
    epoch_embeddings = {}
    
    model.eval()
    for t in epochs:
        high_dim_data = data_provider.train_representation(epoch=t)
        if high_dim_data is None:
            print(f"No data found for epoch {t}, skipping...")
            continue
            
        with torch.no_grad():
            embedding = model.encoder(
                torch.from_numpy(high_dim_data).to(dtype=torch.float32, device=device)
            ).cpu().numpy()
        
        epoch_data[t] = high_dim_data
        epoch_embeddings[t] = embedding
    
    valid_epochs = sorted(list(epoch_data.keys()))
    
    for i in range(len(valid_epochs) - 1):
        curr_epoch = valid_epochs[i]
        next_epoch = valid_epochs[i+1]
        
        curr_data = epoch_data[curr_epoch]
        next_data = epoch_data[next_epoch]
        
        curr_embedding = epoch_embeddings[curr_epoch]
        next_embedding = epoch_embeddings[next_epoch]
        
        min_size = min(curr_data.shape[0], next_data.shape[0])
        curr_data = curr_data[:min_size]
        next_data = next_data[:min_size]
        curr_embedding = curr_embedding[:min_size]
        next_embedding = next_embedding[:min_size]
        
        epoch_pair_key = f"{curr_epoch}->{next_epoch}"
        results[epoch_pair_key] = {}
        
        for k in n_neighbors_list:
            n_trees = 5 + int(round((min_size) ** 0.5 / 20.0))
            n_iters = max(5, int(round(np.log2(min_size))))
            
            nnd_high_curr = NNDescent(
                curr_data, 
                n_neighbors=k, 
                metric=metric,
                n_trees=n_trees,
                n_iters=n_iters,
                max_candidates=60,
                verbose=False
            )
            nnd_high_curr.prepare()
            high_ind_curr, _ = nnd_high_curr.neighbor_graph
            
            nnd_high_next = NNDescent(
                next_data, 
                n_neighbors=k, 
                metric=metric,
                n_trees=n_trees,
                n_iters=n_iters,
                max_candidates=60,
                verbose=False
            )
            nnd_high_next.prepare()
            high_ind_next, _ = nnd_high_next.neighbor_graph
            
            nnd_low_curr = NNDescent(
                curr_embedding, 
                n_neighbors=k, 
                metric=metric,
                n_trees=n_trees,
                n_iters=n_iters,
                max_candidates=60,
                verbose=False
            )
            nnd_low_curr.prepare()
            low_ind_curr, _ = nnd_low_curr.neighbor_graph
            
            nnd_low_next = NNDescent(
                next_embedding, 
                n_neighbors=k, 
                metric=metric,
                n_trees=n_trees,
                n_iters=n_iters,
                max_candidates=60,
                verbose=False
            )
            nnd_low_next.prepare()
            low_ind_next, _ = nnd_low_next.neighbor_graph
            
            temporal_pres = np.zeros(min_size)
            for j in range(min_size):
                high_temporal_neighbors = np.intersect1d(high_ind_curr[j], high_ind_next[j])
                
                low_temporal_neighbors = np.intersect1d(low_ind_curr[j], low_ind_next[j])
                
                temporal_intersection = np.intersect1d(high_temporal_neighbors, low_temporal_neighbors)
                temporal_pres[j] = len(temporal_intersection) / k if len(high_temporal_neighbors) > 0 else 0
                
            avg_temporal_pres = temporal_pres.mean()
            results[epoch_pair_key][k] = avg_temporal_pres
            
    return results

print("\nEvaluating temporal neighbor preservation across epochs...")
n_neighbors_list = [1, 3, 5, 10, 15]
temporal_results = evaluate_temporal_neighbor_preservation(
    data_provider=data_provider,
    model=model,
    n_neighbors_list=n_neighbors_list,
    device=DEVICE,
    epochs=list(range(1, 7, 1))
)

print("\nTemporal Neighbor Preservation Results:")
for epoch_pair, k_results in temporal_results.items():
    print(f"\nEpoch transition {epoch_pair}:")
    for k, preservation_rate in k_results.items():
        print(f"  k={k}: {preservation_rate:.4f}")