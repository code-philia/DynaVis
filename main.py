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
from singleVis.sampler import WeightedRandomSampler, TemporalPreservingSampler
from singleVis.losses import UmapLoss, ReconLoss, SingleVisLoss, TemporalRankingLoss

# Parameters
# content_path = "/home/zicong/data/Code_Retrieval_Samples/merged_train_data/"
content_path = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyiming-240108540153/training_dynamic/temporal_ranking/Model"
epoch_start = 1
epoch_end = 50
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
DEVICE = torch.device("cuda:0")
max_samples = 1000
VIS_MODEL_NAME = "tsvis"
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

# Create temporal edge identifier array
is_temporal = np.zeros(len(edge_to), dtype=bool)
is_temporal[len(s_edge_to):] = True  # Mark temporal edges

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
a, b = find_ab_params(1.0, 0.1)
umap_loss = UmapLoss(
    negative_sample_rate=5,
    device="cuda",
    a=a,
    b=b,
    repulsion_strength=1.0
)
recon_loss = ReconLoss(beta=1.0)
temporal_loss = TemporalRankingLoss(
    data_provider=data_provider,
    temporal_edges=(feature_vectors[t_edge_from], feature_vectors[t_edge_to])
)
criterion = SingleVisLoss(
    umap_loss=umap_loss,
    recon_loss=recon_loss,
    temporal_loss=temporal_loss,
    lambd=1.0,
    gamma=3.0  # Control temporal ranking loss weight
)

# Define optimizer and lr_scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=.1)

#print(edge_to.shape, edge_from.shape, feature_vectors.shape)
#assert 1==2

# Define DataLoader
from singleVis.data_handler import DataHandler
dataset = DataHandler(edge_to, edge_from, feature_vectors, is_temporal=is_temporal)
# n_samples = edge_to.shape[0]
n_samples = int(np.sum(S_N_EPOCHS * probs) // 1)

# 计算要采样的空间边数量
num_spatial_samples = int(np.sum(S_N_EPOCHS * probs[~is_temporal]) // 1)

# 创建采样器
sampler = TemporalPreservingSampler(
    weights=probs,
    num_spatial_samples=num_spatial_samples,
    is_temporal=is_temporal,
    edge_from=edge_from  # 添加edge_from参数
)

# 创建DataLoader
edge_loader = DataLoader(
    dataset, 
    batch_size=2000,
    sampler=sampler,
    drop_last=False
)

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

# save result
save_dir = content_path
 ##### save the visulization model
trainer.save(save_dir=save_dir, file_name="{}".format(VIS_MODEL_NAME))


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
from scipy.stats import spearmanr, kendalltau
def evaluate_proj_nn_perseverance_knn(data, embedding, n_neighbors, metric="euclidean"):
    """
    evaluate projection function, nn preserving property using knn algorithm
    :param data: ndarray, high dimensional representations
    :param embedding: ndarray, low dimensional representations
    :param n_neighbors: int, the number of neighbors
    :param metric: str, by default "euclidean"
    :return nn property: float, nn preserving property
    """
    n_trees = 5 + int(round((data.shape[0]) ** 0.5 / 20.0))
    n_iters = max(5, int(round(np.log2(data.shape[0]))))
    # get nearest neighbors
    nnd = NNDescent(
        data,
        n_neighbors=n_neighbors,
        metric=metric,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=False
    )
    nnd.prepare()
    high_ind, _ = nnd.neighbor_graph
    
    nnd = NNDescent(
        embedding,
        n_neighbors=n_neighbors,
        metric=metric,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=False
    )
    nnd.prepare()
    low_ind, _ = nnd.neighbor_graph

    border_pres = np.zeros(len(data))
    for i in range(len(data)):
        border_pres[i] = len(np.intersect1d(high_ind[i], low_ind[i]))
    
    return border_pres.mean()

def evaluate_proj_nn_ranking_preservation(data, embedding, n_neighbors, metric="euclidean"):
    """
    Evaluate projection function, nearest neighbor ranking preservation using spearman correlation
    and kendall tau correlation, and compute k-nearest neighbor preservation rate.
    
    Parameters:
    :param data: ndarray, high-dimensional representations
    :param embedding: ndarray, low-dimensional representations
    :param n_neighbors: int, number of neighbors
    :param metric: str, default "euclidean"
    
    Returns:
    :return ranking_score: float, nearest neighbor ranking preservation score (Spearman)
    :return kendall_score: float, nearest neighbor ranking preservation score (Kendall Tau)
    :return k_preservation_rate: float, k-nearest neighbor preservation rate
    """
    # Hyperparameters for NNDescent
    n_trees = 5 + int(round((data.shape[0]) ** 0.5 / 20.0))
    n_iters = max(5, int(round(np.log2(data.shape[0]))))
    
    # Compute high-dimensional nearest neighbors
    nnd_high = NNDescent(
        data,
        n_neighbors=n_neighbors,
        metric=metric,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=False
    )
    nnd_high.prepare()
    high_indices, _ = nnd_high.neighbor_graph
    
    # Compute low-dimensional nearest neighbors
    nnd_low = NNDescent(
        embedding,
        n_neighbors=n_neighbors,
        metric=metric,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=False
    )
    nnd_low.prepare()
    low_indices, _ = nnd_low.neighbor_graph
    
    # Compute k-nearest neighbor preservation rate
    k_preservation_rates = []
    for i in range(data.shape[0]):
        high_neighbor_set = set(high_indices[i])
        low_neighbor_set = set(low_indices[i])
        intersection = high_neighbor_set.intersection(low_neighbor_set)
        k_preservation_rates.append(len(intersection) / n_neighbors)
    k_preservation_rate = np.mean(k_preservation_rates)
    
    # Calculate rank preservation using Spearman's rank correlation
    ranking_scores = []
    kendall_scores = []
    
    for i in range(data.shape[0]):
        high_distances = high_indices[i]
        low_distances = low_indices[i]
        
        high_ranks = np.argsort(np.argsort(high_distances))
        low_ranks = np.argsort(np.argsort(low_distances))
        
        spearman_coeff, _ = spearmanr(high_ranks, low_ranks)
        ranking_scores.append(spearman_coeff)
        
        tau, _ = kendalltau(high_ranks, low_ranks)
        kendall_scores.append(tau)
    
    ranking_score = np.mean(ranking_scores)
    kendall_score = np.mean(kendall_scores)
    
    return ranking_score, kendall_score, k_preservation_rate

def show_sample_ranking_preservation(data, embedding, n_neighbors=15, n_samples=3, metric="euclidean"):
    """
    展示几个样本在降维前后的邻居排名保持情况
    
    Parameters:
    :param data: ndarray, 高维表示
    :param embedding: ndarray, 低维嵌入
    :param n_neighbors: int, 邻居数量
    :param n_samples: int, 要展示的样本数量
    :param metric: str, 距离度量
    """
    n_trees = 5 + int(round((data.shape[0]) ** 0.5 / 20.0))
    n_iters = max(5, int(round(np.log2(data.shape[0]))))
    
    nnd_high = NNDescent(
        data,
        n_neighbors=n_neighbors,
        metric=metric,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=False
    )
    nnd_high.prepare()
    high_indices, high_distances = nnd_high.neighbor_graph
    
    nnd_low = NNDescent(
        embedding,
        n_neighbors=n_neighbors,
        metric=metric,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=False
    )
    nnd_low.prepare()
    low_indices, low_distances = nnd_low.neighbor_graph
    
    ranking_scores = []
    kendall_scores = []
    for i in range(data.shape[0]):
        high_ranks = np.argsort(np.argsort(high_indices[i]))
        low_ranks = np.argsort(np.argsort(low_indices[i]))
        
        spearman_coeff, _ = spearmanr(high_ranks, low_ranks)
        tau, _ = kendalltau(high_ranks, low_ranks)
        
        ranking_scores.append(spearman_coeff)
        kendall_scores.append(tau)
    
    indices = np.argsort(ranking_scores)
    selected_indices = [
        indices[-1],  
        indices[len(indices)//2],  
        indices[0]  
    ]
    
    print("\n==== 样本邻居排名保持情况展示 ====")
    for idx in selected_indices:
        print(f"\n样本 #{idx}, Spearman系数: {ranking_scores[idx]:.4f}, Kendall Tau: {kendall_scores[idx]:.4f}")
        
        high_neighbors = high_indices[idx]
        low_neighbors = low_indices[idx]
        
        print("高维空间中的前5个邻居:")
        print(f"{'邻居ID':<10}{'高维排名':<10}{'低维排名':<10}{'排名变化':<10}")
        for j, neighbor_id in enumerate(high_neighbors[:15]):
            # 查找该邻居在低维空间中的排名
            low_rank = np.where(low_neighbors == neighbor_id)[0]
            if len(low_rank) > 0:
                low_rank = low_rank[0]
                print(f"{neighbor_id:<10}{j:<10}{low_rank:<10}{low_rank-j:<10}")
            else:
                print(f"{neighbor_id:<10}{j:<10}{'不在前k中':<10}{'':<10}")
        
        print("\n低维空间中的前5个邻居:")
        print(f"{'邻居ID':<10}{'低维排名':<10}{'高维排名':<10}{'排名变化':<10}")
        for j, neighbor_id in enumerate(low_neighbors[:15]):
            high_rank = np.where(high_neighbors == neighbor_id)[0]
            if len(high_rank) > 0:
                high_rank = high_rank[0]
                print(f"{neighbor_id:<10}{j:<10}{high_rank:<10}{j-high_rank:<10}")
            else:
                print(f"{neighbor_id:<10}{j:<10}{'不在前k中':<10}{'':<10}")

visualizer = DataVisualizer(
    data_provider=data_provider,
    model=model,
    resolution=500,
    save_path=os.path.join(content_path, "visualization_results", f"samples_{max_samples}")
)

for t in range(1,7,1):
    print(f"Processing epoch {t}")
    high_dim_data = data_provider.train_representation(epoch=t)
    if high_dim_data is None:
        print(f"No data found for epoch {t}, skipping...")
        continue
    
    model.eval()
    with torch.no_grad():
        embedding = model.encoder(
            torch.from_numpy(high_dim_data).to(dtype=torch.float32, device=DEVICE)
        ).cpu().numpy()
    
    # npr = evaluate_proj_nn_perseverance_knn(high_dim_data, embedding, n_neighbors=15, metric="euclidean")
    # print(f"Epoch {t}, Neighbour Preserving Rate: {npr:.4f}")
    ranking_score, kendall_score, k_preservation_rate = evaluate_proj_nn_ranking_preservation(
        high_dim_data, embedding, n_neighbors=15, metric="euclidean"
    )
    print(f"Epoch {t}, Nearest Neighbor Ranking Preservation (Spearman): {ranking_score:.4f}")
    print(f"Epoch {t}, Nearest Neighbor Ranking Preservation (Kendall): {kendall_score:.4f}")
    print(f"Epoch {t}, K-Nearest Neighbor Preservation Rate: {k_preservation_rate:.4f}")

    show_sample_ranking_preservation(high_dim_data, embedding, n_neighbors=15, n_samples=3)

    visualizer.plot(epoch=t)
    print(f"Epoch {t} visualization saved.")