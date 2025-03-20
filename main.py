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
from singleVis.losses import UmapLoss, ReconLoss, SingleVisLoss, TemporalRankingLoss, UnifiedRankingLoss
import matplotlib.pyplot as plt
import argparse

# def parse_arguments():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--select_idxs", nargs="+", type=int, default=1)

#     args = parser.parse_args()

#     return args

# args = parse_arguments()

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
MAX_EPOCH = 15
PATIENT = 3
S_N_EPOCHS = 5
B_N_EPOCHS = 5
T_N_EPOCHS = 100
VARIANTS = "SVis"
TEMP_TYPE = "local"
SCHEDULE = None
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
max_samples = 1000
selected_idxs = list(range(96))
observe_idxs=[4,6,48]

print("="*100)
print(f"selected_idxs: {selected_idxs}")
print("="*100)

# Data Provider
# data_provider = DataProvider(content_path, epoch_start, epoch_end, epoch_period, split, selected_groups=selected_groups)
data_provider = DataProvider(content_path, epoch_start, epoch_end, epoch_period, selected_idxs)


if len(selected_idxs) <= 2:
    edge_constructor = SimplifiedEdgeConstructor(
        data_provider=data_provider,
        init_num=INIT_NUM,
        s_n_epochs=S_N_EPOCHS,
        b_n_epochs=B_N_EPOCHS,
        n_neighbors=n_neighbors,
    )
    edge_to, edge_from, probs, feature_vectors, time_step_nums, time_step_idxs_list = edge_constructor.construct()
    is_temporal = np.zeros(len(edge_to), dtype=bool)
    is_temporal[len(edge_to):] = True  # Mark temporal edges
else :
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
    device=DEVICE,
    a=a,
    b=b,
    repulsion_strength=1.0
)
recon_loss = ReconLoss(beta=1.0)
if len(selected_idxs) <= 2:
    temporal_loss = TemporalRankingLoss(
        data_provider=data_provider,
        temporal_edges=(edge_to, edge_from)
    )
else:
    temporal_loss = TemporalRankingLoss(
        data_provider=data_provider,
        temporal_edges=(feature_vectors[t_edge_from], feature_vectors[t_edge_to])
    )
    
# temporal_loss = UnifiedRankingLoss(
#     edge_from=feature_vectors[edge_from],
#     edge_to=feature_vectors[edge_to],
#     device=DEVICE
# )

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

num_spatial_samples = int(np.sum(S_N_EPOCHS * probs[~is_temporal]) // 1)

# sampler = TemporalPreservingSampler(
#     weights=probs,
#     num_spatial_samples=num_spatial_samples,
#     is_temporal=is_temporal,
#     edge_from=edge_from,
#     edge_to=edge_to
# )
sampler = WeightedRandomSampler(probs, n_samples, replacement=True)

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
        
        print("高维空间中的前15个邻居:")
        print(f"{'邻居ID':<10}{'高维排名':<10}{'低维排名':<10}{'排名变化':<10}")
        for j, neighbor_id in enumerate(high_neighbors[:15]):
            low_rank = np.where(low_neighbors == neighbor_id)[0]
            if len(low_rank) > 0:
                low_rank = low_rank[0]
                print(f"{neighbor_id:<10}{j:<10}{low_rank:<10}{low_rank-j:<10}")
            else:
                print(f"{neighbor_id:<10}{j:<10}{'不在前k中':<10}{'':<10}")
        
        print("\n低维空间中的前15个邻居:")
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

################## evaluate ##################

"""
for t in range(epoch_start,epoch_end+1,epoch_period):
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
"""

print("\n==== 全局邻居排名保持性评估（所有epoch数据一起评估）====")

all_high_dim_data = []
all_embeddings = []
epoch_indices = []  # 记录每个数据点来自哪个epoch

for t in range(epoch_start, epoch_end+1, epoch_period):
    high_dim_data = data_provider.train_representation(epoch=t)
    if high_dim_data is None or len(high_dim_data) == 0:
        print(f"No data found for epoch {t}, skipping...")
        continue
    
    model.eval()
    with torch.no_grad():
        embedding = model.encoder(
            torch.from_numpy(high_dim_data).to(dtype=torch.float32, device=DEVICE)
        ).cpu().numpy()
    
    all_high_dim_data.append(high_dim_data)
    all_embeddings.append(embedding)
    epoch_indices.extend([t] * len(high_dim_data))

if len(all_high_dim_data) > 0:
    all_high_dim_data = np.vstack(all_high_dim_data)
    all_embeddings = np.vstack(all_embeddings)
    epoch_indices = np.array(epoch_indices)
    
    print(f"合并后的数据大小: {all_high_dim_data.shape}")
    
    # 计算全局的排名保持性
    ranking_score, kendall_score, k_preservation_rate = evaluate_proj_nn_ranking_preservation(
        all_high_dim_data, all_embeddings, n_neighbors=min(15, len(all_high_dim_data)-1), metric="euclidean"
    )
    print(f"全局Nearest Neighbor Ranking Preservation (Spearman): {ranking_score:.4f}")
    print(f"全局Nearest Neighbor Ranking Preservation (Kendall): {kendall_score:.4f}")
    print(f"全局K-Nearest Neighbor Preservation Rate: {k_preservation_rate:.4f}")
    
    show_sample_ranking_preservation(all_high_dim_data, all_embeddings, 
                                    n_neighbors=min(15, len(all_high_dim_data)-1), 
                                    n_samples=3)
else:
    print("没有找到有效的数据")

print("\n==== 样本时间轨迹的邻居排名保持性评估 ====")

if selected_idxs is not None:
    total_ranking_score = 0
    total_kendall_score = 0
    total_k_preservation_rate = 0
    valid_sample_count = 0

    for sample_idx in selected_idxs:
        print(f"分析样本 #{sample_idx} 的时间轨迹")
        sample_high_dim = []
        sample_embeddings = []
        valid_epochs = []
        
        for t in range(epoch_start, epoch_end+1, epoch_period):
            high_dim_data = data_provider.train_representation(epoch=t, select_sample=[sample_idx])
            if high_dim_data is None or len(high_dim_data) == 0:
                continue
                
            if len(high_dim_data) > 0:
                model.eval()
                with torch.no_grad():
                    embedding = model.encoder(
                        torch.from_numpy(high_dim_data).to(dtype=torch.float32, device=DEVICE)
                    ).cpu().numpy()
                
                sample_high_dim.append(high_dim_data)
                sample_embeddings.append(embedding)
                valid_epochs.append(t)
        
        if len(sample_high_dim) > 1:  # 需要至少两个时间点才能计算
            sample_high_dim = np.vstack(sample_high_dim)
            sample_embeddings = np.vstack(sample_embeddings)
            
            print(f"样本 #{sample_idx} 在 {len(valid_epochs)} 个epoch中出现")
            print(f"有效的epochs: {valid_epochs}")
            
            if len(sample_high_dim) >= 3:  # 至少需要3个点才能计算相关系数
                ranking_score, kendall_score, k_preservation_rate = evaluate_proj_nn_ranking_preservation(
                    sample_high_dim, sample_embeddings, 
                    n_neighbors=min(len(sample_high_dim)-1, 15), 
                    metric="euclidean"
                )
                print(f"样本 #{sample_idx} 轨迹的Nearest Neighbor Ranking Preservation (Spearman): {ranking_score:.4f}")
                print(f"样本 #{sample_idx} 轨迹的Nearest Neighbor Ranking Preservation (Kendall): {kendall_score:.4f}")
                print(f"样本 #{sample_idx} 轨迹的K-Nearest Neighbor Preservation Rate: {k_preservation_rate:.4f}")
                
                total_ranking_score += ranking_score
                total_kendall_score += kendall_score
                total_k_preservation_rate += k_preservation_rate
                valid_sample_count += 1
                
                plt.figure(figsize=(10, 8))
                plt.plot(sample_embeddings[:, 0], sample_embeddings[:, 1], 'b-', alpha=0.5)
                for i, epoch in enumerate(valid_epochs):
                    plt.scatter(sample_embeddings[i, 0], sample_embeddings[i, 1], c='r', s=100)
                    plt.text(sample_embeddings[i, 0], sample_embeddings[i, 1], f'E{epoch}', fontsize=12)
                
                plt.title(f"sample #{sample_idx} moving trace through epochs")
                plt.xlabel("Dimension 1")
                plt.ylabel("Dimension 2")
                plt.grid(True, linestyle='--', alpha=0.7)
                
                trajectory_dir = os.path.join(content_path, "visualization_results", "trajectories")
                os.makedirs(trajectory_dir, exist_ok=True)
                plt.savefig(os.path.join(trajectory_dir, f"sample_{sample_idx}_trajectory.png"), dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"样本 #{sample_idx} 的轨迹图已保存")
            else:
                print(f"样本 #{sample_idx} 的有效数据点不足以计算排名保持性 (需要至少3个点)")

    if valid_sample_count > 0:
        avg_ranking_score = total_ranking_score / valid_sample_count
        avg_kendall_score = total_kendall_score / valid_sample_count
        avg_k_preservation_rate = total_k_preservation_rate / valid_sample_count
        print(f"所有样本的平均 Nearest Neighbor Ranking Preservation (Spearman): {avg_ranking_score:.4f}")
        print(f"所有样本的平均 Nearest Neighbor Ranking Preservation (Kendall): {avg_kendall_score:.4f}")
        print(f"所有样本的平均 K-Nearest Neighbor Preservation Rate: {avg_k_preservation_rate:.4f}")
    else:
        print("没有足够的有效样本来计算平均值")


# print("\n==== 样本空间邻居保持性评估 ====")

# if selected_idxs is not None:
#     sample_spatial_metrics = {idx: {'spearman': [], 'kendall': [], 'knn_rate': []} for idx in selected_idxs}
    
#     for t in range(epoch_start, epoch_end+1, epoch_period):
#         print(f"\n分析 Epoch {t} 中所有样本的空间邻居保持性")
        
#         high_dim_data = data_provider.train_representation(epoch=t)
#         if high_dim_data is None or len(high_dim_data) == 0:
#             print(f"Epoch {t} 没有找到数据，跳过...")
#             continue
        
#         model.eval()
#         with torch.no_grad():
#             embedding = model.encoder(
#                 torch.from_numpy(high_dim_data).to(dtype=torch.float32, device=DEVICE)
#             ).cpu().numpy()
        
#         print(f"Epoch {t} 数据点数: {len(high_dim_data)}")
        
#         # 如果数据点太少，跳过计算
#         if len(high_dim_data) < 2:
#             print(f"Epoch {t} 数据点数量不足以进行有意义的分析（少于3个点）")
#             continue
        
#         for sample_idx in selected_idxs:
#             sample_data = data_provider.train_representation(epoch=t, select_sample=[sample_idx])
            
#             if sample_data is None or len(sample_data) == 0:
#                 print(f"样本 #{sample_idx} 在 Epoch {t} 中不存在")
#                 continue
            
#             # 计算当前样本的降维结果
#             model.eval()
#             with torch.no_grad():
#                 sample_embedding = model.encoder(
#                     torch.from_numpy(sample_data).to(dtype=torch.float32, device=DEVICE)
#                 ).cpu().numpy()
            
#             n_neighbors_actual = min(15, len(high_dim_data)-1)
            
#             from sklearn.metrics.pairwise import euclidean_distances
#             high_dim_distances = euclidean_distances(sample_data, high_dim_data)[0]
            
#             low_dim_distances = euclidean_distances(sample_embedding, embedding)[0]
            
#             self_idx = np.argmin(high_dim_distances)
            
#             high_masked = np.delete(high_dim_distances, self_idx)
#             low_masked = np.delete(low_dim_distances, self_idx)
            
#             original_indices = np.arange(len(high_dim_distances))
#             original_indices = np.delete(original_indices, self_idx)
            
#             high_sorted_indices = np.argsort(high_masked)[:n_neighbors_actual]
#             low_sorted_indices = np.argsort(low_masked)[:n_neighbors_actual]
            
#             high_neighbors = original_indices[high_sorted_indices]
#             low_neighbors = original_indices[low_sorted_indices]
            
#             ranks_high_in_low = []
#             for neighbor in high_neighbors:
#                 low_rank = np.where(low_neighbors == neighbor)[0]
#                 if len(low_rank) > 0:
#                     ranks_high_in_low.append(low_rank[0])
#                 else:
#                     ranks_high_in_low.append(n_neighbors_actual)  # 给一个比k大的排名
            
#             reference_ranks = np.arange(n_neighbors_actual)
#             spearman_coeff, _ = spearmanr(reference_ranks, ranks_high_in_low)
#             kendall_coeff, _ = kendalltau(reference_ranks, ranks_high_in_low)
            
#             # 计算K最近邻保持率
#             intersection = set(high_neighbors).intersection(set(low_neighbors))
#             knn_rate = len(intersection) / n_neighbors_actual
            
#             sample_spatial_metrics[sample_idx]['spearman'].append(spearman_coeff)
#             sample_spatial_metrics[sample_idx]['kendall'].append(kendall_coeff)
#             sample_spatial_metrics[sample_idx]['knn_rate'].append(knn_rate)
            
#             print(f"样本 #{sample_idx} 在 Epoch {t} 的空间邻居保持性: Spearman={spearman_coeff:.4f}, Kendall={kendall_coeff:.4f}, KNN保持率={knn_rate:.4f}")
            
#             # 计算每个样本在所有epoch上的平均空间邻居保持性
#             print("\n==== 各样本在所有epoch上的平均空间邻居保持性 ====")
#             print(f"{'样本ID':<10}{'平均Spearman':<15}{'平均Kendall':<15}{'平均KNN保持率':<15}{'有效Epoch数':<15}")
    
#     for sample_idx in selected_idxs:
#         spearman_scores = sample_spatial_metrics[sample_idx]['spearman']
#         kendall_scores = sample_spatial_metrics[sample_idx]['kendall']
#         knn_rates = sample_spatial_metrics[sample_idx]['knn_rate']
        
#         if len(spearman_scores) > 0:
#             avg_spearman = np.mean(spearman_scores)
#             avg_kendall = np.mean(kendall_scores)
#             avg_knn_rate = np.mean(knn_rates)
#             print(f"{sample_idx:<10}{avg_spearman:.4f}{'':<7}{avg_kendall:.4f}{'':<7}{avg_knn_rate:.4f}{'':<7}{len(spearman_scores):<15}")