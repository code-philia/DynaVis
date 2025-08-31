import os
import time
import torch
import numpy as np
import json
import pickle
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
from singleVis.losses import UmapLoss, ReconLoss, SingleVisLoss, TemporalRankingLoss, UnifiedRankingLoss, TemporalVelocityLoss, AnchorLoss
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from singleVis.data_provider import DataProvider, NewDataProvider, BadNetDataProvider

def get_data_provider_and_config(dataset_name, selected_idxs, epoch_start, epoch_end, epoch_period):
    """根据数据集名称返回对应的数据提供器和配置"""
    if dataset_name == "codesearch":
        content_path = "/home/zicong/data/codesearch/dynavis"
        input_dims = 768
        data_provider = DataProvider(content_path, epoch_start, epoch_end, epoch_period, selected_idxs)
    elif dataset_name == "backdoor":
        content_path = "/home/zicong/data/backdoor_attack/dynamic/resnet18_MNIST_noise_salt_pepper_0.05_s0_t1/Model"
        input_dims = 512
        data_provider = BadNetDataProvider(content_path, epoch_start, epoch_end, epoch_period, selected_idxs)
    elif dataset_name == "badnet_noise":
        content_path = "/home/zicong/data/BadNet_MNIST_noise_salt_pepper_s0_t0/Model"
        input_dims = 512
        data_provider = BadNetDataProvider(content_path, epoch_start, epoch_end, epoch_period, selected_idxs)
    elif dataset_name == "casestudy":
        content_path = "/home/zicong/data/BadNet_MNIST_noise_salt_pepper_0.05_s0_t0/Model"
        input_dims = 512
        data_provider = BadNetDataProvider(content_path, epoch_start, epoch_end, epoch_period, selected_idxs)
    elif dataset_name == "test":
        content_path = "/home/zicong/Model"
        input_dims = 512
        data_provider = BadNetDataProvider(content_path, epoch_start, epoch_end, epoch_period, selected_idxs)
    elif dataset_name == "cifar10":
        content_path = "/home/zicong/data/CIFAR10/Model"
        input_dims = 512
        data_provider = DataProvider(content_path, epoch_start, epoch_end, epoch_period, selected_idxs)
    elif dataset_name == "fmnist":
        content_path = "/home/zicong/data/fminist_resnet18/Model"
        input_dims = 512
        data_provider = DataProvider(content_path, epoch_start, epoch_end, epoch_period, selected_idxs)
    else:
        raise ValueError(f"未知的数据集: {dataset_name}")
    
    return data_provider, content_path, input_dims

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="TimeVis+ Trajectory Analysis")
    parser.add_argument("--dataset", type=str, default="backdoor", help="数据集名称 (cifar10, fmnist, etc.)")
    parser.add_argument("--select_idxs", nargs="+", type=int, default=list(range(100)), help="选择的样本索引")
    parser.add_argument("--epoch_start", type=int, default=1, help="起始epoch")
    parser.add_argument("--epoch_end", type=int, default=50, help="结束epoch")
    parser.add_argument("--epoch_period", type=int, default=1, help="epoch间隔")
    parser.add_argument("--max_epochs", type=int, default=20, help="最大训练轮数")
    parser.add_argument("--patience", type=int, default=5, help="早停的patience")
    parser.add_argument("--anchor_weight", type=float, default=5.0, help="Anchor loss的权重")
    parser.add_argument("--save_dir", type=str, default=None, help="指定保存目录，默认使用数据集路径")
    return parser.parse_args()

def generate_trajectory_visualization(point_trajectories, trajectory_analysis, save_dir):
    """生成轨迹可视化图"""
    try:
        # 选择一些代表性的轨迹进行可视化
        analysis_results = trajectory_analysis["analysis_results"]
        
        if not analysis_results:
            print("没有有效的轨迹可供可视化。")
            return
        
        # 按总移动距离排序，选择前10个最活跃的轨迹
        sorted_trajectories = sorted(
            analysis_results.items(), 
            key=lambda x: x[1]["total_movement"], 
            reverse=True
        )
        
        top_trajectories = sorted_trajectories[:min(10, len(sorted_trajectories))]
        
        # 绘制轨迹图
        plt.figure(figsize=(15, 10))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_trajectories)))
        
        for i, (select_idx, analysis) in enumerate(top_trajectories):
            embeddings = analysis["embeddings"]
            x = [e[0] for e in embeddings]
            y = [e[1] for e in embeddings]
            plt.plot(x, y, marker='o', linestyle='-', color=colors[i], label=f'Point {select_idx}')
            plt.scatter(x[0], y[0], marker='^', s=100, color=colors[i], zorder=5)  # Start
            plt.scatter(x[-1], y[-1], marker='s', s=100, color=colors[i], zorder=5) # End
        
        plt.title("Top 10 Most Active Point Trajectories (TimeVis+)", fontsize=16)
        plt.xlabel("Dimension 1", fontsize=12)
        plt.ylabel("Dimension 2", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        trajectory_viz_file = os.path.join(save_dir, "top_trajectories_visualization.png")
        plt.savefig(trajectory_viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成移动距离分布图
        plt.figure(figsize=(12, 8))
        
        movements_all = []
        for analysis in trajectory_analysis["analysis_results"].values():
            movements_all.extend(analysis["movements"])
        
        if movements_all:
            plt.hist(movements_all, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title("Distribution of Movement Distances per Epoch", fontsize=16)
            plt.xlabel("Movement Distance", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.grid(True, alpha=0.3)
            movement_dist_file = os.path.join(save_dir, "movement_distribution.png")
            plt.savefig(movement_dist_file, dpi=300, bbox_inches='tight')
            plt.close()
        
        # 生成按select_idx排序的轨迹图
        plt.figure(figsize=(15, 10))
        
        # 按select_idx排序显示所有轨迹
        sorted_by_idx = sorted(analysis_results.items(), key=lambda x: int(x[0]))
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_by_idx)))
        
        for i, (select_idx, analysis) in enumerate(sorted_by_idx):
            embeddings = analysis["embeddings"]
            x = [e[0] for e in embeddings]
            y = [e[1] for e in embeddings]
            plt.plot(x, y, marker='.', linestyle='-', color=colors[i], label=f'Point {select_idx}')
        
        plt.title("All Trajectories by Select Index (TimeVis+)", fontsize=16)
        plt.xlabel("Dimension 1", fontsize=12)
        plt.ylabel("Dimension 2", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        all_trajectory_viz_file = os.path.join(save_dir, "all_trajectories_visualization.png")
        plt.savefig(all_trajectory_viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"轨迹可视化已保存到: {trajectory_viz_file}")
        if movements_all:
            print(f"移动距离分布图已保存到: {movement_dist_file}")
        print(f"所有轨迹图已保存到: {all_trajectory_viz_file}")
        
    except Exception as e:
        print(f"生成轨迹可视化时出错: {e}")

def generate_summary(save_dir, args, trajectory_analysis, method_name="TimeVis+"):
    """生成并保存汇总信息"""
    summary_file = os.path.join(save_dir, "trajectory_generation_summary.txt")
    try:
        with open(summary_file, 'w') as f:
            f.write("="*50 + "\n")
            f.write(f"{method_name} Trajectory Analysis Summary\n")
            f.write("="*50 + "\n\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Selected Indices ({len(args.select_idxs)}): {args.select_idxs}\n")
            f.write(f"Epoch Range: {args.epoch_start}-{args.epoch_end}\n")
            f.write(f"Epoch Period: {args.epoch_period}\n")
            f.write(f"Max Training Epochs: {args.max_epochs}\n")
            f.write(f"Anchor Loss Weight: {args.anchor_weight}\n\n")
            
            f.write("--- Analysis Results ---\n")
            f.write(f"Total points with trajectories: {trajectory_analysis['total_points']}\n")
            
            movements_all = []
            for analysis in trajectory_analysis["analysis_results"].values():
                movements_all.extend(analysis["movements"])
            
            if movements_all:
                f.write(f"Average movement per epoch (all points): {np.mean(movements_all):.6f}\n")
                f.write(f"Max movement per epoch (all points): {np.max(movements_all):.6f}\n")
                f.write(f"Min movement per epoch (all points): {np.min(movements_all):.6f}\n")
                f.write(f"Std dev of movement (all points): {np.std(movements_all):.6f}\n")
            else:
                f.write("No movement data available.\n")
        
        print(f"\n汇总信息已保存到: {summary_file}")
    except Exception as e:
        print(f"生成汇总信息时出错: {e}")

if __name__ == '__main__':
    args = parse_arguments()

    # Parameters from args
    selected_idxs = args.select_idxs
    epoch_start = args.epoch_start
    epoch_end = args.epoch_end
    epoch_period = args.epoch_period
    MAX_EPOCH = args.max_epochs
    PATIENT = args.patience
    ANCHOR_WEIGHT = args.anchor_weight
    
    # Get data provider and config
    data_provider, content_path, input_dims = get_data_provider_and_config(
        args.dataset, selected_idxs, epoch_start, epoch_end, epoch_period
    )

    if args.save_dir is not None:
        save_base_dir = args.save_dir
    else:
        save_base_dir = content_path
    
    analysis_root_dir = os.path.join(save_base_dir, "dynavis_trajectory")
    os.makedirs(analysis_root_dir, exist_ok=True)
    
    trajectory_save_dir = os.path.join(save_base_dir, f"trajectory_analysis_{len(selected_idxs)}")
    os.makedirs(trajectory_save_dir, exist_ok=True)

    split = 0
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
    S_N_EPOCHS = 5
    B_N_EPOCHS = 5
    T_N_EPOCHS = 100
    VARIANTS = "SVis"
    TEMP_TYPE = "local"
    SCHEDULE = None
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_samples = 1000
    SAVE_RESULTS = True

    print("="*100)
    print(f"Dataset: {args.dataset}")
    print(f"Content Path: {content_path}")
    print(f"Selected indices: {selected_idxs}")
    print(f"Epochs: {epoch_start}-{epoch_end}")
    print("="*100)

model = SingleVisualizationModel(
    input_dims=input_dims,
    output_dims=output_dims,
    units=units,
    hidden_layer=hidden_layer,
    device=DEVICE
)

model_path = "/home/zicong/data/backdoor_attack/dynamic/BadNet_MNIST_noise_salt_pepper_s0_t0/Model"
model_file = 'model_with_anchor.pth'
checkpoint = torch.load(os.path.join(model_path, model_file))
model.load_state_dict(checkpoint['state_dict'])
model.eval()
model = model.to(DEVICE)

################## 轨迹评估（不包含logits） ##################

print("\n==== 评估每个点的轨迹信息 ====")

results_dir = trajectory_save_dir # 使用新的目录

point_trajectories = {}  # {select_idx: {epoch: {"embedding": [x, y], "feature": [...]}}}
epoch_embeddings = {}  # {epoch: embeddings}
epoch_features = {}    # {epoch: features}

model.eval()
with torch.no_grad():
    # 获取可用的epochs
    available_epochs = []
    for t in range(epoch_start, epoch_end + 1, epoch_period):
        features = data_provider.train_representation(epoch=t)
        if features is not None and len(features) > 0:
            available_epochs.append(t)
    
    print(f"Found available epochs: {available_epochs}")

    for t in tqdm(available_epochs, desc="Processing epochs"):
        # 获取高维特征
        high_dim_data = data_provider.train_representation(epoch=t)
        if high_dim_data is None or len(high_dim_data) == 0:
            continue
        
        # 获取低维投影
        embedding = model.encoder(
            torch.from_numpy(high_dim_data).to(dtype=torch.float32, device=DEVICE)
        ).cpu().numpy()

        epoch_embeddings[t] = embedding
        epoch_features[t] = high_dim_data
        
        # 为每个数据点分配ID并记录轨迹
        for i, select_idx in enumerate(selected_idxs):
            if select_idx not in point_trajectories:
                point_trajectories[select_idx] = {}
            
            trajectory_data = {
                "embedding": embedding[i].tolist(),
                "feature": high_dim_data[i].tolist(),
                "epoch": t,
                "point_index": i
            }
            point_trajectories[select_idx][t] = trajectory_data

# 为每个点保存单独的 embedding 和 feature 文件
for select_idx, trajectory in point_trajectories.items():
    point_dir = os.path.join(results_dir, str(select_idx))
    os.makedirs(point_dir, exist_ok=True)
    
    epochs = sorted(trajectory.keys())
    point_embeddings = np.array([trajectory[e]["embedding"] for e in epochs])
    point_features = np.array([trajectory[e]["feature"] for e in epochs])
    
    np.save(os.path.join(point_dir, "embeddings.npy"), point_embeddings)
    np.save(os.path.join(point_dir, "features.npy"), point_features)

# 分析轨迹变化
print("Analyzing trajectory changes...")

trajectory_analysis = {
    "total_points": len(point_trajectories),
    "selected_idxs": selected_idxs,
    "epoch_range": available_epochs,
    "analysis_results": {}
}

for select_idx, trajectory in point_trajectories.items():
    epochs = sorted(trajectory.keys())
    if len(epochs) < 2:
        continue
    
    # 计算位置变化
    embeddings = [trajectory[e]["embedding"] for e in epochs]
    movements = []
    for i in range(1, len(embeddings)):
        movement = np.linalg.norm(np.array(embeddings[i]) - np.array(embeddings[i-1]))
        movements.append(movement)
    
    trajectory_analysis["analysis_results"][select_idx] = {
        "epochs_present": epochs,
        "total_movement": sum(movements),
        "avg_movement_per_epoch": np.mean(movements) if movements else 0,
        "max_movement": max(movements) if movements else 0,
        "min_movement": min(movements) if movements else 0,
        "movements": movements,
        "embeddings": embeddings,
        "trajectory": trajectory
    }

if SAVE_RESULTS:
    print("Saving trajectory results...")
    
    with open(os.path.join(results_dir, "point_trajectories.json"), "w") as f:
        json.dump(point_trajectories, f, indent=2)
    
    with open(os.path.join(results_dir, "trajectory_analysis.json"), "w") as f:
        json.dump(trajectory_analysis, f, indent=2)
    
    with open(os.path.join(results_dir, "point_trajectories.pkl"), "wb") as f:
        pickle.dump(point_trajectories, f)
    
    with open(os.path.join(results_dir, "trajectory_analysis.pkl"), "wb") as f:
        pickle.dump(trajectory_analysis, f)

    with open(os.path.join(results_dir, "all_epoch_embeddings.pkl"), "wb") as f:
        pickle.dump(epoch_embeddings, f)
    
    with open(os.path.join(results_dir, "all_epoch_features.pkl"), "wb") as f:
        pickle.dump(epoch_features, f)
    
    print(f"Trajectory results saved to {results_dir}")

# 生成轨迹可视化
print("生成轨迹可视化...")
generate_trajectory_visualization(point_trajectories, trajectory_analysis, results_dir)

# 生成汇总信息
print("生成汇总信息...")
generate_summary(results_dir, args, trajectory_analysis, method_name="TimeVis+")

# 打印统计信息
print(f"\n=== 轨迹分析统计 ===")
print(f"总数据点数: {trajectory_analysis['total_points']}")
print(f"有效轨迹数: {len(trajectory_analysis['analysis_results'])}")

movements_all = []
for analysis in trajectory_analysis["analysis_results"].values():
    movements_all.extend(analysis["movements"])

if movements_all:
    print(f"平均位置变化: {np.mean(movements_all):.6f}")
    print(f"最大位置变化: {np.max(movements_all):.6f}")
    print(f"最小位置变化: {np.min(movements_all):.6f}")
    print(f"位置变化标准差: {np.std(movements_all):.6f}")

print("Trajectory analysis complete!")# 分析轨迹变化
print("Analyzing trajectory changes...")

trajectory_analysis = {
    "total_points": len(point_trajectories),
    "epoch_range": list(range(epoch_start, epoch_end + 1, epoch_period)),
    "analysis_results": {}
}

for point_id, trajectory in point_trajectories.items():
    epochs = sorted(trajectory.keys())
    if len(epochs) < 2:
        continue
    
    # 计算位置变化
    embeddings = [trajectory[e]["embedding"] for e in epochs]
    movements = []
    for i in range(1, len(embeddings)):
        movement = np.linalg.norm(np.array(embeddings[i]) - np.array(embeddings[i-1]))
        movements.append(movement)
    
    trajectory_analysis["analysis_results"][str(point_id)] = {
        "epochs_present": epochs,
        "total_movement": sum(movements),
        "avg_movement_per_epoch": np.mean(movements) if movements else 0,
        "max_movement": max(movements) if movements else 0,
        "movements": movements,
        "embeddings": embeddings,
        "trajectory": trajectory
    }

# 保存结果
if SAVE_RESULTS:
    print("Saving trajectory results...")
    
    with open(os.path.join(results_dir, "point_trajectories.json"), "w") as f:
        json.dump(point_trajectories, f, indent=2)
    
    with open(os.path.join(results_dir, "trajectory_analysis.json"), "w") as f:
        json.dump(trajectory_analysis, f, indent=2)
    
    with open(os.path.join(results_dir, "point_trajectories.pkl"), "wb") as f:
        pickle.dump(point_trajectories, f)
    
    with open(os.path.join(results_dir, "trajectory_analysis.pkl"), "wb") as f:
        pickle.dump(trajectory_analysis, f)
    
    print(f"Trajectory results saved to {results_dir}")

# 打印统计信息
print(f"\n=== 轨迹分析统计 ===")
print(f"总数据点数: {trajectory_analysis['total_points']}")
print(f"有效轨迹数: {len(trajectory_analysis['analysis_results'])}")

movements_all = []
for analysis in trajectory_analysis["analysis_results"].values():
    movements_all.extend(analysis["movements"])

if movements_all:
    print(f"平均位置变化: {np.mean(movements_all):.6f}")
    print(f"最大位置变化: {np.max(movements_all):.6f}")
    print(f"最小位置变化: {np.min(movements_all):.6f}")

print("Trajectory analysis complete!")