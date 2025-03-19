import os
import time
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.neighbors import NearestNeighbors
from singleVis.data_provider import DataProvider
from singleVis.spatial_edge_constructor import SpatialEdgeConstructor
from singleVis.temporal_edge_constructor import TemporalEdgeConstructor
from singleVis.visualization_model import SingleVisualizationModel
from singleVis.incr_trainer import IncrementalVisTrainer
from singleVis.backend import find_ab_params
from singleVis.visualizer import IncrDataVisualizer
from Project.TimeVisPlus.singleVis.losses_rank import SingleVisLoss, UmapLoss, ReconLoss
from singleVis.data_handler import DataHandler

# Parameters
content_path = "/home/zicong/data/Code_Retrieval_Samples/merged_train_data/"
epoch_start = 1
epoch_end = 10
epoch_period = 1
split = 0
input_dims = 768  # Adjust according to your data
output_dims = 2
units = 256
hidden_layer = 3
n_neighbors = 15
s_n_epochs = 100
b_n_epochs = 0
final_k = 0
persistence = 0
INIT_NUM = 100
ALPHA = 0.9
BETA = 0.5
MAX_EPOCH = 50
PATIENT = 5
S_N_EPOCHS = 50
B_N_EPOCHS = 5
T_N_EPOCHS = 5
VARIANTS = "SVis"
TEMP_TYPE = "local"
SCHEDULE = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_provider = DataProvider(
    content_path=content_path,
    epoch_start=1,
    epoch_end=10,
    epoch_period=1,
    split=0,
    selected_groups=None,
)

previous_edge_loader = None
current_data_providers = None

for group in range(5):
    print(f"Training group {group}...")
    data_provider.selected_groups = [group]
    save_dir = os.path.join(data_provider.content_path, "models")
    if current_data_providers == None:
        current_data_providers = [data_provider]
    else :
        current_data_providers.append(data_provider)

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

    edge_to = np.concatenate((s_edge_to, t_edge_to), axis=0)
    edge_from = np.concatenate((s_edge_from, t_edge_from), axis=0)
    probs = np.concatenate((s_probs, t_probs), axis=0)
    probs = probs / (probs.max() + 1e-3)
    eliminate_zeros = probs > 1e-3
    edge_to = edge_to[eliminate_zeros]
    edge_from = edge_from[eliminate_zeros]
    probs = probs[eliminate_zeros]

    dataset = DataHandler(edge_to, edge_from, feature_vectors)
    edge_loader = DataLoader(dataset, batch_size=1000, shuffle=True)

    model = SingleVisualizationModel(
        input_dims=768,
        output_dims=2,
        units=256,
        hidden_layer=3
    )
    model.to(DEVICE)

    a, b = find_ab_params(1.0, 0.1)
    umap_loss = UmapLoss(
        negative_sample_rate=5,
        device=DEVICE,
        a=a,
        b=b,
        repulsion_strength=1.0
    )
    recon_loss = ReconLoss(beta=1.0)
    criterion = SingleVisLoss(umap_loss, recon_loss, lambd=1.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    previous_model_path = os.path.join(save_dir, f"model_group_{group-1}.pth") if group > 0 else None
    previous_model = None
    if group > 0:
        previous_model = SingleVisualizationModel(
            input_dims=768,
            output_dims=2,
            units=256,
            hidden_layer=3
        )
        previous_model.load_state_dict(torch.load(previous_model_path))
        previous_model.to(DEVICE)
        previous_model.eval()

    trainer = IncrementalVisTrainer(
        model=model,
        previous_model=previous_model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        edge_loader=edge_loader,
        previous_edge_loader=previous_edge_loader,
        DEVICE=DEVICE
    )

    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"model_group_{group}.pth")
    trainer.train(patience=PATIENT, max_epochs=20)
    torch.save(model.state_dict(), model_path)

    previous_edge_loader = edge_loader

    if group == 0:
        continue
    print(len(current_data_providers))
    visualizer = IncrDataVisualizer(
        data_providers=current_data_providers,
        model=model,
        resolution=500,
        save_path=os.path.join(content_path, "incr_visualization_results", f"group_{group}")
    )

    for t in range(1,11,1):
        print(f"Processing epoch {t}")
        visualizer.plot(epoch=t)
        print(f"Epoch {t} visualization saved.")

print("All done!")