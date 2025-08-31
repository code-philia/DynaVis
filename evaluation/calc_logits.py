import os
import torch
import numpy as np
import json
import pickle
import argparse
from torch import nn
from singleVis.data_provider import DataProvider, NewDataProvider, BadNetDataProvider
import torch.nn.functional as F
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="backdoor", help="Dataset name")
    parser.add_argument("--content_path", type=str, default=None, help="Data path, auto-determined if not provided")
    parser.add_argument("--selected_idxs", nargs="+", type=int, default=list(range(60000)), help="Selected indices")
    parser.add_argument("--epoch_start", type=int, default=1, help="Starting epoch")
    parser.add_argument("--epoch_end", type=int, default=50, help="Ending epoch")
    parser.add_argument("--epoch_period", type=int, default=1, help="Epoch period")
    parser.add_argument("--save_dir", type=str, default=None, help="Save directory")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")
    args = parser.parse_args()
    return args

def get_content_path(dataset_name):
    """根据数据集名称返回对应的路径"""
    paths = {
        "cifar10": "/home/zicong/data/CIFAR10/Model",
        "fmnist": "/home/zicong/data/fminist_resnet18/Model",
        "codesearch": "/home/zicong/data/codesearch/dynavis",
        "backdoor": "/home/zicong/data/backdoor_attack/dynamic/resnet18_MNIST_noise_salt_pepper_0.05_s0_t1/Model",
        "badnet_noise": "/home/zicong/data/BadNet_MNIST_noise_salt_pepper_s0_t0/Model",
        "casestudy": "/home/zicong/data/BadNet_MNIST_noise_salt_pepper_0.05_s0_t0/Model"
    }
    return paths.get(dataset_name, None)

def get_data_provider(dataset_name, content_path, epoch_start, epoch_end, epoch_period, selected_idxs):
    """根据数据集类型返回对应的数据提供器"""
    if dataset_name in ["cifar10", "fmnist"]:
        return DataProvider(content_path, epoch_start, epoch_end, epoch_period, selected_idxs)
    else:
        return BadNetDataProvider(content_path, epoch_start, epoch_end, epoch_period, selected_idxs)    

def load_model(model_path, num_classes, dataset, device):
    """加载指定epoch的模型"""
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
    
    import sys
    sys.path.append('/home/zicong/Project/PyTorch_CIFAR10')
    if dataset == "cifar10":
        from models.resnet_cifar10 import resnet18
    else:
        from models.resnet_for_1channel import resnet18

    model = resnet18(num_classes=num_classes)
    model = model.to(device)
    if dataset == "backdoor":
        import sys
        sys.path.append('/home/zicong/data/backdoor_attack/dynamic/resnet18_MNIST_noise_salt_pepper_0.05_s0_t1/Model')
        from backdoor_model import resnet18
        model = resnet18(num_classes=num_classes)
        model = model.to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    new_key = key[6:]  # 移除'model.'前缀
                else:
                    new_key = key
                new_state_dict[new_key] = value
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

def get_epoch_logits(model, dataset, epoch, device, selected_idxs):
    if dataset == "cifar10":
        img_file = "/home/zicong/data/CIFAR10/Training_data/training_dataset_data.pth"
    elif dataset == "fmnist":
        img_file = "/home/zicong/data/fminist_resnet18/Training_data/training_dataset_data.pth"
    elif dataset == "backdoor":
        img_file = "/home/zicong/data/backdoor_attack/dynamic/resnet18_MNIST_noise_salt_pepper_0.05_s0_t1/Training_data/training_dataset_data.pth"
    elif dataset == "badnet_noise":
        img_file = "/home/zicong/data/BadNet_MNIST_noise_salt_pepper_s0_t0/Training_data/training_dataset_data.pth"
    else:
        raise ValueError(f"不支持的数据集: {dataset}")
    
    if not os.path.exists(img_file):
        print(f"训练数据文件不存在: {img_file}")
        return None, None
    
    print(f"加载训练数据: {img_file}")
    images = torch.load(img_file, map_location='cpu')
    
    # print(f"原始数据形状: {images.shape}")
    # print(f"数据类型: {type(images)}")
    
    valid_indices = []
    
    for idx in selected_idxs:
        if 0 <= idx < len(images):
            valid_indices.append(idx)
        else:
            assert False, f"索引 {idx} 超出范围 (0, {len(images)-1})"
    
    if not valid_indices:
        print("没有有效的选择索引")
        return None, None
    
    selected_images = images[valid_indices]
    print(f"选择数据的形状: {selected_images.shape}")
    
    batch_size = 64
    all_logits = []
    
    model.eval()
    with torch.no_grad():
        num_batches = (len(selected_images) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(selected_images), batch_size), 
                        desc=f"Processing logits for epoch {epoch}", 
                        total=num_batches):
            batch_end = min(i + batch_size, len(selected_images))
            batch_data = selected_images[i:batch_end]  # [batch_size, 3, 32, 32]
            batch_data = batch_data.to(device, dtype=torch.float32)
            if batch_data.max() > 1.0:
                batch_data = batch_data / 255.0
            logits = model(batch_data)
            all_logits.append(logits.cpu().numpy())
    
    if all_logits:
        final_logits = np.concatenate(all_logits, axis=0)
        print(final_logits.shape)
        # print(f"Logits统计: min={final_logits.min():.4f}, max={final_logits.max():.4f}, mean={final_logits.mean():.4f}")
        return final_logits, valid_indices
    else:
        print("没有成功获取任何logits")
        return None, None
    

def main():
    args = parse_arguments()
    
    if args.content_path is not None:
        content_path = args.content_path
    else:
        content_path = get_content_path(args.dataset)
        if content_path is None:
            raise ValueError(f"未知的数据集: {args.dataset}")
    
    if args.save_dir is not None:
        save_dir = args.save_dir
    else:
        save_dir = os.path.join(content_path, "logits_data")
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*100)
    print(f"数据集: {args.dataset}")
    print(f"数据路径: {content_path}")
    print(f"Selected indices: {args.selected_idxs[:10]}...")
    print(f"Total selected indices: {len(args.selected_idxs)}")
    print(f"Epoch range: {args.epoch_start} - {args.epoch_end}")
    print(f"Number of classes: {args.num_classes}")
    print(f"保存目录: {save_dir}")
    print("="*100)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    data_provider = get_data_provider(
        args.dataset, content_path, args.epoch_start, 
        args.epoch_end, args.epoch_period, args.selected_idxs
    )
    
    all_epoch_logits = {}
    
    for epoch in range(args.epoch_start, args.epoch_end + 1, args.epoch_period):
        print(f"\n处理 Epoch {epoch}...")
        
        if args.dataset in ["cifar10", "fmnist"]:
            model_path = os.path.join(content_path, f"Epoch_{epoch}", "subject_model.pth")
            model = load_model(model_path, args.num_classes, args.dataset, device)
        else:
            model_path = os.path.join(content_path, f"Checkpoint_{epoch}", "subject_model.pth")
            model = load_model(model_path, args.num_classes, args.dataset, device)
        
        if model is None:
            print(f"跳过 Epoch {epoch}，无法加载模型")
            continue
        
        logits, valid_indices = get_epoch_logits(model, args.dataset, epoch, device, args.selected_idxs)
        
        if logits is None:
            print(f"跳过 Epoch {epoch}，无法获取logits")
            continue
        
        epoch_data = {
            "epoch": epoch,
            "logits": logits.tolist(),
            "valid_indices": valid_indices,
            "shape": logits.shape,
            "num_samples": len(logits),
            "num_classes": logits.shape[1]
        }
        
        all_epoch_logits[epoch] = epoch_data
        
        epoch_logits_file = os.path.join(save_dir, f"epoch_{epoch}_logits.npy")
        np.save(epoch_logits_file, logits)
        
        epoch_indices_file = os.path.join(save_dir, f"epoch_{epoch}_indices.npy")
        np.save(epoch_indices_file, np.array(valid_indices))
        
        print(f"Epoch {epoch} logits shape: {logits.shape}")
        print(f"保存到: {epoch_logits_file}")
        
        del model
        torch.cuda.empty_cache()
    
    print("\n保存完整的logits数据...")
    
    # 保存为JSON格式（方便查看但精度有限）
    logits_json_file = os.path.join(save_dir, "all_epoch_logits.json")
    with open(logits_json_file, "w") as f:
        json.dump(all_epoch_logits, f, indent=2)
    
    # 保存为pickle格式（保持完整精度）
    logits_pkl_file = os.path.join(save_dir, "all_epoch_logits.pkl")
    with open(logits_pkl_file, "wb") as f:
        pickle.dump(all_epoch_logits, f)
    
    summary = {
        "dataset": args.dataset,
        "content_path": content_path,
        "selected_indices_count": len(args.selected_idxs),
        "epoch_range": f"{args.epoch_start}-{args.epoch_end}",
        "num_classes": args.num_classes,
        "processed_epochs": list(all_epoch_logits.keys()),
        "total_epochs_processed": len(all_epoch_logits),
        "data_description": "Logits distribution for each data point in each epoch",
        "file_format": {
            "epoch_X_logits.npy": "numpy array of shape (num_samples, num_classes)",
            "epoch_X_indices.npy": "corresponding selected indices",
            "all_epoch_logits.pkl": "complete data in pickle format",
            "all_epoch_logits.json": "complete data in JSON format (limited precision)"
        }
    }
    
    summary_file = os.path.join(save_dir, "logits_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n所有结果已保存到: {save_dir}")
    
    print(f"\n=== Logits提取统计 ===")
    print(f"数据集: {args.dataset}")
    print(f"类别数: {args.num_classes}")
    print(f"选择的数据点数: {len(args.selected_idxs)}")
    print(f"处理的epoch数: {len(all_epoch_logits)}")
    print(f"成功处理的epochs: {sorted(all_epoch_logits.keys())}")
    
    if all_epoch_logits:
        sample_epoch = list(all_epoch_logits.keys())[0]
        sample_shape = all_epoch_logits[sample_epoch]["shape"]
        print(f"每个epoch的logits形状: {sample_shape}")
        print(f"数据格式: numpy数组 (样本数, 类别数)")
    
    print("\nLogits提取完成!")
    print(f"文件说明:")
    print(f"  - epoch_X_logits.npy: 每个epoch的logits数组")
    print(f"  - epoch_X_indices.npy: 对应的数据索引")
    print(f"  - all_epoch_logits.pkl: 完整数据(pickle格式)")
    print(f"  - logits_summary.json: 数据汇总信息")

if __name__ == "__main__":
    main()