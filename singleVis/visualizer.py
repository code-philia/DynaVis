import os
import torch
import numpy as np
import re
import matplotlib.pyplot as plt

class DataVisualizer:
    def __init__(self, data_provider, model, resolution=500, save_path="visualization_results"):
        self.data_provider = data_provider
        self.model = model
        self.resolution = resolution
        self.save_path = save_path
        self.model.eval()
        os.makedirs(self.save_path, exist_ok=True)

    def _get_epoch_plot_measures(self, epoch):
        data = self.data_provider.train_representation(epoch)
        data = torch.from_numpy(data).to(device=self.model.device, dtype=torch.float)
        self.model.to(device=self.model.device)
        embedded = self.model.encoder(data).cpu().detach().numpy()

        ebd_min = np.min(embedded, axis=0)
        ebd_max = np.max(embedded, axis=0)
        ebd_extent = ebd_max - ebd_min

        x_min, y_min = ebd_min - 0.1 * ebd_extent
        x_max, y_max = ebd_max + 0.1 * ebd_extent

        x_min = min(x_min, y_min)
        y_min = min(x_min, y_min)
        x_max = max(x_max, y_max)
        y_max = max(x_max, y_max)

        return x_min, y_min, x_max, y_max

    def plot(self, epoch):
        x_min, y_min, x_max, y_max = self._get_epoch_plot_measures(epoch)

        plt.figure(figsize=(8, 8))
        plt.title(f"Visualization of Epoch {epoch}")

        data = self.data_provider.train_representation(epoch)
        data = torch.from_numpy(data).to(device=self.model.device, dtype=torch.float)
        embedded = self.model.encoder(data).cpu().detach().numpy()

        plt.scatter(embedded[:, 0], embedded[:, 1], s=5, alpha=0.5)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        save_file = os.path.join(self.save_path, f"epoch_{epoch}.png")
        plt.savefig(save_file, dpi=300)
        plt.close()  

class IncrDataVisualizer:
    def __init__(self, data_providers, model, resolution=500, save_path="visualization_results"):
        self.data_providers = data_providers
        self.model = model
        self.resolution = resolution
        self.save_path = save_path
        self.model.eval()
        os.makedirs(self.save_path, exist_ok=True)

    def _get_epoch_plot_measures(self, epoch):
        all_embedded = []
        for data_provider in self.data_providers:
            data = data_provider.train_representation(epoch)
            data = torch.from_numpy(data).to(device=self.model.device, dtype=torch.float)
            self.model.to(device=self.model.device)
            embedded = self.model.encoder(data).cpu().detach().numpy()
            all_embedded.append(embedded)

        all_embedded = np.concatenate(all_embedded, axis=0)
        ebd_min = np.min(all_embedded, axis=0)
        ebd_max = np.max(all_embedded, axis=0)
        ebd_extent = ebd_max - ebd_min

        x_min, y_min = ebd_min - 0.1 * ebd_extent
        x_max, y_max = ebd_max + 0.1 * ebd_extent

        x_min = min(x_min, y_min)
        y_min = min(x_min, y_min)
        x_max = max(x_max, y_max)
        y_max = max(x_max, y_max)

        return x_min, y_min, x_max, y_max

    def plot(self, epoch):
        x_min, y_min, x_max, y_max = self._get_epoch_plot_measures(epoch)

        plt.figure(figsize=(8, 8))
        plt.title(f"Visualization of Epoch {epoch}")

        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Define a list of colors for different data providers
        seen_data = set()
        for idx, data_provider in enumerate(self.data_providers):
            print("="*100)
            print(idx)
            print("="*100)
            data = data_provider.train_representation(epoch)
            data = torch.from_numpy(data).to(device=self.model.device, dtype=torch.float)
            embedded = self.model.encoder(data).cpu().detach().numpy()

            for i, point in enumerate(data):
                point_tuple = tuple(point.cpu().numpy())
                if point_tuple not in seen_data:
                    seen_data.add(point_tuple)
                    plt.scatter(embedded[i, 0], embedded[i, 1], s=5, alpha=0.5, color=colors[idx % len(colors)], label=f'Provider {idx + 1}' if i == 0 else None)

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()

        save_file = os.path.join(self.save_path, f"epoch_{epoch}.png")
        plt.savefig(save_file, dpi=300)
        plt.close()  