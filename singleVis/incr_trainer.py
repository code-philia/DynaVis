import torch
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import time
from tqdm import tqdm
import numpy as np

"""
class SingleVisTrainer:
    def __init__(self, model, criterion, optimizer, lr_scheduler, edge_loader):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.edge_loader = edge_loader
        self._loss = 100.0

    def train_step(self):
        self.model.train()
        all_loss = []
        umap_losses = []
        recon_losses = []

        for data in self.edge_loader:
            edge_to, edge_from = data
            edge_to = edge_to.to(device=self.model.device, dtype=torch.float32)
            edge_from = edge_from.to(device=self.model.device, dtype=torch.float32)

            outputs = self.model(edge_to, edge_from)
            umap_l, recon_l, loss = self.criterion(edge_to, edge_from, outputs)
            all_loss.append(loss.item())
            umap_losses.append(umap_l.item())
            recon_losses.append(recon_l.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self._loss = sum(all_loss) / len(all_loss)
        self.lr_scheduler.step()

    def train(self, patience, max_epochs):
        patient = patience
        best_loss = float("inf")
        for epoch in range(max_epochs):
            print(f"Epoch {epoch+1}/{max_epochs}")
            self.train_step()

            if self._loss < best_loss:
                best_loss = self._loss
                patient = patience
            else:
                patient -= 1
                if patient <= 0:
                    print("Early stopping")
                    break
"""

class IncrementalVisTrainer:
    def __init__(self, model, previous_model, criterion, optimizer, lr_scheduler, edge_loader, previous_edge_loader, DEVICE):
        self.model = model
        self.previous_model = previous_model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.previous_edge_loader = previous_edge_loader
        self.DEVICE = DEVICE
        self.edge_loader = edge_loader
        self.consistency_loss_weight = 0.001
        self._loss = 100.0
        self.epoch = 0

    def train_step(self):
        self.model.train()
        all_loss = []
        umap_losses = []
        recon_losses = []
        consistency_losses = []

        t = tqdm(self.edge_loader, leave=True, total=len(self.edge_loader), desc=f"Epoch {self.epoch + 1}")

        for data in t:
            edge_to, edge_from = data
            edge_to = edge_to.to(device=self.DEVICE, dtype=torch.float32)
            edge_from = edge_from.to(device=self.DEVICE, dtype=torch.float32)

            outputs = self.model(edge_to, edge_from)
            umap_l, recon_l, loss = self.criterion(edge_to, edge_from, outputs)

            #current_embedding_to, current_embedding_from = outputs["umap"]
            #current_recon_to, current_recon_from = outputs["recon"]
            #outputs["umap"] = (embedding_to, embedding_from)
            #outputs["recon"] = (recon_to, recon_from)

            if self.previous_model is not None:
                # 使用 previous_edge_loader 计算一致性损失
                for prev_data in self.previous_edge_loader:
                    prev_edge_to, prev_edge_from = prev_data
                    prev_edge_to = prev_edge_to.to(device=self.DEVICE, dtype=torch.float32)
                    prev_edge_from = prev_edge_from.to(device=self.DEVICE, dtype=torch.float32)

                    with torch.no_grad():
                        prev_outputs = self.previous_model(prev_edge_to, prev_edge_from)
                        prev_embedding_to, prev_embedding_from = prev_outputs["umap"]
                        # prev_recon_to, prev_recon_from = prev_outputs["recon"]
                    
                    current_outputs = self.model(prev_edge_to, prev_edge_from)
                    current_embedding_to, current_embedding_from = current_outputs["umap"]
                    # current_recon_to, current_recon_from = current_outputs["recon"]

                    if current_embedding_to.size() == prev_embedding_to.size():
                        consistency_loss = torch.nn.functional.mse_loss(current_embedding_to, prev_embedding_to) + \
                                           torch.nn.functional.mse_loss(current_embedding_from, prev_embedding_from)
                        consistency_loss *= self.consistency_loss_weight
                        loss += consistency_loss
                        consistency_losses.append(consistency_loss.item())

            all_loss.append(loss.item())
            umap_losses.append(umap_l.item())
            recon_losses.append(recon_l.item())

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Add gradient clipping
            self.optimizer.step()

        self._loss = sum(all_loss) / len(all_loss)
        self.lr_scheduler.step()

        self.epoch_loss = self._loss
        self.epoch_umap_loss = sum(umap_losses) / len(umap_losses)
        self.epoch_recon_loss = sum(recon_losses) / len(recon_losses)
        if consistency_losses:
            self.epoch_consistency_loss = sum(consistency_losses) / len(consistency_losses)
        else:
            self.epoch_consistency_loss = 0.0

    def train(self, patience, max_epochs):
        patient = patience
        best_loss = float("inf")
        time_start = time.time()

        for epoch in range(max_epochs):
            print(f"Epoch {epoch + 1}/{max_epochs}")
            prev_loss = self._loss
            self.train_step()

            print(f"UMAP Loss: {self.epoch_umap_loss:.4f}, Recon Loss: {self.epoch_recon_loss:.4f}, Consistency Loss: {self.epoch_consistency_loss:.4f}, Total Loss: {self.epoch_loss:.4f}")

            if prev_loss - self._loss < 1E-2:
                if patient == 0:
                    break
                else:
                    patient -= 1
            else:
                patient = patience

            self.epoch += 1

        time_end = time.time()
        time_spend = time_end - time_start
        print(f"Time spend: {time_spend:.2f} seconds for training vis model...")