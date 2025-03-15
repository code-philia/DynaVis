import torch
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import time
from tqdm import tqdm
import numpy as np
import os

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

class SingleVisTrainer:
    def __init__(self, model, criterion, optimizer, lr_scheduler, edge_loader):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.DEVICE = self.model.device
        self.edge_loader = edge_loader
        self._loss = 100.0
        self.epoch = 0

    def train_step(self):
        self.model.train()
        all_loss = []
        umap_losses = []
        recon_losses = []
        rank_losses = []

        t = tqdm(self.edge_loader, leave=True, total=len(self.edge_loader), desc=f"Epoch {self.epoch + 1}")
        #flag = 0

        for data in t:
            edge_to, edge_from = data
            edge_to = edge_to.to(device=self.DEVICE, dtype=torch.float32)
            edge_from = edge_from.to(device=self.DEVICE, dtype=torch.float32)

            outputs = self.model(edge_to, edge_from)
            #print("Model outputs:", outputs["umap"], outputs["recon"])
            umap_l, recon_l, rank_l, loss= self.criterion(edge_to, edge_from, outputs)
            all_loss.append(loss.item())
            umap_losses.append(umap_l.item())
            recon_losses.append(recon_l.item())
            rank_losses.append(rank_l.item())
            #print(umap_l.item(), recon_l.item(), loss.item())
            #flag+=1
            #assert flag<=2

            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Add gradient clipping
            self.optimizer.step()

        self._loss = sum(all_loss) / len(all_loss)
        self.lr_scheduler.step()

        self.epoch_loss = self._loss
        self.epoch_umap_loss = sum(umap_losses) / len(umap_losses)
        self.epoch_recon_loss = sum(recon_losses) / len(recon_losses)
        self.epoch_rank_loss = sum(rank_losses) / len(rank_losses)

    def train(self, PATIENT, max_epochs):
        patient = PATIENT
        best_loss = float("inf")
        time_start = time.time()

        for epoch in range(max_epochs):
            print(f"Epoch {epoch + 1}/{max_epochs}")
            prev_loss = self._loss
            self.train_step()

            print(f"UMAP Loss: {self.epoch_umap_loss:.4f}, Recon Loss: {self.epoch_recon_loss:.4f}, rank Loss: {self.epoch_rank_loss:.4f}, Total Loss: {self.epoch_loss:.4f}")

            if prev_loss - self._loss < 1E-3:
                if patient == 0:
                    break
                else:
                    patient -= 1
            else:
                patient = PATIENT

            self.epoch += 1

        time_end = time.time()
        time_spend = time_end - time_start
        print(f"Time spend: {time_spend:.2f} seconds for training vis model...")
        
    def load(self, file_path):
        """
        save all parameters...
        :param name:
        :return:
        """
        save_model = torch.load(file_path, map_location="cpu")
        self._loss = save_model["loss"]
        self.model.load_state_dict(save_model["state_dict"])
        self.model.to(self.DEVICE)
        print("Successfully load visualization model...")

    def save(self, save_dir, file_name):
        """
        save all parameters...
        :param name:
        :return:
        """
        save_model = {
            "loss": self._loss,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()}
        save_path = os.path.join(save_dir, file_name + '.pth')
        torch.save(save_model, save_path)
        print("Successfully save visualization model...")