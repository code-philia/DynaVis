import torch
from torch import nn
from .backend import convert_distance_to_probability, compute_cross_entropy
import numpy as np

"""
class UmapLoss(torch.nn.Module):
    def __init__(self, negative_sample_rate, device, a=1.0, b=1.0, repulsion_strength=1.0):
        super(UmapLoss, self).__init__()
        self._negative_sample_rate = negative_sample_rate
        self._a = a
        self._b = b
        self._repulsion_strength = repulsion_strength
        self.DEVICE = device

    def forward(self, embedding_to, embedding_from):
        batch_size = embedding_to.shape[0]
        embedding_neg_to = embedding_to.repeat_interleave(self._negative_sample_rate, dim=0)
        embedding_neg_from = embedding_from.repeat_interleave(self._negative_sample_rate, dim=0)
        randperm = torch.randperm(embedding_neg_from.shape[0])
        embedding_neg_from = embedding_neg_from[randperm]

        distance_embedding = torch.cat(
            (
                torch.norm(embedding_to - embedding_from, dim=1),
                torch.norm(embedding_neg_to - embedding_neg_from, dim=1)
            ),
            dim=0
        )

        probabilities_distance = 1.0 / (1.0 + self._a * distance_embedding ** (2 * self._b))
        probabilities_graph = torch.cat(
            (torch.ones(batch_size), torch.zeros(batch_size * self._negative_sample_rate)), dim=0
        ).to(self.DEVICE)

        attraction_loss = -torch.mean(probabilities_graph * torch.log(torch.clamp(probabilities_distance, 1e-12, 1.0)))
        repulsion_loss = -torch.mean((1.0 - probabilities_graph) * torch.log(torch.clamp(1.0 - probabilities_distance, 1e-12, 1.0)))
        loss = attraction_loss + self._repulsion_strength * repulsion_loss
        return loss
"""

class UmapLoss(nn.Module):
    def __init__(self, negative_sample_rate, device, a=1.0, b=1.0, repulsion_strength=1.0):
        super(UmapLoss, self).__init__()

        self._negative_sample_rate = negative_sample_rate
        self._a = a,
        self._b = b,
        self._repulsion_strength = repulsion_strength
        self.DEVICE = torch.device(device)

    @property
    def a(self):
        return self._a[0]

    @property
    def b(self):
        return self._b[0]

    def forward(self, embedding_to, embedding_from):
        batch_size = embedding_to.shape[0]
        # get negative samples
        embedding_neg_to = torch.repeat_interleave(embedding_to, self._negative_sample_rate, dim=0)
        repeat_neg = torch.repeat_interleave(embedding_from, self._negative_sample_rate, dim=0)
        randperm = torch.randperm(repeat_neg.shape[0])
        embedding_neg_from = repeat_neg[randperm]

        #  distances between samples (and negative samples)
        distance_embedding = torch.cat(
            (
                torch.norm(embedding_to - embedding_from, dim=1),
                torch.norm(embedding_neg_to - embedding_neg_from, dim=1),
            ),
            dim=0,
        )
        probabilities_distance = convert_distance_to_probability(
            distance_embedding, self.a, self.b
        )
        probabilities_distance = probabilities_distance.to(self.DEVICE)

        # set true probabilities based on negative sampling
        probabilities_graph = torch.cat(
            (torch.ones(batch_size), torch.zeros(batch_size * self._negative_sample_rate)), dim=0,
        )
        probabilities_graph = probabilities_graph.to(device=self.DEVICE)

        # compute cross entropy
        (_, _, ce_loss) = compute_cross_entropy(
            probabilities_graph,
            probabilities_distance,
            repulsion_strength=self._repulsion_strength,
        )

        return torch.mean(ce_loss)

class ReconLoss(torch.nn.Module):
    def __init__(self, beta=1.0):
        super(ReconLoss, self).__init__()
        self._beta = beta

    def forward(self, edge_to, edge_from, recon_to, recon_from):
        loss1 = torch.mean(torch.mean(torch.pow(edge_to - recon_to, 2), 1))
        loss2 = torch.mean(torch.mean(torch.pow(edge_from - recon_from, 2), 1))
        return (loss1 + loss2) / 2

class SingleVisLoss(torch.nn.Module):
    def __init__(self, umap_loss, recon_loss, lambd):
        super(SingleVisLoss, self).__init__()
        self.umap_loss = umap_loss
        self.recon_loss = recon_loss
        self.lambd = lambd

    def forward(self, edge_to, edge_from, outputs):
        embedding_to, embedding_from = outputs["umap"]
        umap_l = self.umap_loss(embedding_to, embedding_from)
        recon_to, recon_from = outputs["recon"]
        recon_l = self.recon_loss(edge_to, edge_from, recon_to, recon_from)
        total_loss = umap_l + self.lambd * recon_l
        return umap_l, recon_l, total_loss