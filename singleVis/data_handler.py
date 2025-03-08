from torch.utils.data import Dataset
from PIL import Image

class DataHandler(Dataset):
    def __init__(self, edge_to, edge_from, feature_vector, transform=None):
        self.edge_to = edge_to
        self.edge_from = edge_from
        self.data = feature_vector
        self.transform = transform

    def __getitem__(self, item):

        edge_to_idx = self.edge_to[item]
        edge_from_idx = self.edge_from[item]
        edge_to = self.data[edge_to_idx]
        edge_from = self.data[edge_from_idx]
        if self.transform is not None:
            # TODO correct or not?
            edge_to = Image.fromarray(edge_to)
            edge_to = self.transform(edge_to)
            edge_from = Image.fromarray(edge_from)
            edge_from = self.transform(edge_from)
        return edge_to, edge_from

    def __len__(self):
        # return the number of all edges
        return len(self.edge_to)

class IncrDataHandler:
    def __init__(self, edge_to, edge_from, feature_vectors, groups, attention=None):
        self.edge_to = edge_to
        self.edge_from = edge_from
        self.feature_vectors = feature_vectors
        self.groups = groups
        self.attention = attention

    def __getitem__(self, index):
        return self.edge_to[index], self.edge_from[index], self.groups[index]
    
    def __len__(self):
        # return the number of all edges
        return len(self.edge_to)