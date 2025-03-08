import torch
from torch import nn
import torch.nn.init as init
    
class SingleVisualizationModel(nn.Module):
    def __init__(self, input_dims, output_dims, units, hidden_layer=3, device='cpu'):
        super(SingleVisualizationModel, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.units = units
        self.hidden_layer = hidden_layer
        self.device = device
        self._init_autoencoder()
        
    # TODO find the best model architecture
    def _init_autoencoder(self):
        # self.encoder = nn.Sequential(
        #     nn.Linear(self.input_dims, self.units),
        #     nn.ReLU(True),
        #     nn.Linear(self.units, self.units),
        #     nn.ReLU(True),
        #     nn.Linear(self.units, self.units),
        #     nn.ReLU(True),
        #     nn.Linear(self.units, self.units),
        #     nn.ReLU(True),
        #     # nn.Linear(self.units, self.units),
        #     # nn.ReLU(True),
        #     nn.Linear(self.units, self.output_dims)
        # )
        # self.decoder = nn.Sequential(
        #     nn.Linear(self.output_dims, self.units),
        #     nn.ReLU(True),
        #     nn.Linear(self.units, self.units),
        #     nn.ReLU(True),
        #     nn.Linear(self.units, self.units),
        #     nn.ReLU(True),
        #     nn.Linear(self.units, self.units),
        #     nn.ReLU(True),
        #     # nn.Linear(self.units, self.units),
        #     # nn.ReLU(True),
        #     nn.Linear(self.units, self.input_dims)
        # )
        """
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dims, self.units),
            nn.ReLU(True))
        for h in range(self.hidden_layer):
            self.encoder.add_module("{}".format(2*h+2), nn.Linear(self.units, self.units))
            self.encoder.add_module("{}".format(2*h+3), nn.ReLU(True))
        self.encoder.add_module("{}".format(2*(self.hidden_layer+1)), nn.Linear(self.units, self.output_dims))

        self.decoder = nn.Sequential(
            nn.Linear(self.output_dims, self.units),
            nn.ReLU(True))
        for h in range(self.hidden_layer):
            self.decoder.add_module("{}".format(2*h+2), nn.Linear(self.units, self.units))
            self.decoder.add_module("{}".format(2*h+3), nn.ReLU(True))
        self.decoder.add_module("{}".format(2*(self.hidden_layer+1)), nn.Linear(self.units, self.input_dims))
        """

        ## Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dims, self.units),
            nn.BatchNorm1d(self.units),
            nn.ReLU(True)
        )
        for _ in range(self.hidden_layer):
            self.encoder.add_module("encoder_hidden_{}".format(_), nn.Linear(self.units, self.units))
            self.encoder.add_module("encoder_bn_{}".format(_), nn.BatchNorm1d(self.units))
            self.encoder.add_module("encoder_relu_{}".format(_), nn.ReLU(True))
        self.encoder.add_module("encoder_output", nn.Linear(self.units, self.output_dims))
        init.xavier_uniform_(self.encoder[-1].weight)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.output_dims, self.units),
            nn.BatchNorm1d(self.units),
            nn.ReLU(True)
        )
        for _ in range(self.hidden_layer):
            self.decoder.add_module("decoder_hidden_{}".format(_), nn.Linear(self.units, self.units))
            self.decoder.add_module("decoder_bn_{}".format(_), nn.BatchNorm1d(self.units))
            self.decoder.add_module("decoder_relu_{}".format(_), nn.ReLU(True))
        self.decoder.add_module("decoder_output", nn.Linear(self.units, self.input_dims))
        init.xavier_uniform_(self.decoder[-1].weight)

    def forward(self, edge_to, edge_from):
        #print("Edge to:", edge_to)
        #print("Edge from:", edge_from)
        embedding_to = self.encoder(edge_to)
        embedding_from = self.encoder(edge_from)
        recon_to = self.decoder(embedding_to)
        recon_from = self.decoder(embedding_from)
        #print("Embedding to:", embedding_to)
        #print("Embedding from:", embedding_from)
        #print("Recon to:", recon_to)
        #print("Recon from:", recon_from)
        #assert 1==2

        outputs = dict()
        outputs["umap"] = (embedding_to, embedding_from)
        outputs["recon"] = (recon_to, recon_from)

        return outputs
        
        """
        outputs = dict()
        embedding_to = self.encoder(edge_to)
        embedding_from = self.encoder(edge_from)
        recon_to = self.decoder(embedding_to)
        recon_from = self.decoder(embedding_from)
        
        outputs["umap"] = (embedding_to, embedding_from)
        outputs["recon"] = (recon_to, recon_from)

        return outputs
        """
