import torch.nn as nn
from siamese_network.encoder import SiameseEncoder

class SiamseNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = SiameseEncoder()

    def forward(self, image1, image2):
        embedding1 = self.encoder(image1)
        embedding2 = self.encoder(image2)

        return embedding1, embedding2
    
    def get_embedding(self, image):
        return self.encoder(image)