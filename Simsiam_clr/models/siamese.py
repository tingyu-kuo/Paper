# from turtle import forward
import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F


def D(z, p):
    return 1 - F.cosine_similarity(z.detach(), p, dim=-1).mean()

class Siamese(nn.Module):
    def __init__(self, model=models.resnet50()):
        super().__init__()
        dim = model.fc.in_features
        model.fc = nn.Identity()

        self.backbone = model

        self.projector = nn.Sequential(
            nn.Linear(dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
        )
        self.predictor = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2048)
        )
        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )
    
    def forward(self, x1, x2):
        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)

        L = (D(z1, p2) + D(z2, p1)) / 2
        return L
    
    def encoder_eval(self, x):
        f = self.encoder[0]
        x = f(x)
        return x

    



if __name__ == '__main__':
    model = Siamese(backbone=models.resnet18())
    x1 = torch.randn((8, 3, 224, 224))
    x2 = torch.randn((8, 3, 224, 224))
    x = torch.randn((8, 3, 224, 224))
    labels = torch.randint(0, 100, (8,))
    # print(model)
    model.forward(x1, x2)
    print("success")
