import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision import models


def D(p, z): # negative cosine similarity
    z = z.detach() # stop gradient
    p = F.normalize(p, dim=1) # l2-normalize 
    z = F.normalize(z, dim=1) # l2-normalize 
    return  - F.cosine_similarity(p, z.detach(), dim=-1).mean()

def Xent(outputs, labels):
    criterion = nn.CrossEntropyLoss()
    Xent_loss = criterion(outputs, labels)
    preds = (outputs.argmax(dim=1) == labels).sum().item()
    accuracy = preds / outputs.shape[0]
    return Xent_loss, accuracy

def CosLoss(outputs, labels, num_classes=100):
    preds = (outputs.argmax(dim=1) == labels).sum().item()
    accuracy = preds / outputs.shape[0]
    labels = F.one_hot(labels, num_classes=num_classes)
    # outputs = nn.Softmax()(outputs)
    return - F.cosine_similarity(outputs, labels, dim=-1).mean(), accuracy


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.num_layers = 2

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 


class SimSiam(nn.Module):
    def __init__(self, backbone=models.resnet18(), num_classes=100):
        super().__init__()
        self.num_classes = num_classes
        output_dim = backbone.fc.in_features
        classifier = nn.Linear(in_features=output_dim, out_features=self.num_classes, bias=True)
        # remove fc layer from backbone
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # # build model
        self.projector = projection_MLP(output_dim)
        self.predictor = prediction_MLP()
        self.backbone = nn.Sequential(
            backbone,
            classifier
        )
    
    # for training
    def forward(self, x1, x2, labels=None):
        # f: extractor, c: classifier, g: projector, h: predictor
        f, c, g, h = self.backbone[0], self.backbone[1], self.projector, self.predictor
        y1, y2 = f(x1), f(x2)   # extraction
        z1, z2 = g(y1), g(y2)   # projection
        p1, p2 = h(z1), h(z2)   # prediction
        # outs = c(y1)            # classification
        outs = c(y1) / 2 + c(y2) / 2    # classification (average feature)
        
        L1 = D(p1, z2) / 2 + D(p2, z1) / 2
        L2, accuracy = Xent(outs, labels)
        # L2, accuracy = CosLoss(outs, labels, num_classes=self.num_classes)
        L = L1 + L2
        return L, [L1,L2], accuracy
    
    # for validation
    def valid(self, x, labels):
        outs = self.backbone(x)
        _, accuracy = Xent(outs, labels)
        return accuracy

    # for baseline
    def baseline(self, x, labels):
        out = self.backbone(x)
        L, accuracy = Xent(out, labels)
        return L, accuracy

# if __name__ == "__main__":
#     model = SimSiam()
#     # x1 = torch.randn((2, 3, 224, 224))
#     # x2 = torch.randn_like(x1)

#     # model.forward(x1, x2).backward()
#     # print("forward backwork check")

#     # z1 = torch.randn((200, 2560))
#     # z2 = torch.randn_like(z1)


# # Output:
# # tensor(-0.0010)
# # tensor(-0.0010)
























