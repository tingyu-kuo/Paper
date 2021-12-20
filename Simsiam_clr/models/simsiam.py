import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet50


def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception

def Xent(outputs, labels):
    criterion = nn.CrossEntropyLoss()
    Xent_loss = criterion(outputs, labels)
    preds = (outputs.argmax(dim=1) == labels).sum().item()
    accuracy = preds / outputs.shape[0]
    return Xent_loss, accuracy


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=1024, out_dim=1024):
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
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

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
    def __init__(self, in_dim=1024, hidden_dim=512, out_dim=1024): # bottleneck structure
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
    def __init__(self, backbone=resnet50(), num_classes=100):
        super().__init__()
        output_dim = backbone.fc.in_features
        classifier = nn.Linear(in_features=output_dim, out_features=num_classes, bias=True)
        # remove fc layer from backbone
        backbone.fc = nn.Identity()
        self.projector = projection_MLP(output_dim)
        self.predictor = prediction_MLP()
        # add one fc layer
        self.backbone = nn.Sequential(
            backbone,
            classifier
        )
    
    # for training
    def forward(self, x1, x2, labels):
        # f: extractor, c: classifier, g: projector, h: predictor
        f, c, g, h = self.backbone[0], self.backbone[1], self.projector, self.predictor
        y1, y2 = f(x1), f(x2)
        z1, z2 = g(y1), g(y2)
        p1, p2 = h(z1), h(z2)
        outs = c(y1)
        
        L1 = D(p1, z2) / 2 + D(p2, z1) / 2
        L2, accuracy = Xent(outs, labels)
        L = L1 + L2
        return {'loss': L}, accuracy
    
    # for validation
    def valid(self, x, labels):
        outs = self.backbone(x)
        _, accuracy = Xent(outs, labels)
        return accuracy
        





if __name__ == "__main__":
    model = SimSiam()
    # x1 = torch.randn((2, 3, 224, 224))
    # x2 = torch.randn_like(x1)

    # model.forward(x1, x2).backward()
    # print("forward backwork check")

    # z1 = torch.randn((200, 2560))
    # z2 = torch.randn_like(z1)


# Output:
# tensor(-0.0010)
# tensor(-0.0010)












