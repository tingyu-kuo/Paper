import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet50
from torchvision.transforms.functional import affine


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



class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=1024):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim,),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.num_layers = 2
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
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 

class AttentionModule(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.q_fc = nn.Linear(in_features=dim, out_features=dim, bias=False)
        self.k_fc = nn.Linear(in_features=dim, out_features=dim, bias=False)
        self.v_fc = nn.Linear(in_features=dim, out_features=dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.bn = nn.BatchNorm1d(dim)
        # self.bn = nn.LayerNorm(dim, 1e-6)
        self.dropout = nn.Dropout(0.1)
        # self.relu = nn.LeakyReLU()
        # self.layer = nn.Linear(dim, dim)

    def forward(self, z):

        n = z.size(1) ** 0.5
        v = self.v_fc(z)
        k = self.k_fc(z)
        q = self.q_fc(z)
        
        q, k, v = q.unsqueeze(2), k.unsqueeze(2), v.unsqueeze(2)
        kt = k.transpose(1, 2)
        att = self.softmax(torch.matmul(q,kt/n))
        att = self.dropout(att)
        att = torch.matmul(att, v)     
        att = att + v
        att = att.squeeze(2)
        att = self.bn(att)

        return att


class SimSiam_att(nn.Module):
    def __init__(self, backbone=resnet50()):
        super().__init__()
        
        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim)

        self.encoder = nn.Sequential( # f encoder
            self.backbone,
            self.projector
        )
        self.predictor = prediction_MLP()
        self.self_attention = AttentionModule()
    
    def forward(self, x1, x2):
        # 想法: self attention 出來的 feature 作為 p，與 z 計算相似度
        #      ( 解釋: 經過 self attention module 後的 p，具有 z1 與 z2 之間的相關性，
        #        而 p1 要與 z2相似， p2 要與 z1 相似，因此 z1 與 z2 應具有彼此相關的信息)
        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        a1, a2 = self.self_attention(z1), self.self_attention(z2)

        p1, p2 = h(a1), h(a2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2

        return {'loss': L}



if __name__ == "__main__":
    model = SimSiam_att()
    x1 = torch.randn((2, 3, 224, 224))
    x2 = torch.randn_like(x1)

    model.forward(x1, x2).backward()
    print("forward backwork check")











