"""
ViT tutorial main model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from graphs.models.resnet_ViT import ResNetStage
from torch.nn.utils import weight_norm

class PatchEmbedding(nn.Module):
    def __init__(self, im_size, p_size, d_emb, in_channels):
        super().__init__()
        self.projection = nn.Sequential(
                nn.Conv2d(in_channels= in_channels, out_channels=d_emb, stride=p_size, kernel_size=p_size),
                Rearrange('b e h w -> b (h w) e'))
        self.cls_token = nn.Parameter(torch.randn(1,1,d_emb))
        self.positions = nn.Parameter(torch.randn((im_size//p_size)**2+1, d_emb ))

    def forward(self,x):
        b, _, _,_ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b = b)
        x = torch.cat([cls_tokens, x],dim=1)
        x += self.positions
        return x 

class MultiHeadAttention(nn.Module):
    def __init__(self, d_emb: int, n_heads: int, dropout: float=0):
        super().__init__()
        self.d_emb = d_emb
        self.n_heads = n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_emb, 3*d_emb)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(d_emb, d_emb)

    def forward(self, x):
        qkv = rearrange(self.qkv(x), 'b n (h d qkv) -> qkv b h n d', h=self.n_heads, qkv=3)
        query = qkv[0]
        key = qkv[1]
        value = qkv[2]
        score = torch.einsum('bhqd, bhkd -> bhqk', query, key)
        scaling = self.d_emb**0.5
        att = F.softmax(score, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over third axis
        out = torch.einsum('bhqk, bhkd -> bhqd', att, value)
        out = rearrange(out,'b h n d -> b n (h d)')
        out = self.projection(out)
        return out

class FeedForwardBlock(nn.Module):
    def __init__(self, d_emb:int, d_hid:int, dropout: float=0.):
        super().__init__()
        self.model = nn.Sequential(
                nn.Linear(d_emb, d_hid),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_hid, d_emb))
    def forward(self,x):
        return self.model(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                d_emb:int,
                n_heads: int,
                dropout:float,
                d_hid:int):
        super().__init__()
        self.MultiHeadAttention = MultiHeadAttention(d_emb=d_emb, n_heads=n_heads, dropout=dropout)
        self.FeedForwardBlock = FeedForwardBlock(d_emb = d_emb, d_hid = d_hid, dropout = dropout)
        self.norm = nn.LayerNorm(d_emb)
        self.dropout = nn.Dropout(dropout)

    def sa_block(self,x):
        res = x
        x = self.norm(x)
        x = self.MultiHeadAttention(x)
        x = self.dropout(x)
        x+=res
        return x

    def ff_block(self,x):
        res = x
        x = self.norm(x)
        x = self.FeedForwardBlock(x)
        x = self.dropout(x)
        x += res
        return x

    def forward(self,x):
        x = self.sa_block(x)
        x = self.ff_block(x)
        return x

class TransformerEncoder(nn.Sequential):
        def __init__(self, depth: int = 12, **kwargs):
                    super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

class ClassificationHead(nn.Module):
    """
    |classifier : either 'token' or 'gap'|
    """
    def __init__(self,d_emb:int, n_classes:int, classifier: str='token'):
        super().__init__()
        self.model = nn.Sequential(nn.LayerNorm(d_emb),
                                    nn.Linear(d_emb, n_classes))
        self.classifier = classifier

    def forward(self, x):
        if self.classifier == 'token':
            x = x[:,0]
        elif self.classifier == 'gap':
            x = reduce(x[:,1:x.size(1)-1,:], 'b n e -> b e', reduction='mean')
        return self.model(x)

    
class ViT(nn.Module):
    """
    Vision Transformer (ViT)
    args: 
        in_channels:int,
        im_size: int=384,
        n_heads:int=8, 
        d_hid:int=2048, 
        d_emb:int=768, 
        p_size:int=16,
        depth:int=12,
        dropout:float=0.1,
        is_resnet:bool=False
        ----
        in_classes : number of in channels
        n_classes: number of classes
        n_heads: number of attention heads
        d_hid : hidden layer dimension
        d_emb : embedding dimension
        p_size: patch size
        dropout: float
    """
    def __init__(self,
                 n_classes:int=10,
                 in_channels:int=3,
                 im_size: int=384,
                 n_heads:int=8, 
                 d_hid:int=2048, 
                 d_emb:int=768, 
                 p_size:int=16,
                 depth:int=12,
                 dropout:float=0.1,
                 classifier:str='token',
                 is_resnet:bool=False):
        super().__init__()
        if is_resnet:
            im_size = im_size // 4
            width = int(64)
        self.num_resnet_layers= False
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.n_heads = n_heads
        self.d_hid = d_hid
        self.d_emb = d_emb
        self.p_size = p_size
        self.dropout = dropout
        if is_resnet:
            self.PatchEmbedding = PatchEmbedding(in_channels=width, im_size = im_size, p_size = p_size, d_emb = d_emb)
        else:
            self.PatchEmbedding = PatchEmbedding(in_channels=in_channels, im_size = im_size, p_size = p_size, d_emb = d_emb)
        self.TransformerEncoder = TransformerEncoder(depth =depth, d_emb =d_emb, n_heads= n_heads, d_hid = d_hid, dropout = dropout)
        self.ClassificationHead = ClassificationHead(d_emb = d_emb, n_classes=n_classes, classifier=classifier)
        self.is_resnet= is_resnet
        if is_resnet:
            self.resnet = nn.Sequential(
                    weight_norm(nn.Conv2d(in_channels, width, kernel_size=(7,7), stride=(2,2) ,bias=False, padding='valid')),
                    nn.GroupNorm(num_groups=width, num_channels=width),
                    nn.ReLU(),
                    nn.MaxPool2d(stride=(2,2), kernel_size=(2,2)))
            if self.num_resnet_layers:
                self.resnet.append(ResNetStage(in_channels=width,block_size = self.num_resnet_layers[0], nout = width, first_stride=(1,1)))
                for i, block_size in enumerate(self.num_resnet_layers[1:],1):
                    self.resnet.append(ResNetStage(in_channels=4*width*(2**(i-1)),block_size = block_size, nout = width*(2**i), first_stride=(1,1)))

            
#            resnet50 = torch.hub.load('pytorch/vision:v0.10.0',
#                                    'resnet50',
#                                    pretrained=True)
#            self.resnet = nn.Sequential(resnet50.conv1,
#                                    resnet50.bn1,
#                                    resnet50.relu,
#                                    nn.MaxPool2d(kernel_size = (2,2), stride=(2,2))) # batch x 256, 96, 96
        
    def forward(self, x):
        if self.is_resnet == True:
            x = self.resnet(x)
        x = self.PatchEmbedding(x)
        x = self.TransformerEncoder(x)
        x = self.ClassificationHead(x)
        return x
