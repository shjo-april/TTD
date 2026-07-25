
import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange
from collections import OrderedDict

from .modules import GDecoder

class Sim2Mask(nn.Module):
    def __init__(self, init_w=1.0, init_b=0.0, learnable=True):
        super().__init__()
        self.init_w = init_w
        self.init_b = init_b
        self.learnable = learnable

        assert not ((init_w is None) ^ (init_b is None))
        if learnable:
            self.w = nn.Parameter(torch.full([], float(init_w)))
            self.b = nn.Parameter(torch.full([], float(init_b)))
        else:
            self.w = init_w
            self.b = init_b

    def forward(self, x):
        return torch.sigmoid(x * self.w + self.b)

    def extra_repr(self):
        return f'init_w={self.init_w}, init_b={self.init_b}, learnable={self.learnable}'

class TCLBackbone(nn.Module):
    """Replace frozen transformer layers with learnable weights
    """
    def __init__(self, transformer, freeze_idx):
        super().__init__()
        self.transformer = transformer
        self.transformer.resblocks = self.transformer.resblocks[freeze_idx:]
    
    def forward(self, x):
        x = self.transformer(x, ignore_last_attn=True)
        x = x.permute(1, 0, 2) # LND -> NLD
        return x

class Masker(nn.Module):
    def __init__(self, backbone, image_proj, decoder):
        super().__init__()

        self.backbone = backbone
        self.decoder = nn.Sequential(OrderedDict([
            ("decoder", GDecoder(**decoder)),
            ("image_proj", image_proj)
        ]))
        self.sim2mask = Sim2Mask()
    
    def forward(self, image_feat, text_emb, shape, get_image_emb=False):
        """Make mask by 1:N matching

        Args:
            image [B, 3, H, W]
            image_feat [L, B, C]: CLIP features
            text_emb [N, C]

        Return:
            mask [B, N, H', W'] (H' and W' are downsampled H/W)
        """
        image_emb = self.backbone(image_feat)
        image_emb = rearrange(image_emb[:, 1:], "B (H W) C -> B C H W", H=shape[0], W=shape[1])
        image_emb = self.decoder(image_emb)
        
        norm_image_emb = F.normalize(image_emb, dim=1, eps=1e-6) # BCHW
        norm_text_emb = F.normalize(text_emb, dim=1, eps=1e-6)   # NC
        simmap = torch.einsum("b c h w, n c -> b n h w", norm_image_emb, norm_text_emb)
        
        mask = self.sim2mask(simmap)
        
        if get_image_emb:
            return mask, image_emb
        else:
            return mask
