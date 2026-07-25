import copy
from torch import nn

from einops import rearrange

from .modules import LNProjLayer
from .decoders import TCLBackbone

class CLIPImageEncoder(nn.Module):
    def __init__(self, clip_visual, freeze_idx):
        super().__init__()
        
        # update existing projection layers in CLIP
        self.clip_visual = clip_visual
        self.clip_proj = LNProjLayer(clip_visual.ln_post, clip_visual.proj)
        
        self.clip_visual.ln_post = nn.Identity()
        self.clip_visual.proj = None

        self.patch_size = self.clip_visual.patch_size
        self.output_dim = self.clip_visual.output_dim
        self.width = self.clip_visual.width

        # hook image features
        self._features = []
        
        for resblock in self.clip_visual.transformer.resblocks[freeze_idx:freeze_idx+1]:
            resblock.hook_handler = resblock.register_forward_hook(self.hook)
    
    def clear_features(self): self._features.clear()
    def hook(self, module, input, output): self._features.append(output)
    
    def clone_backbone(self, freeze_idx):
        return TCLBackbone(copy.deepcopy(self.clip_visual.transformer), freeze_idx)
    
    def clone_proj(self):
        return copy.deepcopy(self.clip_proj)

    def _encode(self, x, spatial=False):
        if spatial: 
            # print(x.shape) # [B, C, H, W]
            _H, _W = map(lambda x: x // self.patch_size, x.shape[-2:])
        
        x = self.clip_visual(x, spatial=spatial, ignore_last_attn=spatial)
        # print(x.shape) # [B, L, C]

        if spatial:
            _L = x.shape[1]
            assert _H*_W == _L-1, f"x: {x.shape}, L: {_L}, H: {_H}, W: {_W}"    
            x = rearrange(x[:, 1:], "B (H W) C -> B C H W", H=_H, W=_W)
        
        return x
    
    def forward(self, x, mode='clip'):
        self.clear_features()
        x = self._encode(x, mode != 'clip')

        if mode != 'clip': return x, self._features[0]
        else: self.clear_features(); return x

    def forward_clip(self, x):
        return self.clip_proj(self.forward(x))
    
    def forward_maskclip(self, x):
        return self.forward(x, 'maskclip')
