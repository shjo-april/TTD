import os
os.environ['HF_HOME'] = './cache/huggingface'
os.environ['TORCH_HOME'] = './cache/torch'

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from .tcl import TCL, MaskCLIP

from .utils import resize_for_aspect_ratio, resize

from tools import torch_utils, transforms as T

class VLModel(nn.Module):
    def __init__(self, arch, backbone, device=torch.device('cuda:0'), pamr=False):
        super().__init__()

        if arch == 'MaskCLIP':
            self.model = MaskCLIP(backbone.replace('@', '/'), freeze_idx=11, pamr=pamr)
        elif arch == 'TCL':
            checkpoint = torch.load(f'./weights/{arch}_{backbone}.pt')
            self.model = TCL(pamr=pamr, **checkpoint['parameters'])
            self.model.load_state_dict(checkpoint['weights'], strict=False)
        else:
            raise ValueError(f'Please confirm {arch}')
        
        self.eval()
        self.to(device)

        self.arch = arch
        self.device = device

        self.output_dim = self.model.output_dim
        self.patch_size = self.model.patch_size
        
        self.denorm_fn = T.Denormalize()
        self.image_transform = T.Compose([T.Normalize(), T.ToTensor()])

    def to(self, device):
        super().to(device)
        self.device = device
        self.model = self.model.to(device)
        return self

    def forward(self, image: torch.Tensor, caption: str, tags: list, templates: list=['{}']):
        caption_emb = self.model.build_text_embedding([caption], ['{}'], self.device)
        tag_embs = self.model.build_text_embedding(tags, templates, self.device)
        text_embs = torch.cat([caption_emb, tag_embs], dim=0) # - self.model.build_text_embedding([''], templates, self.device)

        if self.arch == 'TCL': 
            clip_feat, tcl_feat = self.model.clip_image_encoder.forward_maskclip(image[None].to(self.device))
            heatmaps, image_emb = self.model.masker(tcl_feat, text_embs, clip_feat.shape[2:], get_image_emb=True)

        elif self.arch == 'MaskCLIP':
            clip_feat, _ = self.model.clip_image_encoder.forward_maskclip(image[None].to(self.device))
            image_emb = self.model.clip_image_encoder.clip_proj(clip_feat)

            image_emb = torch_utils.normalize(image_emb, dim=1, eps=1e-6)  # BCHW
            simmap = torch.einsum("b c h w, n c -> b n h w", image_emb, text_embs)
            heatmaps = torch.sigmoid((simmap - 0.25) * 10.0)
        
        else:
            raise NotImplementedError(f'Please implement a new architecture ({self.arch})')
        
        image_emb = F.adaptive_avg_pool2d(image_emb, (1, 1))[:, :, 0, 0]
        image_emb = F.normalize(image_emb, dim=1)
        tag_logits = torch.einsum("b c, n c -> b n", image_emb, tag_embs)

        return heatmaps[0], tag_logits[0]
    
    @torch.no_grad()
    def produce_text_embeddings(self, text='', templates: list=[]):
        if len(templates) == 0: templates.append('{}') # add a basic template for a caption, e.g., ['flame', 'smoke']
        if isinstance(text, str): text = [text] # for a caption or a single tag, .e.g., 'Cats are running'

        return self.model.build_text_embedding(text, templates, self.device)
    
    @torch.no_grad()
    def produce_image_embeddings(self, images: torch.Tensor):
        return self.model.generate_clip_features(images.to(self.device))
    
    @torch.no_grad()
    def produce_global_scores(self, image, text_embs):
        if self.arch == 'TCL': 
            clip_feat, image_emb = self.model.clip_image_encoder.forward_maskclip(image[None].to(self.device))
            image_emb, shape = self.model.masker.backbone(image_emb), clip_feat.shape[2:]
            image_emb = rearrange(image_emb[:, 1:], "B (H W) C -> B C H W", H=shape[0], W=shape[1])
            image_emb = self.model.masker.decoder(image_emb)

        elif self.arch == 'MaskCLIP':
            clip_feat, _ = self.model.clip_image_encoder.forward_maskclip(image[None].to(self.device))
            image_emb = self.model.clip_image_encoder.clip_proj(clip_feat)

        else:
            raise NotImplementedError(f'Please implement \"{self.arch}\" for scoring.')

        image_emb = F.adaptive_avg_pool2d(image_emb, (1, 1))[:, :, 0, 0]
        image_emb = torch_utils.normalize(image_emb, dim=1, eps=1e-6)
        scores = torch.einsum("b c, n c -> b n", image_emb, text_embs)[0]

        return scores.cpu().detach().numpy()

    @torch.no_grad()
    def produce_heatmaps(self, image: torch.Tensor, text_embs: torch.Tensor, scales: list=[1.0], hflip: bool=False, image_size: int=448, stride: int=2, mode=None):
        image = image.to(self.device)

        ms_heatmap = None
        oh, ow = image.shape[1:]
        
        if mode == 'sliding':
            image, image_size = resize_for_aspect_ratio(image, image_size)

            stride = image_size // stride
            grid_fn = lambda x: max(x-image_size+stride-1, 0) // stride+1
        
        ih, iw = image.shape[1:]

        # inference
        for scale in scales:
            if mode == 'sliding':
                images = image[None] if scale == 1.0 else resize(image[None], scale=scale)
                if hflip: images = torch.cat([images, images.flip(-1)], dim=0)

                b, _, h, w = images.shape
                
                heatmap = images.new_zeros((b, text_embs.shape[0], h, w))
                cntmap = images.new_zeros((b, 1, h, w))

                for i in range(b):
                    h_grids, w_grids = map(grid_fn, [h, w])
                    
                    for h_idx in range(h_grids):
                        for w_idx in range(w_grids):
                            y1, x1 = h_idx * stride, w_idx * stride
                            x2, y2 = min(x1 + image_size, w), min(y1 + image_size, h)
                            x1, y1 = max(x2 - image_size, 0), max(y2 - image_size, 0)

                            crop_heatmap = self.model.generate_masks(images[i:i+1, :, y1:y2, x1:x2], text_embs)
                            crop_heatmap = resize(crop_heatmap, (y2-y1, x2-x1))

                            heatmap[i:i+1, :, y1:y2, x1:x2] += crop_heatmap
                            cntmap[i:i+1, :, y1:y2, x1:x2] += 1

                if hflip:
                    heatmap[1] = heatmap[1].flip(-1)
                    cntmap[1] = cntmap[1].flip(-1)

                heatmap = torch.mean(heatmap / cntmap, dim=0)
            else:
                sh, sw = int(ih*scale), int(iw*scale)
                while sw % self.patch_size != 0: sw += 1
                while sh % self.patch_size != 0: sh += 1

                images = resize(image[None], (sh, sw))
                if hflip: images = torch.cat([images, images.flip(-1)], dim=0)

                heatmaps = self.model.generate_masks(images, text_embs)
                if hflip: heatmaps[1] = heatmaps[1].flip(-1)
                heatmap = heatmaps.mean(dim=0)
            
            # w/o sliding: torch.Size([1, 3, 448, 608]) torch.Size([2, 512]) torch.Size([2, 112, 152])
            # w/  sliding: torch.Size([1, 3, 448, 597]) torch.Size([2, 512]) torch.Size([2, 448, 597])
            
            if ms_heatmap is None: ms_heatmap = heatmap
            else: ms_heatmap += resize(heatmap, ms_heatmap.shape[1:])

        return resize(ms_heatmap / len(scales), (oh, ow))
