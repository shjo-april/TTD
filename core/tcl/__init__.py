import torch
from torch import nn
from torch.nn import functional as F

from einops import rearrange

from .clip import load, Tokenizer
from .encoders import CLIPImageEncoder
from .decoders import Masker

from ..refinements import PAMR

class MaskCLIP(nn.Module):
    def __init__(self, clip, freeze_idx, pamr=False):
        super().__init__()
        
        self.pretrained_clip = load(clip, device="cpu")
        self.pretrained_clip.eval()
        
        self.clip_text_encoder = self.pretrained_clip.encode_text
        self.clip_image_encoder = CLIPImageEncoder(self.pretrained_clip.visual, freeze_idx-1)
        
        self.output_dim = self.clip_image_encoder.output_dim
        self.patch_size = self.clip_image_encoder.patch_size

        self.text_transform = Tokenizer()
        
        self.pamr_fn = None
        if pamr: self.pamr_fn = PAMR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24])
    
    def build_text_embedding(self, text, templates, device, chunk_size=1024):
        """Build
        Args:
            text: str
            device: cpu or gpu

        Returns:
            text_embs [N, C]: text features
        """
        text_tokens = [torch.stack([self.text_transform(_t.format(t)) for _t in templates]) for t in text]
        text = torch.stack(text_tokens).to(device) # [NUM_CLASSES, NUM_TEMPLATES, CONTEXT_LENGTH] text tokens
        
        num_classes, num_templates = text.shape[:2]
        text = rearrange(text, 'n t l -> (n t) l', n=num_classes, t=num_templates)

        # chunked inference for memory limitation
        N = text.size(0)
        text_embs = torch.cat([self.clip_text_encoder(text[i:i+chunk_size]) for i in range(0, N, chunk_size)])

        text_embs = rearrange(text_embs, '(n t) c -> n t c', n=num_classes, t=num_templates) # [N, T, C]
        text_embs = text_embs.mean(dim=1) # [N, C]
        text_embs = F.normalize(text_embs, dim=1, eps=1e-6)

        return text_embs

    def generate_clip_features(self, image):
        """Generate CLIP image features

        Args:
            image [B, 3, H, W]
        
        Returns:
            image_emb [B, C]: image features
        """
        return self.clip_image_encoder.forward_clip(image)

    def generate_masks(self, image, text_emb):
        """Generate masks for each text embeddings

        Args:
            image [B, 3, H, W]
            text_emb [N, C]

        Returns:
            softmask [B, N, H, W]: softmasks for each text embeddings
        """
        clip_feat, _ = self.clip_image_encoder.forward_maskclip(image)
        image_emb = self.clip_image_encoder.clip_proj(clip_feat)
        
        norm_image_emb = F.normalize(image_emb, dim=1, eps=1e-6)  # BCHW
        norm_text_emb = F.normalize(text_emb, dim=1, eps=1e-6)    # NC
        simmap = torch.einsum("b c h w, n c -> b n h w", norm_image_emb, norm_text_emb)

        mask = torch.sigmoid((simmap - 0.25) * 10.0)

        if self.pamr_fn is not None:
            lr_image = F.interpolate(image, mask.shape[-2:], mode="bilinear", align_corners=True)
            mask = self.pamr_fn(lr_image, mask)

        return mask
        # return mask, image_emb

class TCL(MaskCLIP):
    def __init__(self, clip, freeze_idx, decoder, pamr=False, kp_w=0.3):
        super().__init__(clip, freeze_idx, pamr)
        
        self.kp_w = kp_w
        self.masker = Masker(
            self.clip_image_encoder.clone_backbone(freeze_idx),
            self.clip_image_encoder.clone_proj(),
            decoder
        )
    
    def generate_masks(self, image, text_emb):
        """Generate masks for each text embeddings

        Args:
            image [B, 3, H, W]
            text_emb [N, C]

        Returns:
            mask [B, N, H, W]: soft masks for each text embeddings
        """
        clip_feat, tcl_feat = self.clip_image_encoder.forward_maskclip(image)
        
        # torch.Size([1, 768, 28, 28]) torch.Size([785, 1, 768])
        # print(img_emb.shape, image_feat.shape); input()

        mask = self.masker(tcl_feat, text_emb, clip_feat.shape[2:])
        
        if self.pamr_fn is not None:
            lr_image = F.interpolate(image, mask.shape[-2:], mode="bilinear", align_corners=True)
            mask = self.pamr_fn(lr_image, mask)
        
        if self.kp_w > 0:
            # torch.Size([1, 768, 28, 28]) torch.Size([2, 512]) torch.Size([1, 2, 112, 112])
            # print(clip_feat.shape, text_emb.shape, mask.shape); input()
            mask = self.kp_branch(clip_feat, text_emb, mask, kp_w=self.kp_w)
        
        return mask
    
    def kp_branch(self, clip_feat, text_emb, ori_mask, kp_w):
        """Apply knowledge preservation branch proposed by TCL

        Args:
            clip_feat [B, C, H, W]
            text_emb [N, C]
            ori_mask [B, N, H, W]
            kp_w float

        Returns:
            mask [B, N, H, W] # a merged mask with CLIP and TCL masks
        """
        # assert self.masker.ignore_last_attn, "KP branch is only implemented for ignore_last_attn=True"
        image_emb = self.clip_image_encoder.clip_proj(clip_feat)
        
        norm_image_emb = F.normalize(image_emb, dim=1, eps=1e-6)  # BCHW
        norm_text_emb = F.normalize(text_emb, dim=1, eps=1e-6)    # NC
        simmap = torch.einsum("b c h w, n c -> b n h w", norm_image_emb, norm_text_emb)
        
        # kp mask
        mask = torch.sigmoid((simmap - 0.25) * 10.0)
        mask = F.interpolate(mask, ori_mask.shape[2:], mode='bilinear')

        # mix
        mask = kp_w * mask + (1. - kp_w) * ori_mask
        
        return mask
