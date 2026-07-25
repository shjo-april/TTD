import copy
import math
import torch
import random
import numpy as np

from torch.nn import functional as F

normalize = F.normalize
einsum = torch.einsum

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def calculate_parameters(params):
    return sum(param.numel() for param in params)/1000000.0

def get_learning_rate(optimizer, option='first'):
    lr_list = [group['lr'] for group in optimizer.param_groups]
    if option == 'max': return max(lr_list)
    elif option == 'min': return min(lr_list)
    elif option == 'first': return lr_list[0]
    else: return lr_list

def normalize_masks(masks):
    flat_v = masks.reshape(masks.shape[0], -1)
    max_v, min_v = flat_v.max(dim=1)[0][:, None, None], flat_v.min(dim=1)[0][:, None, None]
    return (masks - min_v) / (max_v - min_v).clip(min=1e-5)

def resize(tensors, size=None, scale=1.0, mode='bilinear', align_corners=True):
    # for mask
    is_mask = len(tensors.size()) == 2
    if is_mask: tensors = tensors.float().unsqueeze(0)

    # for batch
    without_batch = len(tensors.size()) == 3
    if without_batch: tensors = tensors.unsqueeze(0)

    # resize
    if size is None: size = tensors.size()[2:]
    if scale != 1.: size = [int(s * scale) for s in size]
    if mode == 'nearest': align_corners = None
    
    _, _, h, w = tensors.size()
    if size[0] != h or size[1] != w:
        tensors = F.interpolate(tensors, size, mode=mode, align_corners=align_corners)
    
    # for batch
    if without_batch: tensors = tensors[0]

    # for mask
    if is_mask: tensors = tensors.long()[0]

    return tensors

class ExponentialMovingAverage:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        
        for p in self.ema.parameters():
            p.requires_grad_(False)

        self.decay = decay
    
    @torch.no_grad()
    def update(self, model):
        updated_params = 0
        msd = model.state_dict()

        for k, v in self.ema.state_dict().items():
            if 'lora' in k:
                v *= self.decay
                v += (1. - self.decay) * msd[k].detach()
                updated_params += 1

        return updated_params
    
    def get(self): return self.ema
    def to(self, device): self.ema.to(device)
