import torch
import itertools
from torch import nn

class LoraInjectedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, r=4, dropout_p=0.1, scale=1.0):
        super().__init__()

        if r > min(in_features, out_features):
            raise ValueError(f"LoRA rank {r} must be less or equal than {min(in_features, out_features)}")
        
        self.r = r
        self.linear = nn.Linear(in_features, out_features, bias)
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.lora_up = nn.Linear(r, out_features, bias=False)
        self.scale = scale
        self.selector = nn.Identity()

        nn.init.normal_(self.lora_down.weight, std=1/r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, input):
        return self.linear(input) + self.scale * self.dropout(self.lora_up(self.selector(self.lora_down(input))))

class LoraInjectedConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, dilation=1, groups: int=1, bias: bool=True, r: int=4, dropout_p: float=0.1, scale: float=1.0):
        super().__init__()
        
        if r > min(in_channels, out_channels):
            raise ValueError(f"LoRA rank {r} must be less or equal than {min(in_channels, out_channels)}")
        
        self.r = r
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias,
        )
        self.lora_down = nn.Conv2d(
            in_channels=in_channels, out_channels=r,
            kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=False,
        )
        self.dropout = nn.Dropout(dropout_p)
        self.lora_up = nn.Conv2d(
            in_channels=r, out_channels=out_channels,
            kernel_size=1, stride=1, padding=0, bias=False,
        )
        self.selector = nn.Identity()
        self.scale = scale

        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, input):
        return self.conv(input) + self.scale * self.dropout(self.lora_up(self.selector(self.lora_down(input))))

def _find_modules_v2(model, ancestor_class=None, search_class=[], exclude_children_of=[LoraInjectedConv2d, LoraInjectedLinear]):
    """
    Find all modules of a certain class (or union of classes) that are direct or
    indirect descendants of other modules of a certain class (or union of classes).

    Returns all matching modules, along with the parent of those moduless and the
    names they are referenced by.
    """

    # Get the targets we should replace all linears under
    if ancestor_class is not None: ancestors = (module for module in model.modules() if module.__class__.__name__ in ancestor_class)
    # this, incase you want to naively iterate over all modules.
    else: ancestors = [module for module in model.modules()]
    
    # For each target find every linear_class module that isn't a child of a LoraInjectedLinear
    for ancestor in ancestors:
        for fullname, module in ancestor.named_modules():
            if any([isinstance(module, _class) for _class in search_class]):
                # Find the direct parent if this is a descendant, not a child, of target
                *path, name = fullname.split(".")

                parent = ancestor
                while path: parent = parent.get_submodule(path.pop(0))
                
                # Skip this linear if it's a child of a LoraInjectedLinear
                if exclude_children_of and any([isinstance(parent, _class) for _class in exclude_children_of]): continue
                
                # Otherwise, yield it
                yield parent, name, module

def apply_lora(model: nn.Module, r: int=4, ancestor_classes=None):
    params = []
    names = []

    search_classes = [nn.Linear, nn.Conv2d]
    
    for _module, name, _child_module in _find_modules_v2(model, ancestor_classes, search_classes):
        if _child_module.__class__ == nn.Linear:
            weight = _child_module.weight
            bias = _child_module.bias

            if r > min(_child_module.in_features, _child_module.out_features): _tmp = None
            else:
                _tmp = LoraInjectedLinear(_child_module.in_features, _child_module.out_features, _child_module.bias is not None, r=r)
                _tmp.linear.weight = weight
                if bias is not None: _tmp.linear.bias = bias
        
        elif _child_module.__class__ == nn.Conv2d:
            weight = _child_module.weight
            bias = _child_module.bias
            
            if r > min(_child_module.in_channels, _child_module.out_channels): _tmp = None
            else:
                _tmp = LoraInjectedConv2d(
                    _child_module.in_channels, _child_module.out_channels,
                    _child_module.kernel_size, _child_module.stride, _child_module.padding,
                    _child_module.dilation, _child_module.groups, _child_module.bias is not None, r=r,
                )
                _tmp.conv.weight = weight
                if bias is not None: _tmp.conv.bias = bias
        
        else:
            # print(_child_module.__class__, name); input()
            _tmp = None
            continue
        
        # switch the module
        if _tmp is not None:
            _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
            if bias is not None: _tmp.to(_child_module.bias.device).to(_child_module.bias.dtype)

            _module._modules[name] = _tmp
            _module._modules[name].lora_up.weight.requires_grad = True
            _module._modules[name].lora_down.weight.requires_grad = True
            
            names.append(name)
            params.append(_module._modules[name].lora_up.parameters())
            params.append(_module._modules[name].lora_down.parameters())
    
    return params, names

def get_layer_information(model):
    layer_dict = {}
    for module in model.modules():
        try: layer_dict[module.__class__.__name__] += 1
        except KeyError: layer_dict[module.__class__.__name__] = 1
    return layer_dict

def apply(model, arch, r=4):
    for param in model.parameters():
        param.requires_grad = False

    if arch == 'TCL': 
        ancestor_classes = ['ResidualAttentionBlock', 'MultiheadAttention', 'ResConv', 'ConvBlock', 'LayerNorm']
        lora_clip_params, _ = apply_lora(model.model.pretrained_clip, r, ancestor_classes)
        lora_tcl_params, _ = apply_lora(model.model.masker, r, ancestor_classes)
        lora_params = itertools.chain(*(lora_clip_params + lora_tcl_params))
    elif arch in ['MaskCLIP', 'CLIPSurgery']: 
        ancestor_classes = ['ResidualAttentionBlock', 'MultiheadAttention', 'LayerNorm']
        lora_params, _ = apply_lora(model.model.pretrained_clip, r, ancestor_classes)
        lora_params = itertools.chain(*lora_params)
    elif arch == 'SimSeg': 
        ancestor_classes = ['Block', 'Attention', 'Mlp', 'GELU', 'BertLayer', 'BertAttention', 'BertSelfAttention', 'BertIntermediate', 'SimpleProjection', 'LayerNorm']
        lora_params = itertools.chain(*lora_params)
        lora_params, _ = apply_lora(model.model, r, ancestor_classes)
    elif arch == 'SD': 
        ancestor_classes = ["ResnetBlock2D", "AttentionBlock", "Upsample2D", "CrossAttention", "Attention", "GEGLU"]
        lora_params, _ = apply_lora(model, r, ancestor_classes)
    else: 
        raise NotImplementedError(f'Please add the new arch. for LoRA ({arch})')

    return lora_params

def save(model, path):
    checkpoint = model.state_dict()
    for key in list(checkpoint.keys()):
        if not 'lora' in key: 
            del checkpoint[key]
    torch.save(checkpoint, path)

def load(model, path):
    model.load_state_dict(torch.load(path), strict=False)