import torch

from core import networks, utils, lora_utils
from tools import io_utils, cv_utils, torch_utils

args= io_utils.Parser(
    {
        'arch': 'TCL', 'backbone': 'ViT-B@16', 'template': 'OpenAI',
        'path': './images/PYH2024021306840005400_P4.jpg', 'tags': ['flame'], 
        'scales': [1.0], 'hflip': False, 'pamr': False, 'image': 448,
        'r': 4, 'lora': ''
    }
).get()

model = networks.VLModel(args.arch, 'ViT-B@16', torch.device('cuda:0'), args.pamr)
print('{} w/o LoRA: {:.1f}M'.format(args.arch, torch_utils.calculate_parameters(model.parameters())))

if len(args.lora) > 0:
    lora_params = lora_utils.apply(model, args.arch, args.r)
    print('{} w/  LoRA: {:.1f}M'.format(args.arch, torch_utils.calculate_parameters(model.parameters())))
    lora_utils.load(model, args.lora)

cv_image = cv_utils.imread(args.path, 'opencv')[..., ::-1]

text_embs = model.produce_text_embeddings(args.tags, utils.get_templates(args.template))
heatmaps = model.produce_heatmaps(model.image_transform(cv_image), text_embs, args.scales, args.hflip, args.image, mode='sliding')

scores = list(map(float, utils.get_scores(heatmaps).cpu().detach().numpy()))
tags = [f'{tag}: {score:.03f}' for tag, score in zip(args.tags, scores)]
demo = cv_utils.visualize_heatmaps(heatmaps, tags, cv_image[..., ::-1].copy())

cv_utils.imwrite(args.path.replace('./images', './figures'), demo)
cv_utils.imshow('Demo', demo, wait=0)