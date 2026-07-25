import os
import torch
import numpy as np

from core import networks, utils, lora_utils
from tools import json_utils, io_utils, cv_utils, torch_utils, transforms as T

args = io_utils.Parser({
    'gpus': '0', 'root': 'H:/CC3M/', 
    'arch': 'TCL', 'image': 224, 'scales': [1.], 'hflip': False, 'template': 'OpenAI', 
    'stopwords': False, 'sliding': False, 'selection': 'pixel', 'fixed': False, 'visualize': False,
    'r': 4, 'lora': './weights/TCL+TTD.pt', 'lora_tag': 'TTD'
}).get()

# set gpus
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

templates = [] if len(args.template) == 0 else utils.get_templates(args.template)
test_transform = T.Compose([T.RandomResize(args.image, args.image) if not args.fixed else T.FixedResize(args.image), T.Normalize(), T.ToTensor()])

pred_tag = f'{args.arch}_{args.image}_{list(args.scales)}'
if args.hflip: pred_tag += f'@hflip'
if len(args.template) > 0: pred_tag += f'@{args.template}'
if args.stopwords: pred_tag += f'@stopwords'
if args.sliding: pred_tag += f'@sliding'
if args.fixed: pred_tag += f'@fixed'
if args.r > 0: pred_tag += f'@{args.lora_tag}'
pred_tag += f'@{args.selection}'

extractor = utils.SimpleTagExtraction()
model = networks.VLModel(args.arch, 'ViT-B@16', torch.device('cuda:0'), pamr=False)

if args.r > 0:
    lora_utils.apply(model, args.arch, args.r)
    lora_utils.load(model, args.lora)

for json_path in io_utils.progress(io_utils.listdir(args.root + 'json_refined/*/*.json')):
    json_path = json_path.replace('\\', '/')
    pred_path = json_path.replace('json_refined', f'json_{pred_tag}')
    if os.path.isfile(pred_path): continue

    data = json_utils.read(json_path, encoding='utf-8')
    caption, human_tags, ignore_tags = data['caption'], data['human'], data['ignore']
    tags = extractor.extract_tags_from_caption(caption, args.stopwords)
    
    # ignore uncertain samples
    if len(human_tags) == 0: continue
    elif data['incorrect']: continue

    if len(tags) > 0:
        # Tag Pruning: 1) extract tag scores
        cv_image = cv_utils.imread(json_path.replace('/json_refined', '/image').replace('.json', '.jpg'))
        image = test_transform({'image': cv_image[..., ::-1]})['image']

        text_embs = model.produce_text_embeddings(tags, templates)
        tags = np.asarray(tags)

        if args.selection == 'pixel':
            heatmaps = model.produce_heatmaps(image, text_embs, args.scales, args.hflip, mode='sliding' if args.sliding else None)
            scores = utils.get_scores(heatmaps).cpu().detach().numpy()
        
        elif args.selection == 'image':
            scores = model.produce_global_scores(image, text_embs)
            scores = (torch.from_numpy(scores) * 100).softmax(dim=0).numpy()
        
        elif args.selection == 'text':
            caption_emb = model.produce_text_embeddings(caption)
            scores = torch.einsum("b c, n c -> b n", caption_emb, text_embs)[0]
            scores = (scores * 100).softmax(dim=0)
            scores = scores.cpu().detach().numpy()
        
        elif args.selection == 'clip':
            image_emb = model.produce_image_embeddings(image[None])
            image_emb = torch_utils.normalize(image_emb, dim=1, eps=1e-6)
            scores = torch.einsum("b c, n c -> b n", image_emb, text_embs)[0]
            scores = scores.cpu().detach().numpy()

        elif args.selection == 'segmentation':
            heatmaps = model.produce_heatmaps(image, text_embs, args.scales, args.hflip, mode='sliding' if args.sliding else None)
            seg_mask = torch.argmax(heatmaps, dim=0)

            _, counts = torch.unique(seg_mask, return_counts=True)
            scores = counts / torch.sum(counts)
            
            scores = scores.cpu().detach().numpy()
        
        # Tag Pruning: 2) sort all tags by scores
        sorted_indices = np.argsort(scores)[::-1]
        tags = tags[sorted_indices]
        scores = scores[sorted_indices]

        if args.visualize:
            print()
            print(caption, tags)
            print(human_tags, ignore_tags)
            print(tags, scores)
            cv_utils.imshow('Image', cv_image, wait=0)
    else:
        tags = scores = []

    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    json_utils.write(pred_path, {'tags': list(tags), 'scores': list(map(float, scores))}, encoding='utf-8')
