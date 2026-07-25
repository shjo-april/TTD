import os
import torch
import numpy as np

from core import networks, refinements, utils, lora_utils
from tools import cv_utils, io_utils, data_utils, transforms as T

def inference(args, image_paths, indices, device, visualize=False):
    with_bg_class = 'background' in args.dataset.class_names
    if with_bg_class: class_names = args.dataset.class_names[1:]
    else: class_names = args.dataset.class_names

    if visualize: progress = io_utils.Progress(len(indices), 'The last gpu')
    
    model = networks.VLModel(args.arch, args.backbone, device, args.pamr)

    if args.r > 0:
        lora_utils.apply(model, args.arch, args.r)
        lora_utils.load(model, args.lora)
    
    text_embs = model.produce_text_embeddings(class_names, utils.get_templates(args.template))
    
    for i in indices:
        if visualize: progress.update()

        image = model.image_transform(cv_utils.imread(image_paths[i], 'opencv')[..., ::-1])
        heatmaps = model.produce_heatmaps(image, text_embs, args.scales, args.hflip, args.image, args.stride, args.mode)
        
        for mask_dir, bg in args.mask_dirs:
            _heatmaps = heatmaps.clone()

            if with_bg_class:
                assert bg is not None, "Please set a background threshold."
                _heatmaps = utils.add_background(_heatmaps, bg)
            
            _heatmaps = _heatmaps.cpu().numpy()
            if args.crf: _heatmaps = args.crf(args.denorm_fn(image.cpu().numpy()), _heatmaps)
            
            image_name = os.path.basename(image_paths[i]).replace('.jpg', '.png')
            cv_utils.imwrite(io_utils.create_dir(mask_dir) + image_name, np.argmax(_heatmaps, axis=0), args.colors)

if __name__ == '__main__':
    # get arguments
    args= io_utils.Parser(
        {
            'gpus': '0', 'root': 'H:/', 'data': 'VOC2012', 'domain': 'validation',
            'arch': 'TCL', 'backbone': 'ViT-B@16', 'template': 'OpenAI',
            'scales': [1.0], 'hflip': False, 'pamr': False, 'crf': False, 
            'bg': [0.4], 'mode': 'sliding', 'image': 448, 'stride': 2,
            'r': 4, 'lora': './weights/TCL+TTD.pt', 'lora_tag': 'TTD'
        }
    ).get()

    # set gpus
    io_utils.set_gpus(args.gpus)

    # load a model
    model = networks.VLModel(args.arch, args.backbone, torch.device('cpu'), args.pamr)
    del model

    # prepare dataset and colors
    args.dataset = data_utils.Dataset(args.root + f'{args.data}/{args.data}.json')
    args.colors = cv_utils.get_colors(args.dataset.num_classes, args.dataset.ignore, data=args.data) # + 1 if args.data == 'Mapillary-Vistas' else 0

    # prepare a result directory
    if args.bg is None: # without background
        mask_dir = f'./results/{args.arch}@{args.data}@{args.template}@{str(args.scales)}@{args.image}@s={args.stride}/'
        if args.hflip: mask_dir = mask_dir[:-1] + '@hflip/'
        if args.mode != 'sliding': mask_dir = mask_dir[:-1] + f'@{args.mode}/'
        if args.pamr: mask_dir = mask_dir[:-1] + '@pamr/'
        if args.crf: mask_dir, args.denorm_fn, args.crf = mask_dir[:-1] + f'@crf/', T.Denormalize(), refinements.DenseCRF(for_seg=True)
        if args.r > 0: mask_dir = mask_dir[:-1] + f'@{args.lora_tag}/'
        args.mask_dirs = [(mask_dir, None)]
    
    else: # elif len(args.bg) > 0:
        args.mask_dirs = []
        for bg in map(float, args.bg):
            mask_dir = f'./results/{args.arch}@{args.data}@{args.template}@{str(args.scales)}@{args.image}@s={args.stride}/'
            if args.hflip: mask_dir = mask_dir[:-1] + '@hflip/'
            if args.mode != 'sliding': mask_dir = mask_dir[:-1] + f'@{args.mode}/'
            if args.pamr: mask_dir = mask_dir[:-1] + '@pamr/'
            if args.crf: mask_dir, args.denorm_fn, args.crf = mask_dir[:-1] + f'@crf/', T.Denormalize(), refinements.DenseCRF(for_seg=True)
            if args.r > 0: mask_dir = mask_dir[:-1] + f'@LoRA({args.lora_tag})/'
            if bg > 0: mask_dir = mask_dir[:-1] + f'@bg={bg:.02f}/'
            args.mask_dirs.append((mask_dir, bg))

    # split image paths
    image_paths = io_utils.listdir(args.root + f'{args.data}/{args.domain}/image/*.jpg')

    total, num_gpus = len(image_paths), len(args.gpus.split(','))
    total_gpu = total // num_gpus

    params = []
    indices = np.arange(total)

    for gi in range(num_gpus-1):
        param = [args, image_paths, indices[:total_gpu], torch.device('cuda', gi)]
        params.append(param); indices = indices[total_gpu:]

    param = [args, image_paths, indices, torch.device('cuda', num_gpus-1), True]
    params.append(param)

    io_utils.parallel(inference, params, len(params))

