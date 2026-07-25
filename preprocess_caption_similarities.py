import os
import cv2
import torch
import numpy as np

from tqdm import tqdm

from core import networks, utils
from tools import io_utils, cv_utils, json_utils, transforms as T

def inference(args, image_paths, json_paths, score_paths, indices, device, visualize):
    tagger = utils.SimpleTagExtraction()
    model = networks.VLModel(args.arch, args.backbone, device, args.pamr)
    
    image_transform = T.Compose([T.FixedResize(args.image), T.Normalize(), T.ToTensor()])
    if visualize: pbar = tqdm(total=len(indices), desc='Inference (the last process)')
    
    for index in indices:
        if visualize: pbar.update(1)

        image_path = image_paths[index]
        json_path = json_paths[index]
        score_path = score_paths[index]
        
        # preprocessing for image
        image = cv_utils.imread(image_path, 'opencv')
        if image is None: continue # image is empty

        h, w = image.shape[:2]
        if h < 64 or w < 64: continue # image size is too small

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) # convert Gray to BGR
        
        # preprocessing for text
        caption = json_utils.read(json_path, 'utf-8')['caption']
        if len(caption) == 0: continue
        
        tags = tagger.extract_tags_from_caption(caption, remove_stopwords=True)
        if len(tags) == 0: continue
        
        # inference
        image = image_transform(image[..., ::-1]).to(model.device)
        caption_emb = model.produce_text_embeddings(caption)

        heatmaps = model.produce_heatmaps(image, caption_emb, args.scales, args.hflip)
        scores = utils.get_scores(heatmaps).cpu().detach().numpy()
        
        tags = np.asarray(tags)
        caption_score = scores[0]

        # save a torch tensor
        json_utils.write(score_path,
            {
                'caption': caption,
                'score': float(caption_score), 
                'tags': list(map(str, tags)),
            }
        )

if __name__ == '__main__':
    args = io_utils.Parser(
        {
            'gpus': [0], 'data': 'cc3m', 'domain': 'train', 'root': '../dataset/',
            'arch': 'TCL', 'backbone': 'ViT-B@16', 'image': 224, 'template': 'OpenAI', 
            'scales': [1.0], 'hflip': False, 'pamr': False
        }
    ).get()
    
    # set gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(list(map(str, args.gpus)))
    
    # prepare directories
    args.temp = f'./temp/{args.arch}/{args.data}/{args.domain}/'
    
    image_paths = []
    json_paths = []
    score_paths = []

    for image_path in io_utils.progress(io_utils.listdir(args.root + f'{args.data}/{args.domain}/image/*/*.jpg')):
        image_path = image_path.replace('\\', '/')
        json_path = image_path.replace('/image', '/json').replace('.jpg', '.json')

        fname, iname = image_path.split('/')[-2:]
        score_path = io_utils.create_dir(args.temp + fname + '/') + iname.replace('.jpg', '.json')

        if os.path.isfile(score_path):
            continue
        
        image_paths.append(image_path)
        json_paths.append(json_path)
        score_paths.append(score_path)

    """
    [CC3M] I: 2825178, J: 2825178, P: 2825178
    """
    print()
    print(f'I: {len(image_paths)}, J: {len(json_paths)}, S: {len(score_paths)}')
    
    # inference
    params = []
    indices = np.arange(len(image_paths))
    length_per_gpu = len(indices) // len(args.gpus)
    
    for gpu_index in range(len(args.gpus)-1):
        param = [
            args, image_paths, json_paths, score_paths,
            indices[:length_per_gpu], torch.device('cuda', gpu_index), False
        ]
        params.append(param); indices = indices[length_per_gpu:]
    
    param = [
        args, image_paths, json_paths, score_paths, 
        indices, torch.device('cuda', len(args.gpus)-1), True
    ]
    params.append(param)
    
    io_utils.parallel(inference, params, len(params))
