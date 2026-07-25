import os
import tqdm
import copy
import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from core import networks, lora_utils, losses, utils
from tools import json_utils, cv_utils, io_utils, torch_utils
from tools import transforms as T

def collate(batch):
    images = []
    captions = []
    tags = []
    boxes = []

    for image, caption, tag, box in batch:
        images.append(image)
        captions.append(caption)
        tags.append(tag)
        boxes.append(box)
        
    data = {
        'images': torch.stack(images),
        'captions': captions, 
        'tags': tags,
        'boxes': boxes,
    }
    return data

class Dataset:
    def __init__(self, root, data, domain, arch, transform):
        self.image_dir = root
        self.pt_dir = f'./temp/{arch}/'
        
        self.domain = domain
        self.transform = transform

        caption_dict = {}
        for data in data.split('+'):
            for key, value in json_utils.read(f'./temp/{arch}_{data}.json').items():
                caption_dict[f'{data}/{domain}/'+key] = value

        self.pt_names = np.asarray(list(caption_dict.keys()))
        self.caption_scores = np.asarray(list(caption_dict.values()))

        indices = np.argsort(self.caption_scores)[::-1]
        self.pt_names, self.caption_scores = self.pt_names[indices], self.caption_scores[indices]

    def filter(self):
        mean, std = self.caption_scores.mean(), self.caption_scores.std()
        self.pt_names = self.pt_names[self.caption_scores > (mean + std)]
    
    def __len__(self):
        return len(self.pt_names)

    def __getitem__(self, i):
        pt_name = self.pt_names[i]
        image_name = pt_name.replace(f'/{self.domain}/', f'/{self.domain}/image/').replace('.pt', '.jpg')

        image = cv_utils.imread(self.image_dir + image_name)[..., ::-1]
        data = torch.load(self.pt_dir + pt_name)

        output_dict = self.transform({'image': image})
        return output_dict['image'], data['caption'], data['tags'], output_dict['box']

if __name__ == '__main__':
    args = io_utils.Parser(
        {
            'gpus': '0', 'cpus': 16, 'seed': 0, 
            'root': '../', 'data': 'cc3m', 'arch': 'TCL', 'tag': 'TCL@LoRA',
            'lr': 1e-4, 'wd': 5e-5, 'batch': 32, 'epochs': 1, 'r': 4, 'loss': 'L1', 'scales': [1.],
            'lambda_cap': 1., 'lambda_tag': 1.,
        }
    ).get()

    model_dir = io_utils.create_dir(f'./experiments/models/{args.tag}/')
    writer = SummaryWriter(f'./experiments/tensorboards/{args.tag}/')
    print = lambda x: io_utils.log(x, model_dir+'log.txt')
    
    torch_utils.set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    # create model
    device = torch.device('cuda:0')

    model = networks.VLModel(args.arch, 'ViT-B@16', device, pamr=True)
    print('{} w/o LoRA: {:.1f}M'.format(args.arch, torch_utils.calculate_parameters(model.parameters())))
    
    # apply LoRA weights
    lora_params = lora_utils.apply(model, args.arch, args.r)
    print('{} w/  LoRA: {:.1f}M'.format(args.arch, torch_utils.calculate_parameters(model.parameters())))
    
    # create optimizer and scheduler
    optimizer = optim.AdamW(lora_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.wd)
    
    # create transform, dataset, and data loader
    if args.arch in ['TCL', 'MaskCLIP']:
        args.image = 448
        train_transform = T.Compose([T.RandomResize(320, 640), T.Normalize(), T.RandomCrop(args.image), T.ToTensor()])
    else:
        raise NotImplementedError(f'ERROR: Data Augmentation')

    print(f'Data Augmentation: {train_transform}')
    train_dataset = Dataset(args.root, args.data, 'train', args.arch, train_transform)
    
    print('The number of training images w/o filtering: {:,}'.format(len(train_dataset)))
    train_dataset.filter(args.filter)
    print('The number of training images w/  filtering: {:,}'.format(len(train_dataset)))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=args.cpus, collate_fn=collate)

    # define two losses
    caption_loss_fn = losses.ReconstructionLoss(args.loss)
    tag_loss_fn = losses.BalancedReconstructionLoss(args.loss)
    
    # training loop
    iteration = 0
    for epoch in range(1, args.epochs+1):
        train_losses = []
        train_caption_losses = []
        train_tag_losses = []
        
        for data in tqdm.tqdm(train_loader, f'Epoch: {epoch:03d}'):
            images, captions, all_tags, boxes = data['images'].to(device), data['captions'], data['tags'], data['boxes']
            
            caption_loss = 0.
            tag_loss = 0.
            loss_cnt = 0.

            for i in range(images.shape[0]):
                _, h, w = images[i].shape

                tags = all_tags[i]
                xmin, ymin, xmax, ymax = boxes[i]

                # Tag Pruning: 1) extract tag scores
                text_embs = model.produce_text_embeddings(tags)
                heatmaps = model.produce_heatmaps(images[i, :, ymin:ymax, xmin:xmax], text_embs)

                tags = np.asarray(tags)
                scores = utils.get_scores(heatmaps).cpu().detach().numpy()

                # Tag Pruning: 2) sort all tags by scores
                sorted_indices = np.argsort(scores)[::-1]
                tags = tags[sorted_indices]
                scores = scores[sorted_indices]

                # Tag Pruning: 3) calculate adaptive threshold for the largest gap between previous and next tags
                intervals = scores[:-1]-scores[1:]

                th = np.argmax(intervals)
                certain_tags = list(tags[:th+1])
                negative_tags = list(tags[th+1:])
                
                # Pseudo Masks: generate heatmaps for caption and selected tags
                caption_emb = model.produce_text_embeddings(captions[i])
                tag_embs = model.produce_text_embeddings(certain_tags)
                text_embs = torch.cat([caption_emb, tag_embs], dim=0)

                cropped_heatmaps = model.produce_heatmaps(images[i, :, ymin:ymax, xmin:xmax], text_embs, args.scales, False, args.image, mode='sliding')
                cropped_heatmaps = torch_utils.normalize_masks(cropped_heatmaps)
                
                pseudo_heatmaps = cropped_heatmaps.new_ones((1+len(certain_tags)+len(negative_tags), h, w)) * -1
                pseudo_heatmaps[0, ymin:ymax, xmin:xmax] = torch.max(cropped_heatmaps, dim=0)[0]                       # union mask of caption and certain/uncertain tags
                pseudo_heatmaps[1:len(certain_tags)+1, ymin:ymax, xmin:xmax] = cropped_heatmaps[1:len(certain_tags)+1] # masks of certain tags
                pseudo_heatmaps[len(certain_tags)+1:, ymin:ymax, xmin:xmax] = 0.                                       # zero forcing for negative tags
                pseudo_heatmaps[:, ymin:ymax, xmin:xmax] = torch_utils.normalize_masks(pseudo_heatmaps[:, ymin:ymax, xmin:xmax])
                
                pred_heatmaps = model.forward(images[i], captions[i], certain_tags+negative_tags)
                pseudo_heatmaps = torch_utils.resize(pseudo_heatmaps, pred_heatmaps.shape[1:], mode='nearest').detach()
                
                caption_loss += caption_loss_fn(pred_heatmaps[0], pseudo_heatmaps[0])
                tag_loss += tag_loss_fn(
                    pred_heatmaps[1:len(certain_tags)+1], pseudo_heatmaps[1:len(certain_tags)+1], 
                    pred_heatmaps[len(certain_tags)+1: ], pseudo_heatmaps[len(certain_tags)+1: ]
                )
                loss_cnt += 1.
            
            caption_loss /= loss_cnt
            tag_loss /= loss_cnt
            loss = args.lambda_cap * caption_loss + args.lambda_tag * tag_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_caption_losses.append(caption_loss.item())
            train_tag_losses.append(tag_loss.item())

            iteration += 1

            if iteration % 10 == 0:
                lr = torch_utils.get_learning_rate(optimizer)
                train_loss = np.mean(train_losses)
                train_caption_loss = np.mean(train_caption_losses)
                train_tag_loss = np.mean(train_tag_losses)

                writer.add_scalar('L_total', train_loss, iteration)
                writer.add_scalar('L_cap', train_caption_loss, iteration)
                writer.add_scalar('L_tag', train_tag_loss, iteration)
                
                print(f'\nIteration: {iteration:06d}, LR: {lr:.06f}, L: {train_loss:.03f}, L_cap: {train_caption_loss:.03f}, L_tag: {train_tag_loss:.03f}')

                train_losses = []
                train_caption_losses = []
                train_tag_losses = []
            
            if iteration % 100 == 0:
                lora_utils.save(model, model_dir + f'{iteration:06d}.pt')
