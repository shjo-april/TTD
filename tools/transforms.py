import cv2
import torch
import random
import numpy as np

from PIL import Image

from torchvision import transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, output_dict):
        isdict = isinstance(output_dict, dict)
        if not isdict: output_dict = {'image': output_dict}

        for transform in self.transforms: 
            output_dict = transform(output_dict)
        
        if not isdict: output_dict = output_dict['image']
        return output_dict
    
    def __repr__(self):
        text = 'Compose(\n'
        for transform in self.transforms:
            text += '\t{}\n'.format(transform)
        text += ')'
        return text

class FixedResize:
    def __init__(self, image_size=512):
        self.size = (image_size, image_size)
        self.key_dict = {'image': cv2.INTER_CUBIC, 'mask': cv2.INTER_NEAREST}
    
    def resize(self, image, size, mode):
        return cv2.resize(image, size, interpolation=mode)
    
    def __call__(self, output_dict):
        for key in output_dict:
            if 'image' in key: output_dict[key] = self.resize(output_dict[key], self.size, self.key_dict['image'])
            elif 'mask' in key: output_dict[key] = self.resize(output_dict[key], self.size, self.key_dict['mask'])
            elif 'heatmaps' in key: output_dict[key] = [self.resize(mask, self.size, self.key_dict['mask']) for mask in output_dict[key]]
        return output_dict
    
    def __repr__(self):
        return f'FixedResize({self.size})'

class RandomResize:
    def __init__(self, min_image_size, max_image_size):
        self.min_image_size = min_image_size
        self.max_image_size = max_image_size
        self.key_dict = {'image': cv2.INTER_CUBIC, 'mask': cv2.INTER_NEAREST}

    def resize(self, image, size, mode):
        h, w = image.shape[:2]
        if w < h: scale = size / h
        else: scale = size / w
        
        size = (int(round(w*scale)), int(round(h*scale)))
        if size[0] == w and size[1] == h: return image
        else: return cv2.resize(image, size, interpolation=mode)
    
    def __call__(self, output_dict):
        image_size = random.randint(self.min_image_size, self.max_image_size)
        for key in output_dict:
            if 'image' in key: output_dict[key] = self.resize(output_dict[key], image_size, self.key_dict['image'])
            elif 'mask' in key: output_dict[key] = self.resize(output_dict[key], image_size, self.key_dict['mask'])
            elif 'heatmaps' in key: output_dict[key] = [self.resize(mask, image_size, self.key_dict['mask']) for mask in output_dict[key]]
        return output_dict

    def __repr__(self):
        return f'RandomResize({self.min_image_size}, {self.max_image_size})'

class RandomHFlip:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, output_dict):
        if np.random.rand() <= self.p:
            for key in output_dict.keys():
                if 'heatmaps' in key: output_dict[key] = [mask[:, ::-1] for mask in output_dict[key]]
                else: output_dict[key] = output_dict[key][:, ::-1]
        return output_dict

    def __repr__(self):
        return f'RandomHFlip(p={self.p})'

class RandomCrop:
    def __init__(self, size, channels=3, bg_mask=-1):
        if isinstance(size, int):
            self.w_size = size
            self.h_size = size
        elif isinstance(size, tuple):
            self.w_size, self.h_size = size

        self.bg_mask = bg_mask
        self.shape = (channels, self.h_size, self.w_size)
    
    def get_random_boxes(self, image):
        h, w = image.shape[1:]
        
        crop_h = min(self.h_size, h)
        crop_w = min(self.w_size, w)

        w_space = w - self.w_size
        h_space = h - self.h_size

        img_top = 0
        img_left = 0
        cont_top = 0
        cont_left = 0

        if w_space > 0: img_left = random.randrange(w_space + 1)
        else: cont_left = random.randrange(-w_space + 1)

        if h_space > 0: img_top = random.randrange(h_space + 1)
        else: cont_top = random.randrange(-h_space + 1)

        src_bbox = [img_left, img_top, img_left+crop_w, img_top+crop_h]
        dst_bbox = [cont_left, cont_top, cont_left+crop_w, cont_top+crop_h]

        return src_bbox, dst_bbox
    
    def __call__(self, output_dict):
        for key in output_dict:
            if 'image' in key:
                (sxmin, symin, sxmax, symax), (dxmin, dymin, dxmax, dymax) = self.get_random_boxes(output_dict[key])
                break
        
        for key in output_dict:
            if 'image' in key:
                cropped_image = np.zeros(self.shape, output_dict[key].dtype)
                cropped_image[:, dymin:dymax, dxmin:dxmax] = output_dict[key][:, symin:symax, sxmin:sxmax]
                output_dict[key] = cropped_image
            elif 'mask' in key:
                cropped_mask = np.ones(self.shape[1:], output_dict[key].dtype) * self.bg_mask
                cropped_mask[dymin:dymax, dxmin:dxmax] = output_dict[key][symin:symax, sxmin:sxmax]
                output_dict[key] = cropped_mask
            elif 'heatmaps' in key:
                cropped_heatmaps = []
                for heatmap in output_dict[key]:
                    cropped_heatmap = np.ones(self.shape[1:], heatmap.dtype) * self.bg_mask
                    cropped_heatmap[dymin:dymax, dxmin:dxmax] = heatmap[symin:symax, sxmin:sxmax]
                    cropped_heatmaps.append(cropped_heatmap)
                output_dict[key] = cropped_heatmaps
        
        output_dict['box'] = [dxmin, dymin, dxmax, dymax]
        return output_dict

    def __repr__(self):
        size = (self.w_size, self.h_size)
        return f'RandomCrop(size={size})'

class Normalize:
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.mean = mean
        self.std = std

    def __call__(self, data: dict):
        image = np.asarray(data['image'], dtype=np.float32)
        h, w, c = image.shape
        
        norm_image = np.zeros((c, h, w), np.float32)
        norm_image[0, ...] = (image[..., 0] / 255. - self.mean[0]) / self.std[0]
        norm_image[1, ...] = (image[..., 1] / 255. - self.mean[1]) / self.std[1]
        norm_image[2, ...] = (image[..., 2] / 255. - self.mean[2]) / self.std[2]

        data['image'] = norm_image

        if 'mask' in data: data['mask'] = data['mask'].astype(np.float32) / 255.
        if 'heatmaps' in data: data['heatmaps'] = [mask.astype(np.float32) / 255. for mask in data['heatmaps']]

        return data
    
    def __repr__(self):
        return f'Normalize(mean={self.mean}, std={self.std})'

class Denormalize:
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.mean = mean
        self.std = std

    def __call__(self, image: np.ndarray):
        if not isinstance(image, np.ndarray): 
            image = image.cpu().detach().numpy()

        image = image.transpose((1, 2, 0))
        image = (image * self.std) + self.mean
        image = (image * 255).astype(np.uint8)
        return image
    
    def __repr__(self):
        return f'Denormalize(mean={self.mean}, std={self.std})'

class ToTensor:
    def __init__(self):
        pass

    def __call__(self, output_dict):
        for key in output_dict:
            if 'image' in key: output_dict[key] = torch.from_numpy(output_dict[key])
            elif 'mask' in key: output_dict[key] = torch.from_numpy(output_dict[key])
            elif 'heatmaps' in key: output_dict[key] = torch.from_numpy(np.asarray(output_dict[key], dtype=np.float32))
        return output_dict
    
    def __repr__(self):
        return f'ToTensor()'