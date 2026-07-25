import torch

from torch.nn import functional as F

def resize(tensors, size=None, scale=1.0, mode='bilinear', align_corners=False):
    without_batch = len(tensors.size()) == 3

    if without_batch: tensors = tensors.unsqueeze(0)
    if size is None: size = tensors.size()[2:]
    
    size = list(size)
    size[0] = int(size[0] * scale)
    size[1] = int(size[1] * scale)

    if mode == 'nearest': align_corners = None
    
    _, _, h, w = tensors.size()
    if size[0] != h or size[1] != w:
        tensors = F.interpolate(tensors, size, mode=mode, align_corners=align_corners)
    
    if without_batch: tensors = tensors[0]
    return tensors

def resize_for_aspect_ratio(image: torch.Tensor, image_size: int=0):
    oh, ow = image.shape[1:]

    if image_size > 0: # resize
        short, long = (ow, oh) if ow <= oh else (oh, ow)
        new_short, new_long = image_size, int(image_size * long / short)
        nw, nh = (new_short, new_long) if ow <= oh else (new_long, new_short)
        image = resize(image, (nh, nw))
    else: # use original image
        image_size = min(oh, ow)
    
    return image, image_size

def get_scores(heatmaps):
    c, h, w = heatmaps.shape
    return heatmaps.view(c, h*w).max(dim=1)[0]

def get_templates_for_TCL():
    return [
        "itap of a {}.", "a photo of many {}.", "a photo of {}s.",
        "a bad photo of a {}.", "a origami {}.", "a photo of the large {}.", 
        "a {} in a video game.", "art of the {}.", "a photo of the small {}."
    ]

def get_templates_for_OpenAI():
    return [
        'a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.', 'a photo of the hard to see {}.',
        'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.',
        'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.', 'a photo of a hard to see {}.',
        'a bright photo of a {}.', 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.',
        'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.',
        'a black and white photo of the {}.', 'a painting of the {}.', 'a painting of a {}.', 'a pixelated photo of the {}.',
        'a sculpture of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a plastic {}.', 'a photo of the dirty {}.',
        'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.',
        'a {} in a video game.', 'a photo of one {}.', 'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'the origami {}.',
        'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.', 'a origami {}.', 'a low resolution photo of a {}.', 'the toy {}.',
        'a rendition of the {}.', 'a photo of the clean {}.', 'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.',
        'a photo of a weird {}.', 'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.',
        'a pixelated photo of a {}.', 'itap of the {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a plushie {}.',
        'a photo of the nice {}.', 'a photo of the small {}.', 'a photo of the weird {}.', 'the cartoon {}.', 'art of the {}.', 'a drawing of the {}.', 
        'a photo of the large {}.', 'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.', 'itap of a {}.', 'graffiti of the {}.',
        'a toy {}.', 'itap of my {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'a tattoo of the {}.',
    ]

def get_templates_for_MaskCLIP():
    return get_templates_for_OpenAI() + ['there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']

def get_templates(template):
    if template == 'MaskCLIP': return get_templates_for_MaskCLIP()
    elif template == 'OpenAI': return get_templates_for_OpenAI()
    elif template == 'TCL': return get_templates_for_TCL()

def add_background(mask: torch.Tensor, bg_th: float=0, bg_gamma: int=0):
    if bg_gamma > 0: return torch.cat([torch.pow(1. - torch.max(mask, dim=0, keepdim=True)[0], bg_gamma), mask], dim=0)
    else: return torch.cat([mask.new_ones([1] + list(mask.shape[1:])) * bg_th, mask], dim=0)

class SimpleTagExtraction:
    def __init__(self):
        self.stopwords = [
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 
            'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
            'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
            'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 
            'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 
            'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 
            'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't",
            '\'t', '\'s', '\'d', '\'ve', '\'ll'
        ]
        self.punctuation = [
            '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '\n', '\t'
        ]

    def extract_tags_from_caption(self, caption: str, remove_stopwords: bool=False):
        try:
            while caption[0] in self.punctuation: caption = caption[1:]
            while caption[-1] in self.punctuation: caption = caption[:-1]
        except IndexError:
            return []
        
        tags = {}
        caption = caption.lower()

        for each_tag in caption.split(' '): # e.g., 'person is running' to ['person', 'is', 'running']
            each_tags = []

            def recursive_split(_tag: str):
                if len(_tag) == 0: return
                
                include_punctuation = False
                for punc in self.punctuation:
                    if punc in _tag:
                        for __tag in _tag.split(punc):
                            include_punctuation = True
                            recursive_split(__tag)
                        break
                
                if not include_punctuation:
                    each_tags.append(_tag)
            
            recursive_split(each_tag)

            for tag in each_tags:
                try:
                    while tag[0] in self.punctuation: tag = tag[1:]
                    while tag[-1] in self.punctuation: tag = tag[:-1]
                except IndexError:
                    continue

                if len(tag) == 0:
                    continue
                
                if remove_stopwords:
                    if tag in self.stopwords: # e.g., "and"
                        continue
                
                try: tags[tag] += 1
                except KeyError: tags[tag] = 0

        return list(tags.keys())