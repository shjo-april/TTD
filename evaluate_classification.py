import numpy as np

from core import utils
from tools import io_utils, json_utils

args = io_utils.Parser({'root': 'H:/CC3M/', 'arch': 'NLTK', 'threshold': 0.5}).get()

extractor = utils.SimpleTagExtraction()
eval_dict = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}

for json_path in io_utils.progress(io_utils.listdir(args.root + 'json_refined/*/*.json')):
    json_path = json_path.replace('\\', '/')
    data = json_utils.read(json_path, encoding='utf-8')

    caption, human_tags, ignore_tags = data['caption'], data['human'], data['ignore']
    
    # ignore uncertain samples
    if len(human_tags) == 0: continue
    elif data['incorrect']: continue

    # get predicted tags
    if args.arch == 'NLTK': 
        pred_tags = data['NLTK']
    elif args.arch in ['Vicuna-7B', 'Vicuna-33B', 'Qwen-72B']:
        pred_tags = []
        for pred_tag in json_utils.read(json_path.replace('json_refined', f'json_{args.arch.lower()}'), 'utf-8')['tags']:
            if ' ' in pred_tag: pred_tags += pred_tag.split(' ')
            else: pred_tags += [pred_tag]

        candidates = extractor.extract_tags_from_caption(caption)
        pred_tags = [p for p in pred_tags if p in candidates]
    else:
        pred_data = json_utils.read(json_path.replace('json_refined', f'json_{args.arch}'), 'utf-8')

        tags = np.asarray(pred_data['tags'])
        scores = np.asarray(pred_data['scores'])

        if args.threshold > 0:
            pred_tags = list(tags[scores > args.threshold])
        else: 
            # adaptive threshold
            try:
                intervals = scores[:-1]-scores[1:]
                th = np.argmax(intervals)
                pred_tags = list(tags[:th+1])
            except ValueError:
                pred_tags = list(tags)
            
    # calculate FP & TP
    FP = 0
    TP = 0

    for pred_tag in pred_tags:
        if pred_tag in ignore_tags: continue

        if pred_tag in human_tags: TP += 1
        else: FP += 1

    # calculate FN
    FN = 0

    for gt_tag in human_tags:
        if gt_tag in ignore_tags: continue
        if not gt_tag in pred_tags: FN += 1

    # calculate TN
    TN = 0
    all_tags = extractor.extract_tags_from_caption(caption)

    for tag in all_tags:
        if tag in human_tags: continue
        elif tag in ignore_tags: continue

        if not tag in pred_tags: TN += 1

    eval_dict['TP'] += TP
    eval_dict['FP'] += FP
    eval_dict['FN'] += FN
    eval_dict['TN'] += TN

TP = eval_dict['TP']
FP = eval_dict['FP']
FN = eval_dict['FN']
TN = eval_dict['TN']

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = (2 * precision * recall) / (precision + recall)
acc = (TP + TN) / (TP + FP + FN + TN)

print(f'[{args.arch:>10}] Precision: {precision*100:.1f}, Recall: {recall*100:.1f}, F1: {f1*100:.1f}, Acc: {acc*100:.1f}')
