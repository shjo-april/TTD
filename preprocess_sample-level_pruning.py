import torch

from tools import json_utils, io_utils

args = io_utils.Parser({'data': 'cc3m+12m', 'domain': 'train', 'root': './', 'arch': 'TCL'}).get()

caption_dict = {}

for data in args.data.split('+'):
    pt_dir = args.root + f'temp/{args.arch}/{data}/{args.domain}/'
    for json_path in io_utils.progress(io_utils.listdir(pt_dir + '*/*.json'), data):
        key = json_path.replace('\\', '/').replace(pt_dir, '')
        caption_dict[key] = float(json_utils.read(json_path)['score'])
    
caption_dict = {k: v for k, v in sorted(caption_dict.items(), key=lambda item: item[1], reverse=True)}
json_utils.write(args.root + f'temp/{args.arch}_{args.data}.json', caption_dict)