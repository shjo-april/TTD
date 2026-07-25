import os
import ray
import numpy as np

from tqdm import tqdm
from PIL import Image

from tools import evaluators, io_utils, json_utils

@ray.remote
def update_mIoU(obj: evaluators.SemanticSegmentation, pred_mask, gt_mask, image_id):
    meter_dict = obj.set()

    obj_mask = gt_mask != obj.ignore_index
    correct_mask = (pred_mask == gt_mask) * obj_mask
    
    tags = []
    IoUs = []

    for i in range(obj.num_classes):
        meter_dict['P'][i] += np.sum((pred_mask==i)*obj_mask)
        meter_dict['T'][i] += np.sum((gt_mask==i)*obj_mask)
        meter_dict['TP'][i] += np.sum((gt_mask==i)*correct_mask)
        
        union = meter_dict['T'][i] + meter_dict['P'][i] - meter_dict['TP'][i]
        if union == 0:
            continue
        
        tags.append(obj.class_names[i])
        IoUs.append(float(meter_dict['TP'][i] / union))

    meter_dict['image_id'] = image_id
    meter_dict['mIoU'] = float(np.mean(IoUs))
    meter_dict['tags'] = tags
    meter_dict['IoUs'] = IoUs

    return meter_dict

def main(args):
    evaluator = evaluators.SemanticSegmentation(['background', 'foreground'])

    params = []
    sample_dict = {}
    
    ray.init(num_cpus=args.cpus, configure_logging=False)

    for gt_path in tqdm(io_utils.listdir(args.root + '*/*.png')):
        image_name = os.path.basename(gt_path)
        image_id = image_name.replace('.png', '')
        
        gt_mask = np.asarray(Image.open(gt_path))
        pred_mask = np.asarray(Image.open(args.pred + image_name))

        if gt_mask.shape != pred_mask.shape:
            raise Exception(f'Size Error: {image_name} {gt_mask.shape} {pred_mask.shape}')
        
        params.append(update_mIoU.remote(evaluator, pred_mask, gt_mask, image_id))
    
    if len(params) > 0:
        for data in ray.get(params):
            evaluator.add(data)
            sample_dict[data['image_id']] = {'mIoU': data['mIoU'], 'tags': data['tags'], 'IoUs': data['IoUs']}
    
    class_dict = evaluator.print(args.tag)
    
    json_utils.write(
        io_utils.create_dir('./results_CaptionIoU/') + f'{args.tag}.json', 
        {
            'performance': class_dict,
            'samples': sample_dict
        }
    )

if __name__ == '__main__':
    args = io_utils.Parser().add_from_inputs(
        {
            'cpus': os.cpu_count(),
            'root': '../cc3m_for_evaluation/mask_using_HQ-SAM/', 'tag': 'TCL',
            'pred': '//192.168.100.82/disks/nvme0-8tb/240124_ECCV2024_Unsupervised Tag Selection/results_caption/TCL@CC3M@TCL@[1.0]@448@s=2@pamr@bg=0.40/', 
        }
    )
    main(args)
