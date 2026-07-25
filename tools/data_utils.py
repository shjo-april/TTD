import os
from .json_utils import read

class Dataset:
    def __init__(self, path):
        self.data_dict = read(path)
        self.tag = os.path.basename(path).replace('.json', '')

        self.ignore = self.data_dict['ignore']
        self.class_names = [n.lower() for n in self.data_dict['names']]
        self.num_classes = len(self.class_names)
        self.class_dict = {n: i for i, n in enumerate(self.class_names)}

    def __getitem__(self, key):
        if isinstance(key, int): return self.class_names[key]
        elif isinstance(key, str): return self.class_dict[key]
        else: raise ValueError(f'Please check a value (key: {key})')
