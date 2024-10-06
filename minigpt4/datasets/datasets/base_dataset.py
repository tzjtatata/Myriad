"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
from typing import Iterable

from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import default_collate
from PIL import Image
import numpy as np


def imreader(args):
    i, anns = args
    ann = anns[i]
    ann['image'] = Image.open(ann['path']).convert("RGB")
    np.array(ann['image'])  # 为了避免Image的延迟机制


def iter_obj(num, objs):
    for i in range(num):
        yield (i, objs)
    

class BaseDataset(Dataset):

    def __init__(
        self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[], is_preload=False, preload_fn=imreader
    ):
        """
        主要是调用load_annotations函数来读取annotation, 调用preload函数来预载数据
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.ann_paths = ann_paths
        self.load_annotations()  # 为了不影响后续的dataset, 这里就不传参数了

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.is_preload = is_preload
        self._cache = None
        self.preload_fn = preload_fn
        if self.is_preload: self.preload()

        # 把list转化成dict, 不明所以
        # self._add_instance_ids()

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return default_collate(samples)
    
    def load_annotations(self):
        self.annotation = []
        for ann_path in self.ann_paths:
            self.annotation.extend(json.load(open(ann_path, "r"))['annotations'])

    def prepare_img(self, index):
        raise NotImplementedError(f"This Class {self.__class__} do not implement Prepare img function.")
    
    def get_image_path(self, rel_path):
        return os.path.join(self.vis_root, '2cls_highshot', rel_path)
    
    def construct_preload_maps(self):
        return [{'path': self.get_image_path(ann['img_path']), 'rel_path': ann['img_path']} for ann in self.annotation]
    
    def post_preload(self, results):
        self._cache = {item['rel_path']: item['image'] for item in results}
    
    def preload(self):
        from multiprocessing.dummy import Pool
        from tqdm import tqdm
        # may use `from multiprocessing import Pool` instead, but less efficient and
        # NOTE: `multiprocessing.Pool` will duplicate given object for each process.
        print('Starting to load images via multiple imreaders')
        pool = Pool() # use all threads by default
        _cache = self.construct_preload_maps()
        for ann in tqdm(pool.imap(self.preload_fn, iter_obj(len(_cache), _cache)), total=len(self.annotation), desc="Preloading Image"):
            pass
        pool.close()
        pool.join()
        self.post_preload(_cache)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)


class ConcatDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)

    def collater(self, samples):
        # TODO For now only supports datasets with same underlying collater implementations

        all_keys = set()
        for s in samples:
            all_keys.update(s)

        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())

        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)
