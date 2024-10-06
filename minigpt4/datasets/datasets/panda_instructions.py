import copy
import os
import json
from tqdm import tqdm
import ipdb
import random
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence

import torch
import torch.distributed as dist
import transformers
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image


class PandaInstructionDataset(Dataset):
    """Dataset for PandaGPT Instruction tuning."""
    DatasetName='PandaInstructions'

    def __init__(self, meta_path: str, image_root_path: str):
        super(PandaInstructionDataset, self).__init__()
        self.meta_path = meta_path
        with open(self.meta_path, 'r') as f:
            json_data = json.load(f)

        self.norm_transform = transforms.Compose(
                            [
                                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=(0.48145466, 0.4578275, 0.40821073),
                                    std=(0.26862954, 0.26130258, 0.27577711),
                                ),
                            ]
                        )

        self.vis_root = image_root_path
        self.image_path_list, self.caption_list = [], []
        for item in json_data:
            one_image_name, one_caption = item["image_name"], item["conversation"]
            if len(one_caption) > 2:
                one_caption = one_caption[:2]  # 只取第一组问答
            # print(one_caption)
            # TODO: stage 2 dataset format is invalid
            if not one_image_name.endswith('.jpg'):
                one_image_name += '.jpg'
            one_image_path = self.vis_root + '/{}'.format(one_image_name)
            self.image_path_list.append(one_image_path)
            self.caption_list.append(one_caption)
        print(f'[!] collect {len(self.image_path_list)} samples for training')

    def __len__(self): # number of instances
        return len(self.image_path_list)

    def __repr__(self) -> str:
        example = self[0]
        ret = f"{self.DatasetName}: \n" \
        f"\tBuild Info: \n" f"\t\tImage Root: {self.vis_root} \n" f"\t\tMeta Path: {self.meta_path} \n"\
        f"\tExample: {list(example.keys())} \n" f"\t\tImage: {example['image'].shape} \n" f"\t\tQuestion: {example['question']} \n" f"\t\tAnswer: {example['text_input']} \n" 
        return ret

    #def __getitem__(self, i) -> Dict[str, torch.Tensor]: # how to get item, 取一个样本
    def __getitem__(self, i):
        texts = self.caption_list[i]
        question = texts[0]['value']
        answer = texts[1]['value']
        image_path = self.image_path_list[i]
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.norm_transform(image)
        return dict(image = image_tensor, text_input=answer, question="<Img><ImageHere></Img>"+question, scene='object')