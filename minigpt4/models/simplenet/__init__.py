import os
from .simplenet_intf import SimpleNet_Interface, IMAGENET_MEAN, IMAGENET_STD

class SimpleNet(SimpleNet_Interface):
    def __init__(self, ckpt_rt, yaml_path=os.path.dirname(__file__)+"/params.yaml"):
        super().__init__(ckpt_rt, yaml_path=yaml_path)


import torch
from torchvision import transforms
mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), dtype=torch.float32)
std = torch.tensor((0.26862954, 0.26130258, 0.27577711), dtype=torch.float32)
DeNormalizer = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
Normalizer = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

def simplenet_transforms(imgs):
    x = Normalizer(DeNormalizer(imgs))
    return x