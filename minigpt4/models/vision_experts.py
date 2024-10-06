"""
    多个不同的vision expert进行合并
"""
import torch
from minigpt4.models.simplenet import SimpleNet, simplenet_transforms
from minigpt4.models.adrefexpert_v2 import adrefexpert
from minigpt4.models.adexpert import adexpert
from minigpt4.models.aprilgan import AprilGAN
from torch.nn import functional as F


def get_vision_expert(ve_name, expert_args):
    if ve_name == 'simplenet':
        return SimpleNet("./pretrained_models/simplenet_mvtec_pt")
    elif ve_name == 'simplenetV':
        return SimpleNet("./pretrained_models/simplenet_visa_pt")
    elif ve_name == 'adgpt':
        return adexpert()
    elif ve_name == 'patchcore':
        default_args = {
            'round_index': 0, 
            'k_shot': 0, 
        }
        if ve_name in expert_args:
            default_args.update(expert_args[ve_name])
        return adrefexpert(**default_args)
    elif ve_name == 'aprilgan':
        return AprilGAN(preload_aprilgan='mvtec')
    elif ve_name == 'aprilganV':
        return AprilGAN(preload_aprilgan='visa')
    else:
        raise NotImplementedError(f"Not Support Vision Expert: {ve_name}")


class VisionExpert(torch.nn.Module):

    def __init__(self, expert_types, expert_args=dict()):
        super().__init__()
        if isinstance(expert_types, str):
            self.expert_types = expert_types.split('+')
        else:
            self.expert_types = expert_types
        assert isinstance(self.expert_types, list), "Expert types should be List or Str."

        self.experts = torch.nn.ModuleDict(
            {
                ve_name: get_vision_expert(ve_name, expert_args)
                for ve_name in self.expert_types
            }
        )
    
    def forward(self, imgs, scenes, ve_name, querypath=None, testphase=False):
        if ve_name == 'simplenet' or ve_name == 'simplenetV':
            simple_img = simplenet_transforms(imgs.detach().clone())  # Transform不同，得变一下
            _, anomaly_maps = self.experts[ve_name](simple_img, scenes)
            anomaly_maps = anomaly_maps.unsqueeze(1)
        elif ve_name == 'patchcore':
            assert querypath is not None, 'Adrefgpt need at least one querypath. '
            anomaly_maps, _ = self.experts[ve_name](imgs, scenes, querypath=querypath, testphase=testphase)
        elif ve_name == 'adgpt':
            anomaly_maps, _ = self.experts[ve_name](imgs, scenes)
        elif ve_name == 'aprilgan' or ve_name == 'aprilganV':
            resized_img = F.interpolate(imgs.detach().clone(), (518, 518), mode='bicubic', align_corners=True)
            anomaly_maps = self.experts[ve_name](resized_img, scenes)
        else:
            raise NotImplementedError(f"Not Support Vision Expert: {ve_name}")

        return anomaly_maps

        

