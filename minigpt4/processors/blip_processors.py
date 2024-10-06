"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import re
import copy
import numpy as np
import torch

from minigpt4.common.registry import registry
from minigpt4.processors.base_processor import BaseProcessor
from minigpt4.processors.randaugment import RandomAugment
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


class BlipImageBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), dtype=torch.float32)
        if std is None:
            std = torch.tensor((0.26862954, 0.26130258, 0.27577711), dtype=torch.float32)

        self.normalize = transforms.Normalize(mean.tolist(), std.tolist())
        self.denormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())


@registry.register_processor("blip_caption")
class BlipCaptionProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=50):
        self.prompt = prompt
        self.max_words = max_words

    def __call__(self, caption):
        caption = self.prompt + self.pre_caption(caption)

        return caption

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 50)

        return cls(prompt=prompt, max_words=max_words)

    def pre_caption(self, caption):
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption


@registry.register_processor("blip2_image_train")
class Blip2ImageTrainProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=224, mean=None, std=None, min_scale=0.5, max_scale=1.0):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                # transforms.RandomResizedCrop(
                #     image_size,
                #     scale=(min_scale, max_scale),
                #     interpolation=InterpolationMode.BICUBIC,
                # ),
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 224)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )


@registry.register_processor("loc_image_train")
class LocImageTrainProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=224, mean=None, std=None, min_scale=0.5, max_scale=1.0, strong_aug=False, identity=False, debug_mode=False):
        super().__init__(mean=mean, std=std)
        from mmdet.datasets.transforms import ResizeShortestEdge, RandomCrop, Resize
        self.debug_mode = debug_mode
        self.strong_aug = strong_aug

        if strong_aug:
            print("Built a LocImageTrainProcessor with Strong Augmentation.")
            self.ts = [
                RandomCrop((0.5, 0.5), crop_type='relative_range'), Resize((224, 224))
            ]
        else:
            if identity:
                # self.ts = [Resize((224, 224))]
                self.ts = []
            else:
                self.ts = [ResizeShortestEdge(224), RandomCrop((224, 224))]

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                self.normalize,
            ]
        )
    
    def apply_transforms(self, data_sample):
        _tmp = copy.deepcopy(data_sample)
        for t in self.ts:
            _tmp = t(_tmp)
            if _tmp is None:
                return _tmp
        return _tmp

    def __call__(self, data_sample):
        ret = self.apply_transforms(data_sample)
        while ret is None:
            ret = self.apply_transforms(data_sample)
        
        if not self.debug_mode: ret['img'] = self.transform(ret['img'])

        if 'gt_bboxes' in ret:
            ret['gt_bboxes'] = ret['gt_bboxes'].tolist()
        # print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')

        return ret

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 224)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)
        strong_aug = cfg.get("strong_aug", False)
        identity = cfg.get("identity", False)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
            strong_aug=strong_aug,
            identity=identity
        )


@registry.register_processor("blip2_image_eval")
class Blip2ImageEvalProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=224, mean=None, std=None):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 224)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        return cls(image_size=image_size, mean=mean, std=std)