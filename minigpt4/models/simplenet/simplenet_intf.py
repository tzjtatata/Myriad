# ------------------------------------------------------------------
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# ------------------------------------------------------------------
# Modified by Yiming Zhou
# ------------------------------------------------------------------

import os
import sys

import numpy as np
import torch
from torch import nn

from . import backbones, simplenet, utils
import yaml
import cv2
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SimpleNet_Interface(nn.Module):
    def __init__(self, ckpt_rt, **kwargs):
        super().__init__()
        if "yaml_path" in kwargs:
            with open(kwargs["yaml_path"],"r") as f:
                params=yaml.load(f,yaml.Loader)
            self.net_params=params
        else:
            self.net_params=kwargs
        print(f"init models with params:{self.net_params}")
        self.get_net=self.net(**self.net_params)
        self.ckpt_rt=ckpt_rt

        subdirs=os.listdir(self.ckpt_rt)
        ckpt_paths = {}
        for c in subdirs:
            class_name = c.replace('mvtec_', '')
            ckpt_paths[class_name] = os.path.join(self.ckpt_rt, c)
        self.nets = self.get_net((3, 288, 288), list(ckpt_paths.keys()))[0]
        for c_name, c_path in ckpt_paths.items():
            self.load_ckpt(self.nets.header[c_name], c_path)
        
        assert os.path.isdir(self.ckpt_rt), "checkpoint root not found"
        self.transform_img = [
            transforms.Resize(329),
            transforms.CenterCrop(288),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

    def net(self,
            backbone_names=None,
            layers_to_extract_from=None,
            pretrain_embed_dimension=None,
            target_embed_dimension=None,
            patchsize=None,
            embedding_size=None,
            meta_epochs=None,
            aed_meta_epochs=None,
            gan_epochs=None,
            noise_std=None,
            dsc_layers=None, 
            dsc_hidden=None,
            dsc_margin=None,
            dsc_lr=None,
            auto_noise=None,
            train_backbone=None,
            cos_lr=None,
            pre_proj=None,
            proj_layer_type=None,
            mix_noise=None):
        backbone_names = list(backbone_names)
        if len(backbone_names) > 1:
            layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
            for layer in layers_to_extract_from:
                idx = int(layer.split(".")[0])
                layer = ".".join(layer.split(".")[1:])
                layers_to_extract_from_coll[idx].append(layer)
        else:
            layers_to_extract_from_coll = [layers_to_extract_from]

        def get_simplenet(input_shape, cls_names):
            simplenets = []
            for backbone_name, layers_to_extract_from in zip(
                    backbone_names, layers_to_extract_from_coll
            ):
                backbone_seed = None
                if ".seed-" in backbone_name:
                    backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(backbone_name.split("-")[-1])
                backbone = backbones.load(backbone_name)
                backbone.name, backbone.seed = backbone_name, backbone_seed

                simplenet_inst = simplenet.SimpleNet()
                simplenet_inst.load(
                    backbone=backbone,
                    cls_names=cls_names, 
                    layers_to_extract_from=layers_to_extract_from,
                    input_shape=input_shape,
                    pretrain_embed_dimension=pretrain_embed_dimension,
                    target_embed_dimension=target_embed_dimension,
                    patchsize=patchsize,
                    embedding_size=embedding_size,
                    meta_epochs=meta_epochs,
                    aed_meta_epochs=aed_meta_epochs,
                    gan_epochs=gan_epochs,
                    noise_std=noise_std,
                    dsc_layers=dsc_layers,
                    dsc_hidden=dsc_hidden,
                    dsc_margin=dsc_margin,
                    dsc_lr=dsc_lr,
                    auto_noise=auto_noise,
                    train_backbone=train_backbone,
                    cos_lr=cos_lr,
                    pre_proj=pre_proj,
                    proj_layer_type=proj_layer_type,
                    mix_noise=mix_noise,
                )
                simplenets.append(simplenet_inst)
            return simplenets

        return get_simplenet
    
            
    def load_ckpt(self,net,ckpt_dir):
        ckpt_path = os.path.join(ckpt_dir, "ckpt.pth")
        if os.path.exists(ckpt_path):
            print("read ckpt file from {}".format(ckpt_path))
            state_dicts = torch.load(ckpt_path, map_location='cpu')
            print(state_dicts.keys())
            if 'discriminator' in state_dicts:
                net.discriminator.load_state_dict(state_dicts['discriminator'])
                if "pre_projection" in state_dicts:
                    net.pre_projection.load_state_dict(state_dicts["pre_projection"])
            else:
                net.load_state_dict(state_dicts, strict=False)

    def data_transform(self,image):
        img=self.transform_img(image)
        if img.shape[0]!=1 and img.shape[0]!=3:
            img=img.permute(1,2,0)
        img=img.unsqueeze(0)
        # print(img.shape)
        return img

    def predict(self,image,classname,use_torch_transform=False):
        '''
        usage:
        predict(image,classname)
            image      input image should be a numpy array or torch tensor with shape (n,c,h,w)
            classname  image class name, a string, model will load weights according this class name
        
        return scores,masks
            scores     image scores predicted by model, a list has (n) scores
            masks      anomal maps generated by model, a list has (n) numpy arrays, each one has a shape of (h,w)
        '''
        if not isinstance(image,torch.Tensor):
            if use_torch_transform:
                image=self.data_transform(image)
            else:
                image=torch.tensor(image,dtype=float)
        # print(image.shape)
        n,c,h,w=image.shape
        image_shape=(c,h,w)
        assert h==w, "image should have same height and width"
                
        net = self.nets[classname]

        scores, masks = net.predict(image)
        return torch.tensor(scores[0]+1), torch.tensor(masks[0]+1)

    def forward(self, image, classnames):
        '''
        usage:
        predict(image,classname)
            image      input image should be a numpy array or torch tensor with shape (n,c,h,w)
            classname  image class names, a list of string, model will load weights according these class names
        
        return scores,masks
            scores     image scores predicted by model, a list has (n) scores
            masks      anomal maps generated by model, a list has (n) numpy arrays, each one has a shape of (h,w)
        '''
        scores, masks = self.nets.predict(image, classnames)
        scores, masks = torch.tensor(scores, device=image.device, dtype=image.dtype), torch.tensor(masks, device=image.device, dtype=image.dtype)
        return scores + 1, masks + 1  # 这里是后处理，这个模型的输出都要+1

