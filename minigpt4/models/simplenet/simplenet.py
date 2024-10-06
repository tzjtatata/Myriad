# ------------------------------------------------------------------
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# ------------------------------------------------------------------
# Modified by Yiming Zhou
# ------------------------------------------------------------------

"""detection methods."""
import logging
import os
import pickle
from collections import OrderedDict

import math
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter

from . import common, metrics

import sys
import time
import json
import pdb

LOGGER = logging.getLogger(__name__)

use_sigmoid=True
use_tanh=False
no_norm=True
save_img=True
score_path="./mvtec_sig_score.jsonl"

def init_weight(m):

    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)


class Discriminator(torch.nn.Module):
    def __init__(self, in_planes, n_layers=1, hidden=None):
        super(Discriminator, self).__init__()

        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers-1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module('block%d'%(i+1),
                                 torch.nn.Sequential(
                                     torch.nn.Linear(_in, _hidden),
                                     torch.nn.BatchNorm1d(_hidden),
                                     torch.nn.LeakyReLU(0.2)
                                 ))
        self.tail = torch.nn.Linear(_hidden, 1, bias=False)
        self.apply(init_weight)

    def forward(self,x):
        # print("Before DSC body: ", x.max().item(), x.min().item())
        # self.body.eval()
        x = self.body(x)
        # print("After DSC body: ", x.max().item(), x.min().item())
        x = self.tail(x)
        # print("After DSC tail: ", x.max().item(), x.min().item())
        if use_sigmoid:
            x=F.sigmoid(x)
        if use_tanh:
            x=F.tanh(x)
        # print("After DSC activation: ", x.max().item(), x.min().item())
        return x


class Projection(torch.nn.Module):
    
    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super(Projection, self).__init__()
        
        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes 
            self.layers.add_module(f"{i}fc", 
                                   torch.nn.Linear(_in, _out))
            if i < n_layers - 1:
                # if layer_type > 0:
                #     self.layers.add_module(f"{i}bn", 
                #                            torch.nn.BatchNorm1d(_out))
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu",
                                           torch.nn.LeakyReLU(.2))
        self.apply(init_weight)
    
    def forward(self, x):
        
        # x = .1 * self.layers(x) + x
        x = self.layers(x)
        return x


class TBWrapper:
    
    def __init__(self, log_dir):
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir=log_dir)
    
    def step(self):
        self.g_iter += 1


class SimpleHead(torch.nn.Module):

    def __init__(
            self, 
            target_embed_dimension, 
            pre_proj, 
            proj_layer_type, 
            dsc_layers, 
            dsc_hidden, 
        ) -> None:
        super().__init__()
        self.target_embed_dimension = target_embed_dimension
        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.pre_projection = Projection(self.target_embed_dimension, self.target_embed_dimension, pre_proj, proj_layer_type)
        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden)
    
    def forward(self, x):
        if self.pre_proj > 0:
            # print("Before proj: ", x.max().item(), x.min().item())
            features = self.pre_projection(x)
            # print("After proj: ", features.max().item(), features.min().item())

        return self.discriminator(features)



class SimpleNet(torch.nn.Module):
    def __init__(self):
        """anomaly detection class."""
        super(SimpleNet, self).__init__()

    def load(
        self,
        backbone,
        layers_to_extract_from,
        input_shape,
        cls_names, 
        pretrain_embed_dimension, # 1536
        target_embed_dimension, # 1536
        patchsize=3, # 3
        patchstride=1, 
        embedding_size=None, # 256
        meta_epochs=1, # 40
        aed_meta_epochs=1,
        gan_epochs=1, # 4
        noise_std=0.05,
        mix_noise=1,
        noise_type="GAU",
        dsc_layers=2, # 2
        dsc_hidden=None, # 1024
        dsc_margin=.8, # .5
        dsc_lr=0.0002,
        train_backbone=False,
        auto_noise=0,
        cos_lr=False,
        lr=1e-3,
        pre_proj=0, # 1
        proj_layer_type=0,
        **kwargs,
    ):
        self.backbone = backbone
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, train_backbone
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_segmentor = common.RescaleSegmentor(
            target_size=(224, 224)  # 硬编码成(224, 224)尺寸
        )

        self.embedding_size = embedding_size if embedding_size is not None else self.target_embed_dimension
        self.meta_epochs = meta_epochs
        self.lr = lr
        self.cos_lr = cos_lr
        self.train_backbone = train_backbone
        if self.train_backbone:
            self.backbone_opt = torch.optim.AdamW(self.forward_modules["feature_aggregator"].backbone.parameters(), lr)
        # AED
        self.aed_meta_epochs = aed_meta_epochs

        self.pre_proj = pre_proj
        self.header = torch.nn.ModuleDict(
            {
                c_name: SimpleHead(self.target_embed_dimension, pre_proj, proj_layer_type, dsc_layers, dsc_hidden)
                for c_name in cls_names
            }
        )

        # Discriminator
        self.auto_noise = [auto_noise, None]
        self.dsc_lr = dsc_lr
        self.gan_epochs = gan_epochs
        self.mix_noise = mix_noise
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.dsc_margin= dsc_margin 

        self.model_dir = ""
        self.dataset_name = ""
        self.tau = 1
        self.logger = None

    def set_model_dir(self, model_dir, dataset_name):

        self.model_dir = model_dir 
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.logger = TBWrapper(self.tb_dir) #SummaryWriter(log_dir=tb_dir)

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                    input_image = image.to(torch.float)
                with torch.no_grad():
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False, evaluation=False):
        """Returns feature embeddings for images."""

        B = len(images)
        if not evaluation and self.train_backbone:
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
        else:
            _ = self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        
        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features) # pooling each feature to same channel and stack together
        features = self.forward_modules["preadapt_aggregator"](features) # further pooling        


        return features, patch_shapes

    
    def test(self, training_data, test_data):

        ckpt_path = os.path.join(self.ckpt_dir, "ckpt.pth")
        if os.path.exists(ckpt_path):
            print("read ckpt file from {}".format(ckpt_path))
            state_dicts = torch.load(ckpt_path, map_location='cpu')
            print(state_dicts.keys())
            if 'discriminator' in state_dicts:
                self.discriminator.load_state_dict(state_dicts['discriminator'])
                if "pre_projection" in state_dicts:
                    self.pre_projection.load_state_dict(state_dicts["pre_projection"])
            else:
                self.load_state_dict(state_dicts, strict=False)
            '''
            if "pretrained_enc" in state_dicts:
                self.feature_enc.load_state_dict(state_dicts["pretrained_enc"])
            if "pretrained_dec" in state_dicts:
                self.feature_dec.load_state_dict(state_dicts["pretrained_dec"])
            '''
        else:
            print("ckpt path {} not exists".format(ckpt_path))
            return

        aggregator = {"scores": [], "segmentations": [], "features": []}
        scores, segmentations, features, labels_gt, masks_gt = self.predict(test_data)

        aggregator["scores"].append(scores)
        aggregator["segmentations"].append(segmentations)
        aggregator["features"].append(features)

        scores = np.array(aggregator["scores"])
        min_scores = scores.min(axis=-1).reshape(-1, 1)
        max_scores = scores.max(axis=-1).reshape(-1, 1)
        scores = (scores - min_scores) / (max_scores - min_scores)
        scores = np.mean(scores, axis=0)

        segmentations = np.array(aggregator["segmentations"])
        min_scores = (
            segmentations.reshape(len(segmentations), -1)
            .min(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        max_scores = (
            segmentations.reshape(len(segmentations), -1)
            .max(axis=-1)
            .reshape(-1, 1, 1, 1)
        )
        segmentations = (segmentations - min_scores) / (max_scores - min_scores)
        segmentations = np.mean(segmentations, axis=0)

        anomaly_labels = [
            x[1] != "good" for x in test_data.dataset.data_to_iterate
        ]
            
        image_scores = metrics.compute_imagewise_retrieval_metrics(
            scores, anomaly_labels
        )
        auroc=image_scores["auroc"]
        # Pick out fpr and tpr
        tpr=image_scores["tpr"]
        fpr=image_scores["fpr"]
        ths=image_scores["threshold"]
        a=tpr-fpr
        idx=np.argmax(a)
        print("1-tpr,fpr,ths:",1-tpr[idx],fpr[idx],ths[idx])
        print("tpr",tpr)
        print("1-tpr",1-tpr)
        print("fpr",fpr)
        print("ths",ths)

        # Compute PRO score & PW Auroc for all images
        #pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
        #    segmentations, masks_gt
        #)
        #full_pixel_auroc = pixel_scores["auroc"]
        full_pixel_auroc = 0
        return auroc, full_pixel_auroc
    
    def _evaluate(self, test_data, scores, segmentations, features, labels_gt, masks_gt):
        scores = np.squeeze(np.array(scores))
        img_min_scores = scores.min(axis=-1)
        img_max_scores = scores.max(axis=-1)

        if not no_norm:
            scores = (scores - img_min_scores) / (img_max_scores - img_min_scores)
        # scores = np.mean(scores, axis=0)
        auroc = metrics.compute_imagewise_retrieval_metrics(
            scores, labels_gt 
        )["auroc"]

        if len(masks_gt) > 0:
            segmentations = np.array(segmentations)
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )

            norm_segmentations = np.zeros_like(segmentations)
            for min_score, max_score in zip(min_scores, max_scores):
                norm_segmentations += (segmentations - min_score) / max(max_score - min_score, 1e-2)
            norm_segmentations = norm_segmentations / len(scores)

            if no_norm:
                norm_segmentations=segmentations # remove norm

            # Compute PRO score & PW Auroc for all images
            pixel_scores = metrics.compute_pixelwise_retrieval_metrics(
                norm_segmentations, masks_gt)
                # segmentations, masks_gt
            full_pixel_auroc = pixel_scores["auroc"]
            pro = metrics.compute_pro(np.squeeze(np.array(masks_gt)), 
                                            norm_segmentations)

        else:
            full_pixel_auroc = -1 
            pro = -1

        return auroc, full_pixel_auroc, pro

    def predict(self, data, cls_names, prefix=""):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data, prefix)
        return self._predict(data, cls_names)

    #@profile
    def _predict_dataloader(self, dataloader, prefix):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()


        img_paths = []
        scores = []
        masks = []
        features = []
        labels_gt = []
        masks_gt = []
        all_json = []
        base_idx = 0
        from sklearn.manifold import TSNE

        #pdb.set_trace()
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    labels_gt.extend(data["is_anomaly"].numpy().tolist())
                    if data.get("mask", None) is not None:
                        masks_gt.extend(data["mask"].numpy().tolist())
                    image = data["image"]
                    img_paths.extend(data['image_path'])
                _scores, _masks = self._predict(image)
                for score, mask, is_anomaly in zip(_scores, _masks, data["is_anomaly"].numpy().tolist()):
                    scores.append(score)
                    masks.append(mask)
                    
                for p,s,l in zip(data['image_path'],_scores,data['is_anomaly'].numpy()):
                    tmp_d={"image_path":p,"socre":float(s),"gt_label":int(l)}
                    all_json.append(tmp_d)

                if save_img:
                    for idx, (m, i, g, l, p) in enumerate(zip(_masks,image,data['mask'].numpy(),data['is_anomaly'].numpy(),data['image_path'])):
                        #self.logger.logger.add_scalar("socre",s,idx)
                        self.logger.logger.add_image("mask",m[np.newaxis,...]*255,base_idx)
                        self.logger.logger.add_image("mask_gt",g,base_idx)
                        #self.logger.logger.add_image("features",f)
                        self.logger.logger.add_image("image",i,base_idx)
                        #tmp_d={"image_path":p,"socre":float(s),"gt_label":int(l)}
                        #all_json.append(tmp_d)
                        base_idx+=1

        
        with open(score_path,"w") as f:
            for j in all_json:
                tmp=json.dumps(j)
                f.write(tmp+"\n")

        return scores, masks, features, labels_gt, masks_gt

    def _predict(self, images, cls_names):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(images,
                                                 provide_patch_shapes=True, 
                                                 evaluation=True)
            patch_scores, image_scores = [], []
            _feats = features.reshape(batchsize, -1, features.shape[-1])
            # print(_feats.shape)
            for ind, cls_name in enumerate(cls_names):
                # print(cls_name)
                ps = ims = -self.header[cls_name](_feats[ind])
                patch_scores.append(ps)
                image_scores.append(ims)
            patch_scores = torch.cat(patch_scores, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            # print(patch_scores.shape, image_scores.shape)

            # patch_scores = patch_scores.cpu().numpy()
            # image_scores = image_scores.cpu().numpy()
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self.patch_maker.score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            features = features.reshape(batchsize, scales[0], scales[1], -1)
            #masks, features = self.anomaly_segmentor.convert_to_segmentation(patch_scores, features)
            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores, features)

        return list(image_scores), list(masks)#, list(features)

    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "params.pkl")

    def save_to_path(self, save_path: str, prepend: str = ""):
        LOGGER.info("Saving data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(params, save_file, pickle.HIGHEST_PROTOCOL)


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, top_k=0, stride=None):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 2:
            x = torch.max(x, dim=-1).values
        if x.ndim == 2:
            if self.top_k > 1:
                x = torch.topk(x, self.top_k, dim=1).values.mean(1)
            else:
                x = torch.max(x, dim=1).values
        if was_numpy:
            return x.numpy()
        return x
