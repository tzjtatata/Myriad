"""
    AnomalyDetection数据集包含了Anomaly Detection(或者是Defect Classification -- DC)任务。
    区别于之前的CCSBUAlignDataset, 这个数据集全部由Normal数据构成。因此不用来作为测试。

"""
import os
from PIL import Image
from minigpt4.datasets.datasets.base_dataset import BaseDataset, imreader
import numpy as np
import cv2
import torch
import jsonlines
import random

# from minigpt4.processors.transform import PlainBoxFormatter
from minigpt4.datasets.self_sup_tasks import patch_ex
from torchvision.transforms import transforms as T
from torchvision import transforms


instructions = [
    "find out if there are defects in this image.",
    "are there any anomalies in this image?", 
    # "is there any anomaly present in the image?", 
    # "does the image exhibit any abnormalities?", 
    # "are there any irregularities in the image?", 
    # "can you identify any anomalies in the image?"
    # "is there any deviation from the normal pattern in the image?", 
    # "do you notice any distortions or abnormalities in the image?", 
    # "is the image consistent or does it show any anomalies?", 
    # "are there any unexpected elements in the image?", 
    # "does the image display any signs of irregularity?", 
    "can you identify any unusual features in the image?", 
    # "examine the image for any anomalies.", 
    # "analyze the picture to identify any abnormalities.", 
    # "inspect the image and determine if there are any irregularities.", 
    # "look closely at the picture and check for any signs of anomalies.", 
    # "scrutinize the image for any unexpected or unusual elements.", 
    # "evaluate the picture and point out any irregularities from the norm.", 
    # "study the image and note any distortions or abnormalities.", 
    # "assess the picture and determine if there are any discrepancies.", 
    "examine the image closely and identify any potential anomalies.", 
    # "carefully review the picture and flag any possible irregularities.", 
]


instructionTemplates = [
    "This image has not been edited. According on IAD expert opinions, {}",
    "This image has not been edited. According to IAD expert opinions and corresponding visual descriptions, {}",
    "This image has not been edited. According to IAD expert visual descriptions, {}",
]


class TwoClassAnomalyDetectionDataset(BaseDataset):
    DatasetName='2-cls IAD' 

    def __init__(
            self, 
            vis_processor, 
            text_processor, 
            vis_root, ann_paths, 
            img_size=224, crop_size=224, 
            dynamic_instruction=False, 
            is_preload=False, stage='train', 
            version='2'):
        self.version = version
        self.stage = stage
        self.dynamic_instruction = dynamic_instruction

        self.transform = transforms.Compose(
            [transforms.Resize(
                    img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop_size),]
        )

        super().__init__(vis_processor, text_processor, vis_root, ann_paths, is_preload)  # 不预载vision expert mask
    
    def __repr__(self) -> str:
        example = self[0]
        ret = f"{self.DatasetName}: \n" \
        f"\tBuild Info: \n" f"\t\tImage Root: {self.vis_root} \n" f"\t\tVision Processor: {self.vis_processor} \n" \
        f"\tExample: {list(example.keys())} \n" f"\t\tImage: {example['image'].shape} \n" f"\t\tQuestion: {example['question']} \n" f"\t\tAnswer: {example['text_input']} \n" 
        # if self.with_mask:
        #     ret += f"\t\tVision Expert: {example['masks'].shape} \n"
        return ret

    def prepare_img(self, index):
        rel_path = self.annotation[index]['img_path']
        if self.is_preload:
            return self._cache[rel_path].copy()
        else:
            img_path = self.get_image_path(rel_path)
            return Image.open(img_path).convert("RGB")
    
    def prepare_gt(self, index, width, height):
        if self.annotation[index]['is_anomaly'] == '1':
            rel_path = self.annotation[index]['img_path']
            gt_path = os.path.join(self.vis_root, rel_path.replace('test', 'ground_truth').replace('.png', '_mask.png'))  # 对于mvtec的结构来说是这样的。
            gt = np.array(Image.open(gt_path).convert('L')) > 0
        else: 
            gt = np.zeros((height, width))
        return Image.fromarray(gt.astype(np.uint8) * 255, mode='L')
    
    def construct_preload_maps(self):
        return [{'path': self.get_image_path(ann['img_path']), 'rel_path': ann['img_path']} for ann in self.annotation]
    
    def post_preload(self, results):
        self._cache = {item['rel_path']: item['image'] for item in results}
        # if self.with_mask:
        #     self._ve_cache = {item['rel_path']: item['ve'] for item in results}
    
    def load_annotations(self):
        self.annotation = []
        for anno_path in self.ann_paths:
            with jsonlines.open(os.path.join(self.vis_root, anno_path), 'r') as reader:
                self.annotation.extend(list(reader))
        print(f"In {self.DatasetName} Dataset, Has Samples: {len(self.annotation)}")
    
    def get_image_path(self, rel_path):
        return os.path.join(self.vis_root, rel_path)

    def get_ve_path(self, ve_path):
        return os.path.join(self.vis_root, ve_path)

    def get_class_name(self, index):
        if 'mvtec' in self.annotation[index]['img_path']:
            ds = 'mvtec'
        else:
            ds = 'visa'
        
        return ds, self.annotation[index]['img_path'].split('/')[1]

    def get_description_v2(self):
        abnormal_describe = "Yes, there exists anomalies in the image."
        normal_describe = "No, there exists no anomalies in the image."
        return abnormal_describe, normal_describe
    
    def get_description_v4(self):
        abnormal_describe = "Yes, there exists anomalies in the image. These anomalies are caused by unexpected incidents during the product manufacturing process."
        normal_describe = "No, there exists no anomalies in the image."
        return abnormal_describe, normal_describe

    def get_description_v1(self):
        return "This image has defects.", "This object looks perfect."

    def __getitem__(self, index):
        ann = self.annotation[index]

        # total_tic = time.time()
        # tic = time.time()
        # 处理图像和标注bounding box
        image = self.prepare_img(index)
        # print("准备数据:", time.time() - tic)
        # tic = time.time()
        # 提取bbox，并且调整
        image = self.transform(image)
        is_anomaly = (ann['is_anomaly'] == '1')
        scene = ann['img_path'].split('/')[1]

        if self.version == '2':
            abnormal_describe, normal_describe = self.get_description_v2()
        elif self.version == '1':
            abnormal_describe, normal_describe = self.get_description_v1()
        elif self.version == '3':
            abnormal_describe, normal_describe = self.get_description_v2()
            if is_anomaly:
                defect_type = ann['img_path'].split('/')[-2]
                if defect_type == 'combined':
                    defect_type = 'several kinds of defects'
                # abnormal_describe = abnormal_describe + f" The image shows broken {scene.replace('_', ' ')} with {defect_type.replace('_', ' ')}."
                abnormal_describe = abnormal_describe + f" The image shows broken objects with {defect_type.replace('_', ' ')}."
            else:
                normal_describe = normal_describe + f" The image shows perfect objects."
        elif self.version == '4':
            abnormal_describe, normal_describe = self.get_description_v4()
            
        else:
            raise NotImplementedError(f"Not Implement V.{self.version} descriptions. ")
        # print(self.get_class_name(index)[0])

        data_sample = {
            'img': np.asarray(image)
        }
        data_sample = self.vis_processor(data_sample)
        
        
        instruction = instructions[0]
        ret = {
            "image": data_sample['img'],
            "scene": scene,  # 0通常是mvtec或者是1cls
            "question": "<Img><ImageHere></Img>"+instructionTemplates[1].format(instruction),
            "question2": "<Img><ImageHere></Img>"+instructionTemplates[1].format(instruction),
            "question3": "<Img><ImageHere></Img>"+instructionTemplates[1].format(instruction),
            "text_input": abnormal_describe if is_anomaly else normal_describe,
            "image_id": index,
            "is_anomaly": is_anomaly, 
            "img_path": os.path.join(self.vis_root, ann['img_path']), 
        }

        return ret

