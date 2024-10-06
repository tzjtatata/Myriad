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

# from minigpt4.processors.transform import PlainBoxFormatter
from minigpt4.datasets.self_sup_tasks import patch_ex
from torchvision.transforms import transforms as T
from torchvision import transforms

# QuestionPrompts = [
   # "Can you find any abnormal part in this image?", 
    # "Find out if there are defects in this image. ",
# ]
# branchs = [
#     # "Can you find any abnormal part in this image?", 
#     "According on IAD expert opinions, ",
#     "According to IAD expert opinions and corresponding visual descriptions, ",
#     "According to IAD expert visual descriptions, ",
# ]
# QuestionPrompts = [
#     # "Can you find any abnormal part in this image?", 
#     "detect the defects in this image. ",
#     "count the defects numbers in this image. ",
#     "locate the defects in this image. ",
# ]
QuestionPrompts = [
    "This image may be simulated by photo editing. According on IAD expert opinions, find out if there are defects in this image.",
    "This image may be simulated by photo editing. According to IAD expert opinions and corresponding visual descriptions, find out if there are defects in this image.",
    "This image may be simulated by photo editing. According to IAD expert visual descriptions, find out if there are defects in this image.",
]
def imreader(args):
    i, anns = args
    ann = anns[i]
    ann['image'] = Image.open(ann['path']).convert("RGB")
    ann['ve'] = cv2.imread(ann['ve_path'])
    np.array(ann['image'])  # 为了避免Image的延迟机制

# MVTEC
MVTEC_WIDTH_BOUNDS_PCT = {'bottle':((0.03, 0.4), (0.03, 0.4)), 'cable':((0.05, 0.4), (0.05, 0.4)), 'capsule':((0.03, 0.15), (0.03, 0.4)), 
                    'hazelnut':((0.03, 0.35), (0.03, 0.35)), 'metal_nut':((0.03, 0.4), (0.03, 0.4)), 'pill':((0.03, 0.2), (0.03, 0.4)), 
                    'screw':((0.03, 0.12), (0.03, 0.12)), 'toothbrush':((0.03, 0.4), (0.03, 0.2)), 'transistor':((0.03, 0.4), (0.03, 0.4)), 
                    'zipper':((0.03, 0.4), (0.03, 0.2)), 
                    'carpet':((0.03, 0.4), (0.03, 0.4)), 'grid':((0.03, 0.4), (0.03, 0.4)), 
                    'leather':((0.03, 0.4), (0.03, 0.4)), 'tile':((0.03, 0.4), (0.03, 0.4)), 'wood':((0.03, 0.4), (0.03, 0.4))}

# k, x0 pairs
MVTEC_INTENSITY_LOGISTIC_PARAMS = {'bottle':(1/12, 24), 'cable':(1/12, 24), 'capsule':(1/2, 4), 'hazelnut':(1/12, 24), 'metal_nut':(1/3, 7), 
            'pill':(1/3, 7), 'screw':(1, 3), 'toothbrush':(1/6, 15), 'transistor':(1/6, 15), 'zipper':(1/6, 15),
            'carpet':(1/3, 7), 'grid':(1/3, 7), 'leather':(1/3, 7), 'tile':(1/3, 7), 'wood':(1/6, 15)}

MVTEC_BACKGROUND = {'bottle':(200, 60), 'screw':(200, 60), 'capsule':(200, 60), 'zipper':(200, 60), 
              'hazelnut':(20, 20), 'pill':(20, 20), 'toothbrush':(20, 20), 'metal_nut':(20, 20)}


def get_position(centers):
    position = []
    for center in centers:
        center_x = center[0] / 224
        center_y = center[1] / 224

        if center_x <= 1/3 and center_y <= 1/3:
            position.append('top left')
        elif center_x <= 1/3 and center_y > 1/3 and center_y <= 2/3:
            position.append('top')
        elif center_x <= 1/3 and center_y > 2/3:
            position.append('top right')

        elif center_x <= 2/3 and center_y <= 1/3:
            position.append('left')
        elif center_x <= 2/3 and center_y > 1/3 and center_y <= 2/3:
            position.append('center')
        elif center_x <= 2/3 and center_y > 2/3:
            position.append('right')

        elif center_y <= 1/3:
            position.append('bottom left')
        elif center_y > 1/3 and center_y <= 2/3:
            position.append('bottom')
        elif center_y > 2/3:
            position.append('bottom right')
    return list(set(position))
def find_first_file_in_directory(directory_path):
    try:
        file_list = os.listdir(directory_path)
        for item in file_list:
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path):
                return item_path
        return None

    except OSError as e:
        print(f"Error while accessing directory: {e}")
        return None

class AnomalyDetectionDataset(BaseDataset):
    DatasetName='AnomalyDetection'

    def __init__(self, vis_processor, text_processor, vis_root, ve_root, ann_paths, img_size=224, crop_size=224, version=0,with_mask=False, with_ref=False, with_pos=False, is_preload=False, stage='train', nsa_max_width=0.4):
        # self.pbf = PlainBoxFormatter()
        self.version = version
        self.with_mask = with_mask
        self.with_ref = with_ref
        self.with_pos = with_pos
        self.ve_root = ve_root
        self.stage = stage
        if nsa_max_width != 0.4:
            print("NSA max half-width is set to:", nsa_max_width)
        if 'VISA' in ann_paths[0]:
            self.self_sup_args={
                'width_bounds_pct': ((0.03, 0.4), (0.03, 0.4)),
                'intensity_logistic_params': (1/12, 24),
                'num_patches': 2,
                'min_object_pct': 0,
                'min_overlap_pct': 0.25,
                'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                'shift':True, 
                'same':False, 
                'mode':cv2.NORMAL_CLONE, 
                'label_mode':'logistic-intensity',
                'skip_background': None,
                'resize_bounds': (.5, 2)
            }
        else:
            self.self_sup_args={
                'num_patches': 2, #if single_patch else NUM_PATCHES.get(class_name),
                'min_object_pct': 0,
                'min_overlap_pct': 0.25,
                'gamma_params':(2, 0.05, 0.03), 'resize':True, 
                'shift':True, 
                'same':False, 
                'mode':cv2.NORMAL_CLONE,
                'label_mode':'logistic-intensity',
            }

        # self.transform = T.Resize(
        #     (224, 224), interpolation=T.InterpolationMode.BICUBIC
        # )
        self.transform = transforms.Compose(
            [transforms.Resize(
                    img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop_size),])
        if is_preload and self.with_mask:
            super().__init__(vis_processor, text_processor, vis_root, ann_paths, is_preload, preload_fn=imreader)  # 预载vision expert mask.
        else:
            super().__init__(vis_processor, text_processor, vis_root, ann_paths, is_preload)  # 不预载vision expert mask
        
        if self.with_ref:
            self.references = {}
            from minigpt4.datasets.datasets.cc_sbu_dataset import MVTEC_OBJS
            for ind, item in enumerate(self.annotation):
                scene = item['img_path'].split('/')[1]
                if scene not in self.references and item['is_anomaly'] == '0':
                    assert scene in MVTEC_OBJS, f"Scene {scene} not in MVTEC_OBJS"
                    self.references[scene] = self.prepare_img(ind)       
            print(f"With Reference: {list(self.references.keys())}")
    
    def __repr__(self) -> str:
        example = self[0]
        ret = f"{self.DatasetName}: \n" \
        f"\twith Vision Expert: {self.with_mask} \n" \
        f"\tBuild Info: \n" f"\t\tImage Root: {self.vis_root} \n" f"\t\tVision Expert Root: {self.ve_root} \n" f"\t\tVision Processor: {self.vis_processor} \n" \
        f"\tExample: {list(example.keys())} \n" f"\t\tImage: {example['image'].shape} \n" f"\t\tQuestion: {example['question']} \n" f"\t\tAnswer: {example['text_input']} \n" 
        if self.with_mask:
            ret += f"\t\tVision Expert: {example['masks'].shape} \n"
        return ret
    
    def get_gt_seg(self, img_path, height, width, is_anomaly):
        if not is_anomaly:
            return Image.fromarray(np.zeros((height, width)), mode='L')
        gt_seg_path = img_path.split('/')
        gt_seg_path = gt_seg_path[:-3] + ['ground_truth'] + gt_seg_path[-3:]
        gt_seg_path = '/'+os.path.join(*gt_seg_path)[:-3]+'png'
        img_mask = np.array(Image.open(gt_seg_path).convert('L')) > 0
        img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
        return img_mask

    def prepare_img(self, index):
        rel_path = self.annotation[index]['img_path']
        if self.is_preload:
            return self._cache[rel_path].copy()
        else:
            img_path = self.get_image_path(rel_path)
            return Image.open(img_path).convert("RGB")
    
    def prepare_ve(self, index):
        rel_path = self.annotation[index]['img_path']
        if self.is_preload:
            return self._ve_cache[rel_path].copy()
        else:
            ve_path = os.path.join(self.ve_root, self.annotation[index]['ve_path'])
            return cv2.imread(ve_path)
    
    def construct_preload_maps(self):
        return [{'path': self.get_image_path(ann['img_path']), 'rel_path': ann['img_path'], 've_path': os.path.join(self.ve_root, ann['ve_path'])} for ann in self.annotation]
    
    def post_preload(self, results):
        self._cache = {item['rel_path']: item['image'] for item in results}
        if self.with_mask:
            self._ve_cache = {item['rel_path']: item['ve'] for item in results}
    
    def load_annotations(self):
        self.annotation = []
        for anno_path in self.ann_paths:
            with jsonlines.open(os.path.join(self.vis_root, anno_path), 'r') as reader:
                self.annotation.extend(list(reader))
        print(f"In {self.DatasetName} Dataset, Has Samples: {len(self.annotation)}")
    
    def get_image_path(self, rel_path):
        return os.path.join(self.vis_root, rel_path)

    def get_ve_path(self, ve_path):
        return os.path.join(self.ve_root, ve_path)

    def get_class_name(self, index):
        if 'MVTEC' in self.ann_paths[0]:
            ds = 'mvtec'
        else:
            ds = 'visa'
        
        return ds, self.annotation[index]['img_path'].split('/')[1]

    def __getitem__(self, index):
        ann = self.annotation[index]

        # total_tic = time.time()
        # tic = time.time()
        # 处理图像和标注bounding box
        image = self.prepare_img(index)
        # print("准备数据:", time.time() - tic)
        # tic = time.time()
        # 提取bbox，并且调整
        width, height = image.size
        caption = ann["caption"]
        # self.stage = 'train'
        if self.stage == 'train':
            image = self.transform(image)
            src_index = np.random.randint(len(self))
            while src_index == index:
                src_index = np.random.randint(len(self))
            src_image = self.transform(self.prepare_img(src_index))
            # print("两次Resize Transform:", time.time() - tic)
            # tic = time.time()
            
            # TODO: 在VisA上，有可能会aug到背景上，这个其实不太好；我们应该只在前景物体上进行augment。这样就会涉及到一个比较好的前景分割的结果。
            ds, class_name = self.get_class_name(index)
            if ds == 'mvtec':
                self_sup_args = {
                    'width_bounds_pct': MVTEC_WIDTH_BOUNDS_PCT.get(class_name), 
                    'intensity_logistic_params': MVTEC_INTENSITY_LOGISTIC_PARAMS.get(class_name), 
                    'skip_background': MVTEC_BACKGROUND.get(class_name),
                }
            else:
                self_sup_args = {}
            
            aug_image, mask, centers = patch_ex(np.asarray(image), np.asarray(src_image), **self_sup_args, **self.self_sup_args)
            while np.sum(mask) == 0:
                # print("aaaa")
                aug_image, mask, centers = patch_ex(np.asarray(image), np.asarray(src_image), **self_sup_args, **self.self_sup_args)
            # print("模拟异常:", time.time()-tic)
            # tic=time.time()
            if len(centers) > 0:
                position = []
                for center in centers:
                    center_x = center[0] / 224
                    center_y = center[1] / 224

                    if center_x <= 1/3 and center_y <= 1/3:
                        position.append('upper left')
                    elif center_x <= 1/3 and center_y > 1/3 and center_y <= 2/3:
                        position.append('top')
                    elif center_x <= 1/3 and center_y > 2/3:
                        position.append('upper right')

                    elif center_x <= 2/3 and center_y <= 1/3:
                        position.append('left')
                    elif center_x <= 2/3 and center_y > 1/3 and center_y <= 2/3:
                        position.append('center')
                    elif center_x <= 2/3 and center_y > 2/3:
                        position.append('right')

                    elif center_y <= 1/3:
                        position.append('lower left')
                    elif center_y > 1/3 and center_y <= 2/3:
                        position.append('bottom')
                    elif center_y > 2/3:
                        position.append('lower right')

            
                if len(centers) > 1:
                    # abnormal_describe =  "Yes, there are " + str(len(centers)) + " anomalies in the image, they are at the "
                    abnormal_describe =  "Yes, there exists anomalies in the image, they are at the "
                    for i in range(len(centers)):
                        if i == 0:
                            abnormal_describe += position[i]
                        elif i == 1 and position[i] != position[i-1]:
                            if i != len(centers) - 1:
                                abnormal_describe += ", "
                                abnormal_describe += position[i]
                            else:
                                abnormal_describe += " and " + position[i] + " of the image."
                    
                        elif i == 1 and position[i] == position[i-1]:
                            if i == len(centers) - 1:
                                abnormal_describe += " of the image."

                else:
                    abnormal_describe = "Yes, there exists anomalies in the image, at the " + position[0] + " of the image."
                

            aug_sample = {
                'img': aug_image,
                'gt_seg_map': mask,
                # 'gt_bboxes': np.array(boxes).astype(float)
            }
            aug_sample = self.vis_processor(aug_sample)

        data_sample = {
            'img': np.asarray(self.transform(image))
        }
        data_sample = self.vis_processor(data_sample)
        # print("裁剪/Resize正常图片:", time.time() - tic)
        # assert np.sum(mask) != 0, f"Boxes: {data_sample['gt_bboxes']}"
        if self.version == 0:
            normal_describe = "No, there exists no anomalies in the image."         
            abnormal_describe = "Yes, there exists anomalies in the image."
        elif self.version == 1:
            normal_describe = "No, there exists no anomalies in the image."         
            abnormal_describe = "Yes, there exists anomalies in the image. These anomalies are simulated by photo editing."
        else:
            print(f"Not Support Version.{self.version}")


        ret = {
            "image": data_sample['img'],
            "scene": ann['img_path'].split('/')[1],  # 0通常是mvtec或者是1cls
            "question": "<Img><ImageHere></Img>"+QuestionPrompts[1],
            "question2": "<Img><ImageHere></Img>"+QuestionPrompts[1],
            "question3": "<Img><ImageHere></Img>"+QuestionPrompts[1],
            "text_input": normal_describe,
            "image_id": index,
            "is_anomaly": (ann['is_anomaly'] == '1'), 
            "img_path": os.path.join(self.vis_root, ann['img_path']), 
        }

        if self.stage == 'train': 
            ret['aug_image'] = aug_sample['img']
            # ret['gt_mask'] = aug_sample['gt_seg_map']
            if np.sum(aug_sample['gt_seg_map']) == 0.0:
                ret['aug_text_input'] = normal_describe
            else:
                ret['aug_text_input'] = abnormal_describe

        return ret

