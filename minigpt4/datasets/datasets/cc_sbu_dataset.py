import os
from PIL import Image, ImageDraw
import webdataset as wds
from minigpt4.datasets.datasets.base_dataset import BaseDataset
from minigpt4.datasets.datasets.caption_datasets import CaptionDataset
import numpy as np
import torch
import cv2
import random
import json

from minigpt4.processors.transform import Expand2square, PlainBoxFormatter, norm_box_xyxy
from mmdet.structures.mask import BitmapMasks


class CCSBUDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

    def to_dict(self, sample):
        return {
            "image": sample[0],
            "text_input": self.text_processor(sample[1]["caption"]),
        }


class CCSBUAlignDatasetWithAug(CaptionDataset):

    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, with_mask=False, with_ref=False):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        # Shikra的transform
        self.shikra_transform = Expand2square()
        self.mask_transform = Expand2square(background_color=(0,0,0))
        self.pbf = PlainBoxFormatter()
        self.with_mask = with_mask
        self.with_ref = with_ref

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        # ann = self.annotation[index]
        self.annotation = []
        # TODO: 改成Concat数据集
        choice = random.choice([0,1])
        if choice==0:
            files = json.load(open('/mnt/vdb1/whl/MiniGPT-4-main/cc_sbu_align/filtergood.json', "r"))['annotations']
        elif choice==1:
            files = json.load(open('/mnt/vdb1/whl/MiniGPT-4-main/cc_sbu_align/filterbad.json', "r"))['annotations']
        # print('choice',choice,ann)
        index = random.randint(0,len(files)-1)
        ann = files[index]

        # 处理图像和标注bounding box
        img_file = ann["image_id"]#'{}.jpg'.format(ann["image_id"])
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")
        # 提取bbox，并且调整
        width, height = image.size
        caption = ann["caption"]
        format_str, bboxes = self.pbf.extract(caption)
        bboxes = np.array(bboxes).reshape(-1, 4).tolist()
        if len(bboxes) != 0:
            bboxes = np.array([[box[1], box[0], box[3], box[2]] for box in bboxes])  # 太难顶了……得把这个纵轴搞回来
            whwh = np.array([width, height, width, height])
            origin_boxes = bboxes * whwh
            image, labels = self.shikra_transform(image, {'boxes': origin_boxes.tolist()})
            norm_bboxes = [norm_box_xyxy(box, w=width, h=height) for box in labels['boxes']]
            caption = format_str.format(*[str(bbox) for bbox in norm_bboxes])  # 把处理过的图放进去
        else:
            image, _ = self.shikra_transform(image)
        image = self.vis_processor(image)

        # 处理mask
        if self.with_mask:
            subname = image_path.split('/')[-4]
            if image_path.split('/')[-2]=='bad':
                subphase  = 'Anomaly'
            elif image_path.split('/')[-2]=='good':
                subphase  = 'Normal'
            maskdir = '/mnt/vdb1/whl/VAND-APRIL-GAN-master/processresults/visa/zero_shot/imgs/'+subname+'/'+subphase+'/'+image_path.split('/')[-1][0:-4]+'_mask.JPG'
            # maskdir = image_path.replace('')
            mask = cv2.imread(maskdir)
            mask = np.array(self.mask_transform(Image.fromarray(mask))[0])
            mask = torch.from_numpy(mask[:,:,0:1])
            mask = (mask/255.0).float()
            mask = mask.permute([2,0,1])
        
        question = ann["question"]
        if self.with_ref:
            try:
                refimage_path = os.path.join(self.vis_root, img_file).replace('bad','good')
            except:
                refimage_path = os.path.join(self.vis_root, img_file).replace('good','good')

            refroot = '/'.join(refimage_path.split('/')[0:-1])
            refnames = random.sample(os.listdir(refroot),1)
            refimage = Image.open(refroot+'/'+refnames[0]).convert("RGB")
            refimage, _ = self.shikra_transform(refimage)
            refimage = self.vis_processor(refimage)

        ret = {
            "image": image,
            "question":'###Human: '+question+' ###Assistant: ',
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }
        if self.with_mask:
            ret["masks"] = mask
        if self.with_ref:
            ret["refimage"] = refimage

        return ret


from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


def _encode_text_with_prompt_ensemble(model: CLIPTextModel, objs, tokenizer: CLIPTokenizer, device='cpu'):
    prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
    prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
    prompt_state = [prompt_normal, prompt_abnormal]
    prompt_templates = ['a bad photo of a {}.', 'a low resolution photo of the {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a bright photo of a {}.', 'a dark photo of the {}.', 'a photo of my {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a photo of one {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'a low resolution photo of a {}.', 'a photo of a large {}.', 'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'a dark photo of a {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']

    text_prompts = {}
    for obj in objs:
        text_features = []
        for i in range(len(prompt_state)):
            prompted_state = [state.format(obj) for state in prompt_state[i]]
            prompted_sentence = []
            for s in prompted_state:
                for template in prompt_templates:
                    prompted_sentence.append(template.format(s))
            # print(len(prompted_sentence))
            # 输入是一个N=175或者245长度的描述，输出为(N, 每个单词长度, dims), 其中dims=768
            inputs = tokenizer(prompted_sentence, padding=True, return_tensors='pt')  # 输出有两个key: input_ids和attention_masks
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            # print(inputs['input_ids'].shape)
            # 如果是CLIPTextModel输出有两个key: last_hidden_state, pooler_output
            # 如果是CLIPTextModelWithProjection输出有两个key: last_hidden_state, text_embeds
            outputs = model(**inputs)  
            # print(outputs['last_hidden_state'].shape, outputs['text_embeds'].shape)  
            class_embeddings = outputs['text_embeds']
            # class_embeddings = outputs['pooler_output']
            # print(class_embeddings.shape)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # 把同一类的表述平均一下，成为一个中间的结果。
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            text_features.append(class_embedding)

        # print([item.shape for item in text_features])
        text_features = torch.stack(text_features, dim=1)  # 输出是(768, 2)
        # print("text_features:", text_features.shape)
        text_prompts[obj] = text_features.detach().cpu()

    return text_prompts


def encode_text_with_prompt_ensemble(objs, device='cpu'):
    text_model = CLIPTextModelWithProjection.from_pretrained("/home/lyz/.cache/huggingface/hub/models--openai--clip-vit-large-patch14").to(device)
    tokenizer = CLIPTokenizer.from_pretrained("/home/lyz/.cache/huggingface/hub/models--openai--clip-vit-large-patch14")
    with torch.no_grad():
        return _encode_text_with_prompt_ensemble(text_model, objs, tokenizer, device=device)


VISA_OBJS = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
MVTEC_OBJS = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']


class CCSBUAlignDataset(CaptionDataset):

    def __init__(self, vis_processor, text_processor, vis_root, ve_root, ann_paths, with_mask=False, with_ref=False, with_pos=True, with_gt_seg=False):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.pbf = PlainBoxFormatter()
        self.with_mask = with_mask
        self.with_ref = with_ref
        self.with_pos = with_pos
        self.with_gt_seg = with_gt_seg
        self.ve_root = ve_root
        print("Build Dataset with gt seg", self.with_gt_seg)
        # 这里添加新参数之后，要去dataset builder那边传参。
        if self.with_gt_seg:
            self.text_features = torch.load('/home/ubuntu/lyz/archive/pretrain/clip_visa_text_features.pth')
    
    def get_gt_seg(self, img_path, height, width, is_anomaly):
        if not is_anomaly:
            return Image.fromarray(np.zeros((height, width)), mode='L')
        gt_seg_path = img_path.split('/')
        gt_seg_path = gt_seg_path[:-3] + ['ground_truth'] + gt_seg_path[-3:]
        gt_seg_path = '/'+os.path.join(*gt_seg_path)[:-3]+'png'
        img_mask = np.array(Image.open(gt_seg_path).convert('L')) > 0
        img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
        return img_mask

    def __getitem__(self, index):
        # TODO this assumes image input, not general enough
        # ann = self.annotation[index]
        self.annotation = []
        # TODO: 改成Concat数据集
        choice = random.choice([0,1])
        if choice==0:
            files = json.load(open(os.path.join(self.vis_root, 'train_good.json'), "r"))['annotations']
        elif choice==1:
            # print(os.path.join(self.vis_root, 'train_bad.json'))
            files = json.load(open(os.path.join(self.vis_root, 'train_bad.json'), "r"))['annotations']
        # print('choice',choice,ann)
        index = random.randint(0,len(files)-1)
        ann = files[index]

        # 处理图像和标注bounding box
        img_file = ann["image_id"]#'{}.jpg'.format(ann["image_id"])
        image_path = os.path.join(self.vis_root, '2cls_highshot',img_file)
        image = Image.open(image_path).convert("RGB")
        # 提取bbox，并且调整
        bboxes = ann['gt_bboxes']
        width, height = image.size
        caption = ann["caption"]
        data_sample = {
            'img': np.array(image),
        }
        if len(bboxes) != 0:
            # print(bboxes)
            _, bboxes = self.pbf.extract(str(bboxes))
            # print(bboxes)
            bboxes = np.array(bboxes).reshape(-1, 4).tolist()
            bboxes = np.array([[box[1], box[0], box[3], box[2]] for box in bboxes])  # 太难顶了……得把这个纵轴搞回来
            whwh = np.array([width, height, width, height])
            bboxes = bboxes * whwh
            data_sample['gt_bboxes'] = bboxes
            data_sample['gt_bboxes_labels'] = np.array(ann['gt_bboxes_labels'])

        # 处理mask
        if self.with_mask:
            subname = image_path.split('/')[-4]
            if image_path.split('/')[-2]=='bad':
                subphase  = 'Anomaly'
            elif image_path.split('/')[-2]=='good':
                subphase  = 'Normal'
            maskdir = self.ve_root +'/visa/zero_shot/imgs/'+subname+'/'+subphase+'/'+image_path.split('/')[-1][0:-4]+'_mask.JPG'
            # maskdir = image_path.replace('')
            mask = cv2.imread(maskdir)
            # cv2的resize函数脑子有问题，格式是(width, height)而不是numpy的shape
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            mask = mask[:,:,0].astype(np.uint8)  # range: (0, 255)
            data_sample['gt_seg_map'] = mask  # （H, W)
        
        if self.with_gt_seg:
            gt_seg = self.get_gt_seg(image_path, height, width, is_anomaly=(choice==1))
            data_sample['gt_seg_map'] = np.asarray(gt_seg).astype(np.uint8)
        
        data_sample = self.vis_processor(data_sample)
        
        question = ann["question"]
        if 'gt_bboxes' in data_sample:

            new_height, new_width = data_sample['img'].shape[1:]
            # print(new_width, new_height)
            norm_bboxes = [norm_box_xyxy(box, w=new_width, h=new_height) for box in data_sample['gt_bboxes']]
            if self.with_pos:
                boxes_with_labels = [f"{label}{str(bbox)}" for label, bbox in zip(data_sample['gt_bboxes_labels'], norm_bboxes)]
            else:
                boxes_with_labels = [f"{label}" for label, bbox in zip(data_sample['gt_bboxes_labels'], norm_bboxes)]
            caption = caption.format(ann['scene'], ', '.join(boxes_with_labels))  # 把处理过的图放进去
        else:
            caption = caption.format(ann['scene'])

        ret = {
            "image": data_sample['img'],
            "question": "<Img><ImageHere></Img>"+question,
            "text_input": caption,
            "image_id": index,
            # "test": data_sample
        }
        if self.with_mask:
            ret['masks'] = data_sample['gt_seg_map'].astype(int) / 255.0  # change range: (0, 255) -> (0, 1)
        if self.with_gt_seg:
            gt_seg = data_sample['gt_seg_map'].astype(int) / 255.0
            gt_seg[gt_seg > 0.5], gt_seg[gt_seg <= 0.5] = 1, 0
            ret['gt_seg'] = gt_seg
            ret['text_features'] = self.text_features[ann['scene']]
        return ret
