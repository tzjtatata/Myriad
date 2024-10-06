
from minigpt4.models.header import *
import torch.nn.functional as F
from minigpt4.models.model.ImageBind import *
from minigpt4.models.model.ImageBind import data
from minigpt4.models.model.modeling_llama import LlamaForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
import kornia as K

import torch
from torch.nn.utils import rnn
import random
import os
import csv
import jsonlines
class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k):
        super(LinearLayer, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for i in range(k)])

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                tokens[i] = tokens[i].transpose(0,1)
                tokens[i] = self.fc[i](tokens[i][:, 1:, :])
            else:
                B, C, H, W = tokens[i].shape
                tokens[i] = self.fc[i](tokens[i].view(B, C, -1).permute(0, 2, 1).contiguous())
        return tokens
    
CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper', 'object',
               'candle', 'cashew', 'chewinggum', 'fryum', 'macaroni', 'pcb', 'pipe fryum']

prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']

prompt_state = [prompt_normal, prompt_abnormal]
prompt_templates = ['a photo of a {}.', 'a photo of the {}.']

objs = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper', 'object',
        'candle', 'cashew', 'chewinggum', 'fryum', 'macaroni', 'pcb', 'pipe fryum', 'macaroni1', 'macaroni2','pcb1', 'pcb2', 'pcb3', 'pcb4', 'capsules']


prompt_sentences = {}

for obj in objs:
    prompt_sentence_obj = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(obj) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        prompted_sentence = data.load_and_transform_text(prompted_sentence, torch.cuda.current_device())
        prompt_sentence_obj.append(prompted_sentence)
    prompt_sentences[obj] = prompt_sentence_obj
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
def encode_text_with_prompt_ensemble(model, obj, device):

    global prompt_sentences
    # print('aaaaaaaaaaaaaa',obj)
    normal_sentences = []
    abnormal_sentences = []
    for idx in range(len(obj)):
        sentence = prompt_sentences[obj[idx].replace('_', ' ')]
        # print(sentence)
        normal_sentences.append(sentence[0])
        abnormal_sentences.append(sentence[1])

    normal_sentences = torch.cat(normal_sentences).to(device)
    abnormal_sentences = torch.cat(abnormal_sentences).to(device)
    # print(normal_sentences,abnormal_sentences)

    class_embeddings_normal = model({ModalityType.TEXT: normal_sentences})[ModalityType.TEXT][0]
    class_embeddings_abnormal = model({ModalityType.TEXT: abnormal_sentences})[ModalityType.TEXT][0]

    # class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
    class_embeddings_normal = class_embeddings_normal.reshape((len(obj), len(prompt_templates) * len(prompt_normal), 1024))
    class_embeddings_normal = class_embeddings_normal.mean(dim=1, keepdim=True)
    class_embeddings_normal = class_embeddings_normal / class_embeddings_normal.norm(dim=-1, keepdim=True)

    class_embeddings_abnormal = class_embeddings_abnormal.reshape((len(obj), len(prompt_templates) * len(prompt_abnormal), 1024))
    class_embeddings_abnormal = class_embeddings_abnormal.mean(dim=1, keepdim=True)
    class_embeddings_abnormal = class_embeddings_abnormal / class_embeddings_abnormal.norm(dim=-1, keepdim=True)

    text_features = torch.cat([class_embeddings_normal, class_embeddings_abnormal], dim=1)

    return text_features


class adrefexpert(nn.Module):

    def __init__(self,round_index,k_shot, pt='mvtec') -> None:
        super().__init__()
        imagebind_ckpt_path = './pretrained_models/anomalyGPT_ckpt/imagebind_huge.pth'
        args = {'model': 'openllama_peft', 'imagebind_ckpt_path': './pretrained_models/anomalyGPT_ckpt/imagebind_huge.pth', 'vicuna_ckpt_path': '/mnt/vdb1/pretrained_models/vicuna-7b-v0', 'anomalygpt_ckpt_path': '/mnt/vdb1/whl/AnomalyGPT-old/code/ckpt/pytorch_mvtec_model.pt', 'delta_ckpt_path': '/mnt/vdb1/whl/AnomalyGPT-old/pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt', 'stage': 2, 'max_tgt_len': 128, 'lora_r': 32, 'lora_alpha': 32, 'lora_dropout': 0.1}
        self.visual_encoder, self.visual_hidden_size = imagebind_model.imagebind_huge(args)  # 这里的args只用到了'layers'这个key。也就是具体输出哪些层。
        imagebind_ckpt = torch.load(imagebind_ckpt_path, map_location=torch.device('cpu'))
        self.visual_encoder.load_state_dict(imagebind_ckpt, strict=True)
        self.image_decoder = LinearLayer(1280, 1024, 4)

        save_model = torch.load(f'./pretrained_models/anomalyGPT_ckpt/pytorch_{pt}_model.pt', map_location=torch.device('cpu'))

        new_state_dict = self.image_decoder.state_dict()
        for name, param in save_model.items():
            if 'image_decoder.' in name:
                name_ = '.'.join(name.split('.')[1:])
                if name_ in new_state_dict:
                    new_state_dict[name_] = param
        
        self.image_decoder.load_state_dict(new_state_dict)
        self.image_decoder.eval()  #t
        for name, param in self.image_decoder.named_parameters():
            param.requires_grad = False

        # free vision encoder
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder.eval()
            
        if k_shot==0:
            k_shot = k_shot + 1

        datas_csv_path = './data/visa/split_csv/1cls.csv'
        VISA_CLASS_NAMES = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2','pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
        self.visa_normal_img_path = {}
        self.visa_references = {}
        for class_name in VISA_CLASS_NAMES:
            self.visa_normal_img_path[class_name] = []
        with open(datas_csv_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] in VISA_CLASS_NAMES and len(self.visa_normal_img_path[row[0]]) < round_index * 4 + k_shot and row[1] == 'train':
                    self.visa_normal_img_path[row[0]].append(row[3].split('/')[-1])
        for i in VISA_CLASS_NAMES:
            self.visa_normal_img_path[i] = self.visa_normal_img_path[i][round_index * 4:]
            self.visa_references[i] = [ os.path.join(f'./data/TrainADDataset/1cls/{i}/train/good', p) for p in self.visa_normal_img_path[i] ] # 现在VisA的reference路径是真实路径惹。

        NVTEC_CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid','hazelnut', 'leather', 'metal_nut', 'pill', 'screw','tile', 'toothbrush', 'transistor', 'wood', 'zipper']
        self.mvtec_normal_img_path = {}
        self.mvtec_references = {}
        for class_name in NVTEC_CLASS_NAMES:
            self.mvtec_normal_img_path[class_name] = []
        for c_name in NVTEC_CLASS_NAMES:
            self.mvtec_normal_img_path[c_name] = [str(round_index * 4).zfill(3)+".png", str(round_index * 4 + 1).zfill(3)+".png",
                                str(round_index * 4 + 2).zfill(3)+".png", str(round_index * 4 + 3).zfill(3)+".png"]

            self.mvtec_normal_img_path[c_name] = self.mvtec_normal_img_path[c_name][:k_shot]
            self.mvtec_references[c_name] = [ os.path.join(f'./data/TrainADDataset/mvtec/{c_name}/train/good', p) for p in self.mvtec_normal_img_path[c_name] ]  # 现在MVTec的reference路径是真实路径惹。
        
        self._ref_bank = {}

    def rot90_img(self,x,k):
        # k is 0,1,2,3
        degreesarr = [0., 90., 180., 270., 360]
        degrees = torch.tensor(degreesarr[k]).to(torch.float16).to("cuda")
        x = K.geometry.transform.rotate(x, angle = degrees, padding_mode='reflection')
        return x

    def encode_image_for_one_shot_with_aug(self, image_paths):
        image_tensors = data.load_and_transform_vision_data(image_paths, "cuda").to(torch.float16)
        B,C,H,W = image_tensors.shape
        # print(B,C,H,W)

        rotated_images = torch.zeros((4, B, C, H, W)).to(torch.float16).to("cuda")


        for j, degree in enumerate([0, 1, 2, 3]):
            rotated_img = self.rot90_img(image_tensors, degree)
            # 存储旋转后的图像
            rotated_images[j] = rotated_img

        image_tensors = rotated_images.transpose(0,1).reshape(B * 4, C, H, W)

        inputs = {ModalityType.VISION: image_tensors}
        # convert into visual dtype
        inputs = {key: inputs[key] for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            patch_features = embeddings['vision'][1] # bsz x h*w x 1280
            for i in range(len(patch_features)):
                patch_features[i] = patch_features[i].transpose(0, 1)[:, 1:, :].reshape(B,4,256,1280).reshape(B, 4 * 256, 1280)

        return patch_features
    
    def encode_image_from_tensor(self, image_tensors):
        if not isinstance(image_tensors, list):
            image_tensors = [image_tensors]
        inputs = {ModalityType.VISION: torch.stack(image_tensors, dim=0).to(image_tensors[0].device)}
        inputs = {key: inputs[key].to(torch.float16) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['vision'][0] # bsz x 1024
            patch_features = embeddings['vision'][1] # bsz x h*w x 1024
        patch_tokens = self.image_decoder(patch_features)
        return patch_tokens

    def encode_image_for_one_shot(self, image_paths):
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths, "cuda")}
        inputs = {key: inputs[key].to(torch.float16) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            patch_features = embeddings['vision'][1] # bsz x h*w x 1280
            for i in range(len(patch_features)):
                patch_features[i] = patch_features[i].transpose(0, 1)[:, 1:, :]
        return patch_features

    def encode_image_for_one_shot_from_tensor(self, image_tensors):
        if not isinstance(image_tensors, list):
            image_tensors = [image_tensors]
        inputs = {ModalityType.VISION: torch.stack(image_tensors, dim=0).to(image_tensors[0].device)}
        inputs = {key: inputs[key].to(torch.float16) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            patch_features = embeddings['vision'][1] # bsz x h*w x 1280
            for i in range(len(patch_features)):
                patch_features[i] = patch_features[i].transpose(0, 1)[:, 1:, :]

        return patch_features
    
    # def get_img_features(self, c_name):
    #     if c_name not in self._img_bank:
    #         tmp = data.load_and_transform_vision_data(image_paths, "cuda")
    #         self._img_bank[c_name] = [tmp[ind] for ind in range(tmp.shape[0])]
        
    #     return [ item.detach() for item in self._img_bank[c_name] ]

    # def encode_reference_by_scenes(self, cls_names):
    #     ret = []
    #     for c_name in cls_names:
    #         ret.extend(self.get_img_features(c_name))
    #     return self.encode_image_for_one_shot_from_tensor(ret)

    def forward(self, images,cls_names,return_masks=True,querypath=None,testphase=False):  
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            if querypath:  #### 
                normal_paths = []
                for cls_name in cls_names:
                    if cls_name in self.visa_references:
                        normal_paths.extend(self.visa_references[cls_name])
                    else:
                        normal_paths.extend(self.mvtec_references[cls_name])

                if testphase ==True:
                    query_patch_tokens = self.encode_image_for_one_shot_from_tensor(images) 
                    normal_patch_tokens = self.encode_image_for_one_shot(normal_paths)

                else:
                    query_patch_tokens = self.encode_image_for_one_shot_from_tensor(images) 
                    normal_patch_tokens = self.encode_image_for_one_shot(normal_paths)


                sims = []
                B = images.shape[0]

                for i in range(len(query_patch_tokens)):
                    query_patch_tokens_reshaped = query_patch_tokens[i].view(B,256,1,1280)
                    normal_tokens_reshaped = normal_patch_tokens[i].reshape(B,1,-1,1280)
                    cosine_similarity_matrix = F.cosine_similarity(query_patch_tokens_reshaped, normal_tokens_reshaped, dim=-1)
                    sim_max, _ = torch.max(cosine_similarity_matrix, dim=-1)
                    sims.append(sim_max)

                sim = torch.mean(torch.stack(sims,dim=0), dim=0).reshape(B,1,16,16)
                simmask = 1-sim
                sim = F.interpolate(sim,size=224, mode='bilinear', align_corners=True)
                anomaly_map_all = 1 - sim 
                return anomaly_map_all,simmask
            else:
                patch_tokens = self.encode_image_from_tensor(images)
                class_name = cls_names
                feats_text_tensor = encode_text_with_prompt_ensemble(self.visual_encoder, [i for i in class_name], patch_tokens[0].device)
                anomaly_maps = []
                anomaly_masks = []
                for layer in range(len(patch_tokens)):
                    patch_tokens[layer] = patch_tokens[layer] / patch_tokens[layer].norm(dim=-1, keepdim=True)
                    anomaly_map = (100.0 * patch_tokens[layer] @ feats_text_tensor.transpose(-2,-1))
                    B, L, C = anomaly_map.shape
                    H = int(np.sqrt(L))
                    anomaly_mask = anomaly_map.permute(0, 2, 1).view(B, 2, H, H)
                    anomaly_mask = torch.softmax(anomaly_mask, dim=1)
                    anomaly_masks.append(anomaly_mask[:,1:,:,:])   
                    anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                                size=224, mode='bilinear', align_corners=True)
                    anomaly_map = torch.softmax(anomaly_map, dim=1)
                    anomaly_maps.append(anomaly_map[:,1:,:,:])        


                output_maps = torch.mean(torch.stack(anomaly_maps),0)
                output_masks = torch.mean(torch.stack(anomaly_masks),0)
                return output_maps,output_masks
        