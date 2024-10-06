# from minigpt4.models.header import *
# import torch.nn.functional as F
# from minigpt4.models.model.ImageBind import data, ModalityType, imagebind_model
# import torch
# from torch import nn
# import numpy as np


from minigpt4.models.header import *
import torch.nn.functional as F
from minigpt4.models.model.ImageBind import *
from minigpt4.models.model.ImageBind import data
from minigpt4.models.model.modeling_llama import LlamaForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
import kornia as K
# from minigpt4.models.modeling_llama import LlamaForCausalLM
import torch
from torch.nn.utils import rnn


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
# prompt_templates = [
#                         'a cropped photo of the {}.', 'a cropped photo of a {}.', 'a close-up photo of a {}.', 'a close-up photo of the {}.',
#                         'a bright photo of the {}.', 'a bright photo of a {}.', 'a dark photo of a {}.', 'a dark photo of the {}.',
#                         'a dark photo of the {}.', 'a dark photo of a {}.', 'a jpeg corrupted photo of a {}.', 'a jpeg corrupted photo of the {}.',
#                         'a blurry photo of the {}.', 'a blurry photo of a {}.', 'a photo of a {}.', 'a photo of the {}.',
#                         'a photo of the small {}.', 'a photo of a small {}.', 'a photo of the large {}.', 'a photo of a large {}.',
#                         'a photo of the {} for visual insprction.', 'a photo of a {} for visual insprction.',
#                         'a photo of the {} for anomaly detection.', 'a photo of a {} for anomaly detection.'
#                         ]
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


class adexpert(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        imagebind_ckpt_path = './pretrained_models/anomalyGPT_ckpt/imagebind_huge.pth'
        # vicuna_ckpt_path = args['vicuna_ckpt_path']
        # max_tgt_len = args['max_tgt_len']
        # stage = args['stage']
        args = {'model': 'openllama_peft', 'imagebind_ckpt_path': './pretrained_models/anomalyGPT_ckpt/imagebind_huge.pth', 'vicuna_ckpt_path': './pretrained_models/vicuna-7b-v0', 'anomalygpt_ckpt_path': '/mnt/vdb1/whl/AnomalyGPT-old/code/ckpt/pytorch_mvtec_model.pt', 'delta_ckpt_path': '/mnt/vdb1/whl/AnomalyGPT-old/pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt', 'stage': 2, 'max_tgt_len': 128, 'lora_r': 32, 'lora_alpha': 32, 'lora_dropout': 0.1}
        self.visual_encoder, self.visual_hidden_size = imagebind_model.imagebind_huge(args)  # 这里的args只用到了'layers'这个key。也就是具体输出哪些层。
        imagebind_ckpt = torch.load(imagebind_ckpt_path, map_location=torch.device('cpu'))
        self.visual_encoder.load_state_dict(imagebind_ckpt, strict=True)
        self.image_decoder = LinearLayer(1280, 1024, 4)

        # self.prompt_learner = PromptLearner(1, 4096)
        save_model = torch.load('./pretrained_models/anomalyGPT_ckpt/pytorch_mvtec_model.pt', map_location=torch.device('cpu'))
        # print(decoder_ckpt)
        # print(new_state_dict)

        new_state_dict = self.image_decoder.state_dict()
        # print(new_state_dict)
        # 去掉开头的image_decoder, 其实可以直接使用self.load_state_dict()，这样就不用改变state_dict了。
        for name, param in save_model.items():
            if 'image_decoder.' in name:
                name_ = '.'.join(name.split('.')[1:])
                if name_ in new_state_dict:
                    # print('aaaaaaaaaaaaaaaaa',name_,name)
                    new_state_dict[name_] = param
        # print(new_state_dict)
        # TODO: @whl 为什么这里没有eval?
        self.image_decoder.load_state_dict(new_state_dict)
        for name, param in self.image_decoder.named_parameters():
            param.requires_grad = False

        # free vision encoder
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder.eval()

    def encode_image_from_tensor(self, image_tensors):
        if not isinstance(image_tensors, list):
            image_tensors = [image_tensors]
        inputs = {ModalityType.VISION: torch.stack(image_tensors, dim=0).to(image_tensors[0].device)}
        # convert into visual dtype
        # print('-----------------------------------',self.llama_model.dtype)
        inputs = {key: inputs[key].to(torch.float32) for key in inputs}
        # print("Inputs:", inputs[ModalityType.VISION][0])
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['vision'][0] # bsz x 1024
            patch_features = embeddings['vision'][1] # bsz x h*w x 1024
            # print("Patch features:", patch_features[0])
            # print(patch_features[0].max())
            # print(patch_features[0].max())
            # print(patch_features[0].max())

        patch_tokens = self.image_decoder(patch_features)
        # print("Patch tokens:", patch_tokens[0])
        return patch_tokens
    
    def forward(self, images, cls_names, return_masks=True):  # 为了不打扰whl的代码，这里默认值为True

        # image_paths = inputs['images']
        patch_tokens = self.encode_image_from_tensor(images)
        # print("Patch tokens:", patch_tokens[0])
        class_name = cls_names
        # print(class_name)
        loss_pixel = 0
        # feats_text_tensor = encode_text_with_prompt_ensemble(self.visual_encoder, ['object' for _ in class_name], patch_tokens[0].device)
        feats_text_tensor = encode_text_with_prompt_ensemble(self.visual_encoder, [i for i in class_name], patch_tokens[0].device)
        # print("text features:", feats_text_tensor[0])

        # print(feats_text_tensor)
        anomaly_maps = []
        anomaly_masks = []
        for layer in range(len(patch_tokens)):
            # print( patch_tokens[layer].max(),feats_text_tensor.max())
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

        # print(output_maps.max().item(),output_masks.min().item())
        # output_maps = output_maps / 100
        # output_maps = output_maps + 1e-6
        if return_masks:
            return output_maps, output_masks
        return output_maps
