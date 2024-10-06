import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from torch.nn import functional as F

from minigpt4.common.registry import registry
from minigpt4.models.blip2 import Blip2Base, disabled_train
from minigpt4.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer, GenerationConfig
from minigpt4.models.networks import VETokenizer,VEInstructorV2,LoraAdaptorV2

from minigpt4.models.adexpert import adexpert
from minigpt4.models.adrefexpert_v2 import adrefexpert

# For LoRA
from peft import get_peft_model, LoraConfig, TaskType

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import rnn
import torch.nn.functional as F
from math import exp
from torch.nn import init
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        print(m)
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

@registry.register_model("myriad")
class Myriad(Blip2Base):
    """
    BLIP2 GPT-LLAMA model For Class-Agnostic Anomaly detection with Self-Supervision Style.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/minigpt4.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        freeze_llama=True, 
        use_lora=False,
        bliva_like=False, 
        round_index=0,
        k_shot=0, 
        use_ve=False, 
        use_ref=False,
        do_random=False, 
        adaptor_type='none', 
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    ):
        super().__init__()
        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource

        # Train参数
        self.do_random = do_random

        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        self.use_ve = use_ve
        self.use_ref = use_ref
        # if self.use_ve:
        self.round_index = round_index
        self.k_shot = k_shot

        self.expert_adaptor = LoraAdaptorV2(dims=1408,input_dim=4)

        self.vision_expert = adrefexpert(round_index=self.round_index,k_shot=self.k_shot)  # adrefexpert() # adexpert
    
        for name, param in self.vision_expert.named_parameters():
            param.requires_grad = False
        self.vision_expert.eval()
        self.VETokenizer = VETokenizer()
        self.VEInstructor = VEInstructorV2()

        # BLIVA like
        self.bliva_like = bliva_like
        if self.bliva_like:
            self.bliva_fc = nn.Linear(
                1408, 5120
            )

        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False

            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
                
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze Qformer")
        print('Loading Q-Former Done')

        # LoRA
        if use_lora:
            self.lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                r=8, 
                lora_alpha=16, 
                lora_dropout=0.05, 
                bias="none",
                target_modules=["q_proj", "v_proj"]  # follow LISA default settings.
            )
        else:
            self.lora_config = None

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )
            if self.lora_config is not None:
                self.llama_model = get_peft_model(self.llama_model, self.lora_config)
                self.llama_model.print_trainable_parameters()
        
        self.freeze_llama = freeze_llama
        if self.freeze_llama and (self.lora_config is None):  
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
        print('Loading LLAMA Done')
        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        imagebind_ckpt = torch.load('./pretrained_models/pretrained_minigpt4_7b.pth', map_location=torch.device('cpu'))
        weightpth = imagebind_ckpt['model']['llama_proj.weight']
        biaspth = imagebind_ckpt['model']['llama_proj.bias']
        new_state_dict = self.llama_proj.state_dict()
        new_state_dict['weight'] = weightpth
        new_state_dict['bias'] = biaspth

        self.llama_proj.load_state_dict(new_state_dict)
        for name, param in self.llama_proj.named_parameters():
            param.requires_grad = False

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('self.prompt_list',self.prompt_list)
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []
        
    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def encode_img(self, image, maps,stage):
        device = image.device
        b =  image.shape[0]
        with self.maybe_autocast():
            image_embeds = self.visual_encoder(image)
            if self.bliva_like: bliva_features = self.bliva_fc(image_embeds)

            image_embeds = self.ln_vision(self.expert_adaptor(image_embeds)).to(device)

            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            if stage==1 or stage==2:
                queryprompter = self.VEInstructor(maps)
                query_tokens = torch.cat([query_tokens,queryprompter],1)

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_llama = self.llama_proj(query_output.last_hidden_state)
            B = image.shape[0]
            if stage==0 or stage==1:
                maskprompter = self.VETokenizer(maps)
                inputs_llama = torch.cat([inputs_llama,maskprompter],1)

            if self.bliva_like: inputs_llama = torch.cat([inputs_llama, bliva_features], dim=1)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)

        return inputs_llama, atts_llama
    def encode_img_oneshot(self, image, oneshotmaps,stage):
        device = image.device
        b =  image.shape[0]
        with self.maybe_autocast():
            image_embeds = self.visual_encoder(image)
            if self.bliva_like: bliva_features = self.bliva_fc(image_embeds)

            image_embeds = self.ln_vision(self.expert_adaptor(image_embeds)).to(device)

            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

            if stage==1 or stage==2:
                queryprompter = self.VEInstructor(oneshotmaps)
                query_tokens = torch.cat([query_tokens,queryprompter],1)

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_llama = self.llama_proj(query_output.last_hidden_state)
            B = image.shape[0]
            if stage==0 or stage==1:

                maskprompter = self.VETokenizer(oneshotmaps)
                inputs_llama = torch.cat([inputs_llama,maskprompter],1)

            if self.bliva_like: inputs_llama = torch.cat([inputs_llama, bliva_features], dim=1)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)

        return inputs_llama, atts_llama

    def embed_tokens(self, *args, **kwargs):
        if self.lora_config is not None:
            return self.llama_model.model.model.embed_tokens(*args, **kwargs)
        return self.llama_model.model.embed_tokens(*args, **kwargs)
    
    def prepare_sample(self, samples,stage):
        image = samples["image"]
        if 'aug_image' in samples and self.training:
            image = torch.cat([image, samples['aug_image']])
        if stage==0:
            questions = samples.get("question", None)
        elif stage==1:
            questions = samples.get("question2", None)
        elif stage==2:
            questions = samples.get("question3", None)

        if self.training:
            if 'aug_text_input' in samples:
                text_inputs = samples['text_input'] + samples['aug_text_input']  # 测试的时候没有text_input 
            else:
                text_inputs = samples['text_input']
        else:
            text_inputs = None
        
        b = image.shape[0]
        with torch.no_grad():
            scenes = samples['scene']
            if self.training:
                if 'aug_image' in samples:
                    scenes = scenes + scenes
                    refpaths = samples['img_path']+samples['img_path']
                else:
                    scenes = scenes
                    refpaths = samples['img_path']
                anomaly_maps,_ = self.vision_expert(image, scenes)
                oneshot_anomaly_maps,_ = self.vision_expert(image, scenes,querypath=refpaths)

            else:
                refpaths = samples['img_path']
                anomaly_maps,_ = self.vision_expert(image, scenes)
                oneshot_anomaly_maps,_ = self.vision_expert(image, scenes,querypath=refpaths,testphase=True)


        # print(oneshot_anomaly_maps.shape,anomaly_maps.shape)
        return image, questions, text_inputs, anomaly_maps, oneshot_anomaly_maps

    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            p_befores = []
            p_afters = []
            for i in range(len(prompt)):

                p_before, p_after = prompt[i].split('<ImageHere>')
                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_before_embeds = self.embed_tokens(p_before_tokens.input_ids)
                p_after_embeds = self.embed_tokens(p_after_tokens.input_ids)

                p_befores.append(p_before_embeds)
                p_afters.append(p_after_embeds)
            wrapped_img_embeds = torch.cat([torch.stack(p_befores)[:,0,:,:], img_embeds, torch.stack(p_afters)[:,0,:,:]], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    def forward(self, samples):
        promptstagenum = random.choice([0,1,2])

        image, questions, text_inputs, maps,onemaps = self.prepare_sample(samples,promptstagenum)
        taskstage = random.choice([0,1])

        if taskstage==0:
            img_embeds, atts_img = self.encode_img(image, maps,promptstagenum)
        else:
            img_embeds, atts_img = self.encode_img_oneshot(image, onemaps,promptstagenum)


        prompt = ['###Human: ' + q + ' ###Assistant: ' for q in questions]
        if self.training and ('aug_image' in samples):
            prompt = prompt + prompt
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)
        # img_embeds_oneshot, atts_img_oneshot = self.prompt_wrap(img_embeds_oneshot, atts_img_oneshot, prompt)

        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in text_inputs]
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(image.device)
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos begin of sequuese
        )
        targets = torch.cat([empty_targets, targets], dim=1)
        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]
        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        loss = outputs.loss
        return {"loss": loss}

    @torch.no_grad()
    def generate(self, samples, **generate_kwargs):
        stagenum = 1
        image, questions, text_inputs, maps, refs = self.prepare_sample(samples,stagenum)
        if self.k_shot>0:
            savemaps = refs
            img_embeds, atts_img = self.encode_img_oneshot(image, refs,stagenum)
        else:
            savemaps = maps
            img_embeds, atts_img = self.encode_img(image, maps,stagenum)

        questions = ['###Human: ' + q + ' ###Assistant: ' for q in questions]
        inputs_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, questions)
        
        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            **generate_kwargs
        )
        return {
           "token_ids": outputs, 
           "ve_anomaly_maps": savemaps,
        }

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        freeze_llama = cfg.get("freeze_llama", True)
        use_lora = cfg.get("use_lora", False)
        bliva_like = cfg.get("bliva_like", False)  # 像是Bliva那样，直接把CLIP特征+FC也跟着输入到llama里。
        use_ve = cfg.get("use_ve", False)
        use_ref = cfg.get("use_ref", False)
        noise_level = cfg.get("noise_level", 0.15)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)
        round_index = cfg.get("round_index", 0)
        k_shot = cfg.get("k_shot", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            freeze_llama=freeze_llama,
            use_lora=use_lora,
            bliva_like=bliva_like, 
            round_index=round_index,
            k_shot=k_shot,
            use_ve=use_ve, 
            use_ref=use_ref,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model
