"""
    这个文件用于进行带mask和reference的模型的evaluation.
    v2: 加入了confident的支持。
"""
import argparse
import os
import random

import numpy as np
import torch
import time
import json
import jsonlines
import torch.backends.cudnn as cudnn
import csv
from tqdm import tqdm
from PIL import Image
import cv2
from torch.utils.data import DataLoader
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import StoppingCriteriaSub
from minigpt4.datasets.datasets.anomaly_detection import AnomalyDetectionDataset
from transformers import StoppingCriteria, StoppingCriteriaList
from datetime import datetime

# imports modules for registration
from minigpt4.processors.transform import Expand2square
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
import minigpt4.tasks as tasks
from minigpt4.common.utils import disable_torch_init
from minigpt4.datasets.data_utils import prepare_sample



def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--task_type", type=str, default='aqa', choices=['aqa', 'roi', 'al', '1cls', 'adroi', 'shot'])
    parser.add_argument("--split", type=str, default='eval', choices=['eval', 'test', 'train','eval_un','eval_fewshot','visa','mvtec'])
    parser.add_argument("--ckpt", type=int, default=-1)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--round_index", type=int, default=14)
    parser.add_argument("--k_shot", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)

    
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_context_emb(img_embeds, questions, model):
    batch_size = img_embeds.shape[0]
    texts = questions.split('<ImageHere>')
    assert img_embeds.shape[1] == len(texts) - 1
    text_embeds = []
    for ind, text in enumerate(texts):
        text_token = model.llama_tokenizer(
            text, return_tensors="pt", add_special_tokens=(ind==0)).to(img_embeds.device)
        text_embed = model.llama_model.model.embed_tokens(text_token.input_ids).expand(batch_size, -1, -1)
        text_embeds.append(text_embed)
        # print('text:', text_embed.shape)
        if ind < img_embeds.shape[1]:
            # print('roi:', img_embeds[:, ind].shape)
            text_embeds.append(img_embeds[:, ind])
    q_embed = torch.cat(text_embeds, dim=1)
    return q_embed


def anomaly_map_handler(output):
    if 've_anomaly_maps' not in output:
        return []
    anomaly_maps = output['ve_anomaly_maps']
    bs = anomaly_maps.shape[0]
    anomaly_map_list = []
    for ind in range(bs):
        _tmp = anomaly_maps[ind].detach().cpu().expand(3, -1, -1)
        _tmp = _tmp.permute(1, 2, 0).numpy() * 255.
        anomaly_map_list.append(_tmp.astype(np.uint8))

    return anomaly_map_list


ROOTS = {
    'train': "./data/TrainADDataset", 
    'eval': "./data/EvalADDataset", 
    'test': "./data/EvalADDataset",
    'visa': "./data/EvalADDataset", 
    'mvtec': "./data/EvalADDataset", 
}

ANNO_FILES = {
    'aqa': {
        'train': 'AQA_train.jsonl',
        'test': 'AQA_test.jsonl',
        'eval': 'AQA_eval.jsonl'
    }, 
    'roi': {
        'train': 'AQA_train.jsonl',
        'test': 'AQA_test.jsonl',
        'eval': 'AQA_eval.jsonl'
    }, 
    'al': {
        'eval': 'val_coco.json'
    }, 
    'ad': {
        'eval': 'DC_MVTEC_test_normal.jsonl',
        # 'eval': 'DC_VISA_test_normal.jsonl',
    }, 
    'ad_few': {
        'eval': 'DC_VISA_test_normal.jsonl',
        # 'eval': 'DC_VISA_test_normal.jsonl',
    }, 
    'adroi': {
        'eval': 'DC_MVTEC_test_normal.jsonl'
    }, 
    '1cls': {
        'visa': 'DC_VISA_test_normal.jsonl',
        'mvtec': 'DC_MVTEC_test_normal.jsonl'
    }, 
    'shot': {
        'visa': 'DC_VISA_test_normal.jsonl',
        'mvtec': 'DC_MVTEC_test_normal.jsonl'
    }, 
    
}


def build_dataset(args, ds_cfg):
    if args.task_type in ['aqa']:
        return AQADataset(
            LocImageTrainProcessor(identity=True), 
            None, 
            ROOTS[args.split], 
            ann_paths=[os.path.join(ROOTS[args.split], ANNO_FILES[args.task_type][args.split])], 
            add_noa=True,
            with_ve=ds_cfg[args.task_type].get('with_ve', False), 
            stage='test'
        )
    elif args.task_type in ['roi']:
        return ROIDataset(
            LocImageTrainProcessor(identity=True), 
            None, 
            ROOTS[args.split], 
            ann_paths=[os.path.join(ROOTS[args.split], ANNO_FILES[args.task_type][args.split])], 
            with_ve=ds_cfg[args.task_type].get('with_ve', False), 
            stage='test'
        )
    elif args.task_type in ['al']:
        return AlignDataset(
            LocImageTrainProcessor(identity=True),
            None,
            ROOTS[args.split],
            ve_root="/mnt/vdb1/datasets/aprilgan_processresults", 
            ann_paths=[os.path.join(ROOTS[args.split], ANNO_FILES[args.task_type][args.split])],
            with_mask=ds_cfg['align'].get('with_mask', False), 
            with_gt_seg=ds_cfg['align'].get('with_gt_seg', False), 
        )
    elif args.task_type in ['ad']:
        return AnomalyDetectionDataset(
            LocImageTrainProcessor(identity=True),
            None,
            ROOTS[args.split],
            ve_root="/mnt/vdb1/datasets/aprilgan_processresults", 
            ann_paths=[os.path.join(ROOTS[args.split], ANNO_FILES[args.task_type][args.split])],
            with_mask=ds_cfg['anomaly_detection'].get('with_mask', False), 
            is_preload=True, 
            stage='test'
        )
    elif args.task_type in ['ad_few']:
        return AnomalyDetectionDataset(
            LocImageTrainProcessor(identity=True),
            None,
            ROOTS[args.split],
            ve_root="/mnt/vdb1/datasets/aprilgan_processresults", 
            ann_paths=[os.path.join(ROOTS[args.split], ANNO_FILES[args.task_type][args.split])],
            with_mask=ds_cfg['anomaly_detection'].get('with_mask', False), 
            is_preload=True, 
            stage='test'
        )
    elif args.task_type in ['1cls','shot']:
        return AnomalyDetectionDataset(
            LocImageTrainProcessor(identity=True),
            None,
            ROOTS[args.split],
            ve_root="/mnt/vdb1/datasets/aprilgan_processresults", 
            ann_paths=[ANNO_FILES[args.task_type][args.split]],
            img_size=ds_cfg['anomaly_detection'].get("img_size", 224), 
            crop_size=ds_cfg['anomaly_detection'].get("crop_size", 224), 
            with_mask=ds_cfg['anomaly_detection'].get('with_mask', False), 
            is_preload=True, 
            stage='test'
        )
    elif args.task_type in ['adroi']:
        print("Num RoIs:", ds_cfg['adroi'].get("num_rois", 3))
        return ADRoIDataset(
            LocImageTrainProcessor(identity=True),
            None,
            ROOTS[args.split],
            ve_root="/mnt/vdb1/datasets/aprilgan_processresults", 
            ann_paths=[os.path.join(ROOTS[args.split], ANNO_FILES[args.task_type][args.split])],
            is_preload=True, 
            num_rois=ds_cfg['adroi'].get("num_rois", 3), 
            stage='test'
        )
    else:
        raise NotImplementedError(f"Not implement for task type {args.task_type}")


if __name__ == '__main__':
    # ========================================
    #             Model Initialization
    # ========================================
    print('Initializing Chat')
    args = parse_args()
    cfg = Config(args)

    disable_torch_init()

    model_config = cfg.model_cfg
    round_index = args.round_index
    model_config.round_index = round_index
    k_shot = args.k_shot
    model_config.k_shot = k_shot
    if args.ckpt != -1:
        old_ckpt = model_config.ckpt
        ckpt_path = old_ckpt.split('/')
        ckpt_path[-1] = f'checkpoint_{args.ckpt}.pth'
        model_config.ckpt = '/'.join(ckpt_path)
    #
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    # print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',model_config)
    # print(model_config{'round_index'})
    

    # print('round_indexround_indexround_indexround_indexround_indexround_indexround_indexround_indexround_indexround_indexround_indexround_indexround_index',round_index)
    # print(model_config['round_index'])
    # round_index = model_config.round_index
    # ========================================
    #             Data Preparation
    # ========================================
    system="Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions."
    stop_words_ids = [torch.tensor([835]).cuda(),
                          torch.tensor([2277, 29937]).cuda()]  # '###' can be encoded in two different ways.
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    dataloader = DataLoader(
        build_dataset(args, cfg.datasets_cfg),
        batch_size=1 if args.task_type == 'al' else args.bs,
        num_workers=4,
        pin_memory=True,
    )

    # 保存路径
    ckpt_name, _ = os.path.splitext(os.path.basename(model_config.ckpt))
    num_ckpt = int(ckpt_name.split('_')[-1])
    prefix = f"results_ckpt{num_ckpt}_training={args.task_type}_split={args.split}_kshot={k_shot}_roundindex={round_index}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    save_path = os.path.join('/', *model_config.ckpt.split('/')[1:-1], f"{prefix}.jsonl")
    # if model_config.get("use_ve", False):
        # anomaly_map_dir = os.path.join(os.path.join('/', *model_config.ckpt.split('/')[1:-1],'result', prefix))
        # os.makedirs(anomaly_map_dir) 
        # print(f"Anomaly Maps will be saved to {anomaly_map_dir}")
    print(f"Results will be saved to {save_path}")
    generate_kwargs = {
        'max_new_tokens': 90,
        'stopping_criteria': stopping_criteria,
        # 'num_beams': 1,
        'do_sample': True,
        'use_cache':True,
        'min_length': 1,
        # 'top_p': 0.9,
        'top_p': 0.01,
        # 'repetition_penalty': 1.0,
        # 'length_penalty': 1,
        'temperature': 1.0,
    }
    
    model.eval()
    results = []
    use_global_feat = model_config.get("use_global_feat", False)
    append_global_feat = model_config.get("append_global_feat", False)
    get_mem=lambda x:max(x,sum([torch.cuda.memory_allocated(i) for i in range(1)]))
    max_mem = 0.0
    all_time = 0.0
    with jsonlines.open(save_path, 'w') as writer:
        for testid, data_sample in tqdm(enumerate(iter(dataloader)), total=len(dataloader), desc='Evaluation AQA Task:'):
            if testid >=args.start:
                with torch.no_grad():
                    samples = prepare_sample(data_sample)
                    batch_size = samples['image'].shape[0]
                    # 这里应该不加上###Human: ... ###Assistant:，这个应该是model的工作
                    # if args.task_type == 'aqa':
                    #     samples['question'] = ["Which part of images has defects? (A) <Img><ImageHere></Img> (B) <Img><ImageHere></Img> (C) <Img><ImageHere></Img> (D) all parts are normal. "]
                    # elif args.task_type in ['roi', 'ad', 'adroi','ad_few']:
                    #     pass
                    # elif args.task_type == 'al':
                    #     samples['question'] = [
                    #         "###Human: <Img><ImageHere></Img> Describe this image in detail and find out anomaly defects if they exist. ###Assistant: "
                    #     ] * batch_size
                    # else:
                    #     raise NotImplementedError(f"Not implement for task type {args.task_type}")
                    max_mem = get_mem(max_mem)
                    t1 = time.time()
                    outputs = model.generate(samples, **generate_kwargs)
                    t2 = time.time()
                    max_mem = get_mem(max_mem)
                    all_time += (t2 - t1)
                    if isinstance(outputs, dict):
                        token_ids = outputs['token_ids']
                        anomaly_maps = anomaly_map_handler(outputs)
                    else:
                        token_ids = outputs

                    token_ids = torch.clamp(token_ids,1,40000)
                    output_text = model.llama_tokenizer.batch_decode(token_ids, add_special_tokens=False)

                    for ind, text in enumerate(output_text):
                        if args.task_type in ['aqa']:
                            item = {
                                'image_id': data_sample['image_id'][ind].item(),
                                'output': text.split("###")[0],
                                'question': samples['question'] if len(samples['question']) == 1 else samples['question'][ind], 
                                'options': samples['options'][ind].detach().cpu().numpy().tolist(),
                                'answer': data_sample['answer'][ind].item(), # gt.
                                'is_anomaly': data_sample['is_anomaly'][ind].item(),  # gt
                            }
                        elif args.task_type in ['roi']:
                            item = {
                                'image_id': data_sample['image_id'][ind].item(),
                                'output': text.split("###")[0],
                                'question': samples['question'] if len(samples['question']) == 1 else samples['question'][ind], 
                                'options': samples['options'][ind].detach().cpu().numpy().tolist(),
                                # 'answer': data_sample['answer'][ind].item(), # gt.
                                'is_anomaly': data_sample['is_anomaly'][ind].item(),  # gt
                            }
                        elif args.task_type in ['al', 'ad', 'adroi', 'ad_few','1cls','shot']:
                            imagepath = samples['img_path'][ind]
                            item = {
                                'image_id': data_sample['image_id'][ind].item(),
                                'image_path': '/'.join(imagepath.split('/')[-5:]),

                                # 'origin_output': model.llama_tokenizer.decode(token_ids[ind], add_special_tokens=False),  # 说明batch_decode和decode没有本质区别。
                                # 'question': samples['question'] if len(samples['question']) == 1 else samples['question'][ind], 
                                'is_anomaly': data_sample['is_anomaly'][ind].item(),  # gt
                            }
                            if anomaly_maps is not None:
                                # anomaly_map_path = os.path.join(anomaly_map_dir, f"{data_sample['image_id'][ind].item():04d}.png")
                                # os.makedirs(anomaly_map_dir+'/'.join(samples['img_path'][ind].split('/')[-5:-1]),exist_ok=True)
                                # print(anomaly_map_dir+samples['img_path'][ind],anomaly_map_dir+'/'.join(samples['img_path'][ind].split('/')[-5:-1]))
                                # cv2.imwrite(anomaly_map_dir+'/'.join(samples['img_path'][ind].split('/')[-5:]), anomaly_maps[ind])
                                if 'Yes' in text.split("###")[0] and data_sample['is_anomaly'][ind]==True:
                                    item['error'] = '0'
                                elif 'No' in text.split("###")[0] and data_sample['is_anomaly'][ind].item()==False:
                                    item['error'] = '0'
                                else:
                                    item['error'] = '1'
                                item['output'] = text.split("###")[0]

                                item['anomaly_score'] = str(round((anomaly_maps[ind].max())/255.0,4))
                        else:
                            raise NotImplementedError(f"Not implement for task type {args.task_type}")
                        writer.write(item)
    print("CUDA Memory:", torch.cuda.max_memory_allocated() / (1024 * 1024))
    print("Max Memory: ", max_mem / (1024 * 1024))
    print("Mean Time: ", all_time / len(dataloader))
