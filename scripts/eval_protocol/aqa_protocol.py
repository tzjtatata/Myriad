"""
    进行AQA任务答案的检查。
    输入: *.jsonl文件
    格式: {
        image_id,
        width,
        height,
        img_path(relative),
        output(text),
        question,
        answer,
        is_anomaly
    }
"""
import os
import sys
import math
import numpy as np
import argparse
import jsonlines
from sklearn.metrics import precision_score, accuracy_score, recall_score, roc_auc_score, confusion_matrix
# import torch
import cv2
from PIL import Image
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from minigpt4.visual_utils.visualize_tools import apply_ad_scoremap, draw_box


ANSWER_MAP = {
    0: "<A>",
    1: "<B>",
    2: "<C>",
    3: "<D>"
}


def get_model_answer(text, mode=0):
    if mode == 0:
        for k, v in ANSWER_MAP.items():
            if v in text:
                return k
        return -1
    elif mode == 1:
        for k, v in enumerate(['A', 'B', 'C', 'D']):
            if v in text.split(':')[-1]:
                return k
        return -1
    else:
        raise NotImplementedError(f"Not Implement Mode {mode}")


def cal_anomaly_scores(preds, split):
    from tqdm import tqdm
    root = "/mnt/vdb1/datasets/EvalADDataset"
    ve_root = "/mnt/vdb1/datasets/aprilgan_processresults"
    vis_root = os.path.join(root, '2cls_highshot')
    anno_path = os.path.join(root, f"AQA_{split}.jsonl")
    print("With Anno file: ", anno_path)

    with jsonlines.open(anno_path, 'r') as reader:
        annos = list(reader)
    
    image_infos = {
        item['id']: item for item in preds
    }

    # 根据annos把gt和ve提取出来
    for ann in tqdm(annos, total=len(annos), desc="Get GT and VE masks"):
        assert ann['image_id'] in image_infos, f"Image with ID {ann['image_id']} not in results."
        if 've' in image_infos[ann['image_id']]: continue # 如果已经处理过了，就跳过

        width, height = ann['width'], ann['height']

        ve_path = ann['ve_path'] if 've_path' in ann else ann['aprilgan_path']  # 兼容两个不同的格式
        # 这里需要做处理，因为原来的数据集里是硬编码的，现在时代变了
        if ve_path.startswith('/mnt'):  # 硬编码
            ve_path = os.path.join(ve_root, *ve_path.split('/')[6:])
        else:  # 软编码
            ve_path = os.path.join(ve_root, ve_path)
        # 检验是否有错误的ve_path；结果：全对
        # print(ve_path, os.path.exists(ve_path))
        # if not os.path.exists(ve_path): 
        #     raise ValueError(u"存在Vision Expert地址错误")
        ve = cv2.imread(ve_path)
        # cv2的resize函数脑子有问题，格式是(width, height)而不是numpy的shape
        ve = cv2.resize(ve, (224, 224), interpolation=cv2.INTER_NEAREST)[:, :, 0] # (height, width), range: (0, 1)
        image_infos[ann['image_id']]['ve'] = ve
        # print(ve.max(), ve.min())  # 确定了值域没问题

        # 获取图像
        img_path = os.path.join(vis_root, ann['img_path'])
        image_infos[ann['image_id']]['img'] = Image.open(img_path).convert('RGB').resize((224, 224), Image.Resampling.BICUBIC)

        # 获取gt mask
        if 'good' in ann['img_path']:  # Normal
            gt = np.zeros((224, 224)).astype(float)
        else:
            prefixes = ann['img_path'].split('/')  # 场景/split/bad/具体图像号.JPG
            gt_path = os.path.join(vis_root, prefixes[0], 'ground_truth', *prefixes[1:])
            gt_path = gt_path[:-3]+'png'  # 和图像不一样，gt是png结尾
            gt = np.array(Image.open(gt_path).convert('L').resize((224, 224), Image.Resampling.NEAREST)) > 0
            gt = gt.astype(float)
        
        image_infos[ann['image_id']]['gt'] = gt
        # print(gt.shape, gt.max(), gt.min())
    
    # 开始计算真正的ve
    px_preds = []
    px_gts = []
    for k, item in tqdm(image_infos.items(), total=len(image_infos), desc="Cal Pixel-wise metrics"):
        # print(k)
        gt = item['gt']
        ve = item['ve']
        defects = item['defects']
        new_img = draw_box(item['img'].copy(), np.array(defects), color=(255, 0, 0))
        new_img = draw_box(new_img, np.array(item['normals']))


        if len(defects) == 0:  # 如果压根没预测出来defects，就返回一个全0的数
            px_preds.append(np.zeros_like(gt).ravel())
        else:  # 如果预测出来了defects，就保留defects的部分。
            pred_ve = np.zeros_like(ve)
            for box in defects:
                # print("box:", box)
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)
                pred_ve[y1:y2, x1:x2] = ve[y1:y2, x1:x2]
            cv2.imwrite(
                os.path.join("/home/lyz/vdb/results/test_dataset/aqa_loc", f'{k}_pred.png'), 
                apply_ad_scoremap(new_img.copy(), pred_ve)
            )
            px_preds.append(pred_ve.ravel())
            cv2.imwrite(
                os.path.join("/home/lyz/vdb/results/test_dataset/aqa_loc", f'{k}_gt.png'), 
                apply_ad_scoremap(new_img.copy(), gt)
            )
        px_gts.append(gt.ravel())
        # print('GT:', px_gts[-1].shape)
        # print('Pred:', px_preds[-1].shape)
    
    # 摊平，并计算AUROC
    px_preds = np.concatenate(px_preds)
    px_gts = np.concatenate(px_gts)
    print(f"Pixel Pred shape: {px_preds.shape}, Pixel GT shape: {px_gts.shape}")
    print("Pixel-AUROC:", roc_auc_score(px_gts, px_preds))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--protocol", type=str, choices=['v1', 'v2'], default='v2')  # v1: 只有选对了选项才认为是abnormal，否则是normal; v2: 不选D就是AbNormal，选D就是Normal
    parser.add_argument("--loc", action='store_true')
    parser.add_argument("--mode", type=int, default=0)
    args = parser.parse_args()
    with jsonlines.open(args.result_path, 'r') as reader:
        records = list(reader)
    split = 'test' if 'test' in args.result_path else 'eval'

    root = "/mnt/vdb1/datasets/EvalADDataset"
    anno_path = os.path.join(root, f"AL_VisA_test.jsonl")
    with jsonlines.open(anno_path, 'r') as reader:
        gt_annos = list(reader)
    image_info = {}
    for item in gt_annos:
        if item['image_id'] in image_info: continue
        image_info[item['image_id']] = {
            "id": item['image_id'],
            "gt": 0 if 'good' in item['img_path'] else 1,
            "pred": [],
            "defects": [],
            'normals': []
        }
    
    preds = []
    gts = []  # 这个是记录normal或者是abnormal的结果
    qa_results = []
    for r in records:
        ans, out_text = r['answer'], r['output']
        pred = get_model_answer(out_text, mode=args.mode)
        # print(out_text, pred, ans)
        gts.append(1 if r['is_anomaly'] else 0)
        if pred == -1:
            print(out_text)
            qa_results.append(pred)
            preds.append(pred)
        else:
            if args.protocol == 'v1':
                # 选对了才是Normal
                if pred == ans:  # 答对了
                    preds.append(1 if r['is_anomaly'] else 0)
                    qa_results.append(1)
                else:  # 答错了
                    preds.append(0 if r['is_anomaly'] else 1)
                    qa_results.append(0)
            else:
                if pred == 3:
                    preds.append(0)
                else:
                    preds.append(1)
                if pred == ans:  # 答对了
                    qa_results.append(1)
                else:  # 答错了
                    qa_results.append(0)

    # print(len(qa_results))
    qa_results_np = np.array(qa_results)
    print(u"预测的不知道是什么东西的数量:", np.sum(qa_results_np == -1))
    print(u"问答的正确数量和正确率:", np.sum(qa_results_np == 1), np.sum(qa_results_np == 1) / (len(records)-np.sum(qa_results_np == -1)))
    print(u"问答的错误数量和错误率:", np.sum(qa_results_np == 0), np.sum(qa_results_np == 0) / (len(records)-np.sum(qa_results_np == -1)))
    abnormal_qa_results = qa_results_np[np.array(gts) == 1]
    print(u"在包含异常的问题中，正确率为:", np.sum(abnormal_qa_results == 1) / abnormal_qa_results.shape[0])
    normal_qa_results = qa_results_np[np.array(gts) == 0]
    print(u"在正常的问题中，正确率为:", np.sum(normal_qa_results == 1) / normal_qa_results.shape[0])
    
    # image level 结果
    unknown_in_abnormal = 0
    for r in records:
        image_id = r['image_id']

        ans, out_text = r['answer'], r['output']
        pred = get_model_answer(out_text, mode=args.mode)
        if pred == -1:
            if r['is_anomaly']: unknown_in_abnormal += 1
            image_info[image_id]['pred'].append(-1)
        else:
            if args.protocol == 'v1':
                if pred == ans:  # 答对了
                    image_info[image_id]['pred'].append(1 if r['is_anomaly'] else 1)
                else:  # 答错了
                    image_info[image_id]['pred'].append(0 if r['is_anomaly'] else 1)
            elif args.protocol == 'v2':
                if pred == 3:  
                    # 预测为Normal就不往defects里加框
                    image_info[image_id]['pred'].append(0)
                    image_info[image_id]['normals'].extend(r['options'])
                else:  
                    # 预测为非Normal就往defects里加框
                    image_info[image_id]['pred'].append(1)
                    image_info[image_id]['defects'].append(r['options'][pred])
            else:
                raise NotImplementedError(f"Not implement protocol {args.protocol}.")
    
    image_info = [image_info[i] for i in range(len(image_info))]
    gts = [item['gt'] for item in image_info]
    preds = []
    for item in image_info:
        if 1 in item['pred']:
            preds.append(1)
        elif 0 in item['pred']:
            preds.append(0)
        else: preds.append(-1)
    
    print("Unknown图像占比:", np.sum(np.array(preds) == -1), np.sum(np.array(preds) == -1) / len(preds))
    
    print("#"*16, u"只计算识别样本", "#"*16)
    if True:  # 不算unknown样本的结果
        preds_np = np.array(preds)
        gts_np = np.array(gts)[preds_np != -1]
        preds_np = preds_np[preds_np != -1]

        conf_m = confusion_matrix(gts_np, preds_np)
        print(conf_m)
        over_kill = conf_m[0, 1] / (conf_m[0, 0] + conf_m[0, 1])
        miss = conf_m[1, 0] / (conf_m[1, 0] + conf_m[1, 1])
        acc = accuracy_score(gts_np, preds_np)
        precision = precision_score(gts_np, preds_np)
        recall = recall_score(gts_np, preds_np)
        auroc = roc_auc_score(gts_np, preds_np)

        print(u"过杀率:", over_kill)
        print(u"漏检率:", miss)
        print(u"Acc:", acc)
        print(u"Precision:", precision)
        print(u"Recall:", recall)
        print(u"AUROC:", auroc)
    
    if args.loc:
        cal_anomaly_scores(image_info, split)