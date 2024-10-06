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
import numpy as np
import argparse
import jsonlines
from sklearn.metrics import precision_score, accuracy_score, recall_score, roc_auc_score, confusion_matrix
# import torch


def draw_roc(gts, preds):
    from sklearn.metrics import roc_curve, auc
    from matplotlib import pyplot as plt

    fpr, tpr, thresholds = roc_curve(gts, preds, pos_label=1)
    
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
            lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig("./roc.png")
    plt.show()


def get_model_answer(text, mode=0):
    if mode == 0:
        abnormal_words = [
            'Yes',
        ]
        normal_words = [
            'No',
        ]
        def has_words(input, words):
            for word in words:
                if word in input:
                    return True
            return False

        if has_words(text, abnormal_words):
            return 1
        elif has_words(text, normal_words):
            return 0
        return -1  
    elif mode == 2:
        if 'C' in text:
            return 0
        if ('is A.' in text) or ('is B.' in text):
            return 1
        return -1
    elif mode == 3:
        if 'D' in text:
            return 0
        if ('is A.' in text) or ('is B.' in text) or ('is C.' in text):
            return 1
        return -1
    else:
        raise NotImplementedError("")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, required=True)
    parser.add_argument("--protocol", type=str, choices=['v1', 'v2'], default='v2')  # v1: 只有选对了选项才认为是abnormal，否则是normal; v2: 不选D就是AbNormal，选D就是Normal
    parser.add_argument("--mode", type=int, default=0)
    args = parser.parse_args()
    with jsonlines.open(args.result_path, 'r') as reader:
        records = list(reader)
    
    preds = []
    gts = []  # 这个是记录normal或者是abnormal的结果
    for r in records:
        out_text = r['output']
        pred = get_model_answer(out_text, mode=args.mode)
        # print(out_text, pred, ans)
        gts.append(1 if r['is_anomaly'] else 0)
        preds.append(pred)
        if pred == -1:
            print(out_text)
    
    # print(len(qa_results))
    qa_results_np = np.array(preds)
    print(u"预测的不知道是什么东西的数量:", np.sum(qa_results_np == -1))
    print(u"预测为异常的样本数以及比率:", np.sum(qa_results_np == 1), np.sum(qa_results_np == 1) / (len(records)-np.sum(qa_results_np == -1)))
    print(u"预测为正常的样本数以及比率:", np.sum(qa_results_np == 0), np.sum(qa_results_np == 0) / (len(records)-np.sum(qa_results_np == -1)))
    
    # image level 结果
    image_info = {}
    unknown_in_abnormal = 0
    has_anomaly_score = False
    for r in records:
        image_id = r['image_id']
        if image_id not in image_info:
            image_info[image_id] = {
                "id": image_id,
                "gt": 1 if r['is_anomaly'] else 0,
                "pred": [],
            }
        else:
            image_info[image_id]['gt'] += 1 if r['is_anomaly'] else 0
        
        out_text = r['output']
        pred = get_model_answer(out_text, mode=args.mode)
        if pred == -1:
            if r['is_anomaly']: unknown_in_abnormal += 1
        image_info[image_id]['pred'].append(pred)
        
        if 'anomaly_map_scores' in r:
            has_anomaly_score = True
            image_info[image_id]['anomaly_map_scores'] = r['anomaly_map_scores']
    
    image_info = [image_info[i] for i in range(len(image_info))]
    gts = [1 if item['gt'] >= 1 else 0 for item in image_info]
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
    
    if has_anomaly_score:
        print("#"*16, u"使用anomaly map scores", "#"*16)
        anomaly_map_preds = [item['anomaly_map_scores'] for item in records]
        gts = [item['is_anomaly'] for item in records]
        preds_np = np.array(anomaly_map_preds)
        print(preds_np.max(), preds_np.min())
        gts_np = np.array(gts)
        print(preds_np.shape, gts_np.shape)
        auroc = roc_auc_score(gts_np, preds_np)
        print(u"AUROC:", auroc)
        for th in range(1, 10):
            th_preds = preds_np.copy()
            th_preds[th_preds >= th * 0.1], th_preds[th_preds < th * 0.1] = 1, 0
            print(f"Th={th * 0.1} Acc:", accuracy_score(gts_np, th_preds))
        draw_roc(gts_np, preds_np)