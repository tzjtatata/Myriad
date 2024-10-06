import numpy as np
import os
import argparse
import jsonlines
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    r_path = args.path
    # r_path = "/home/lyz/vdb/results/minigpt/mvtec/fg_normal_cutpaste_ve2word_nodetach_adaptor_lr=1e-3/20230913194611/results_ckpt9_task=ad_split=eval_20230914_121258.jsonl"
    # r_path = "/home/lyz/vdb/results/minigpt/mvtec/ve_normal_cutpaste_ve=aprilgan_ve2word_24k/20230911002148/results_ckpt8_task=ad_split=eval_20230912_215333.jsonl"
    # r_path = "/home/lyz/vdb/results/minigpt/mvtec/ve_normal_cutpaste_ve=mvtec+aprilgan_ve2word_qformer=append/20230912232425/results_ckpt9_task=ad_split=eval_20230913_112054.jsonl"
    with jsonlines.open(r_path, 'r') as reader:
        records = list(reader)
    # print(r_path)

    preds = []
    gts = []
    overkill_scores, missing_scores = [], []  # 分别是过杀的分数和漏检的分数统计。
    for r in records:
        # preds.append(r['anomaly_score'])
        if r['is_anomaly']:
            gts.append(1)

            missing_scores.append(float(r['anomaly_score']))
        else:
            gts.append(0)

            overkill_scores.append(float(r['anomaly_score']))
        preds.append(float(r['anomaly_score']))

    plt.hist([overkill_scores, missing_scores], label=['normal', 'anomaly'])
    # plt.yticks([i for i in range(0, 510, 40)])
    plt.ylim(0, 1000)
    # plt.xticks([i/10. for i in range(0, 11)])
    plt.legend(loc='upper left')
    plt.savefig('./test.png')
    # print(len(preds),len(gts))
    # print("Abnormal:")
    # ab_hist = np.histogram(abnormal_scores)
    # print(ab_hist[1].tolist())
    # print(ab_hist[0].tolist())

    # gts = np.array(gts)
    # preds = np.array(preds).astype(np.f)
    # preds = 
    # print("Normal:")
    # normal_hist = np.histogram(normal_scores)
    # print(normal_hist[1].tolist())
    # print(normal_hist[0].tolist())
    # print(gts.max(),gts.min().preds.max(),gts.min())
    print("AUROC:", roc_auc_score(gts, preds))