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
    print(r_path)

    preds = []
    gts = []
    overkill_scores, missing_scores = [], []  # 分别是过杀的分数和漏检的分数统计。
    for r in records:
        if r['is_anomaly']:
            gts.append(1)
            if 'perfect' in r['output']:
                missing_scores.append(r['anomaly_map_scores'])
        else:
            gts.append(0)
            if 'defects' in r['output']:
                overkill_scores.append(r['anomaly_map_scores'])
        preds.append(r['anomaly_map_scores'])

    plt.hist([overkill_scores, missing_scores], label=['overkill', 'missing'])
    # plt.yticks([i for i in range(0, 510, 40)])
    plt.ylim(0, 90)
    # plt.xticks([i/10. for i in range(0, 11)])
    plt.legend(loc='upper left')
    plt.savefig('./test.png')
    
    # print("Abnormal:")
    # ab_hist = np.histogram(abnormal_scores)
    # print(ab_hist[1].tolist())
    # print(ab_hist[0].tolist())


    # print("Normal:")
    # normal_hist = np.histogram(normal_scores)
    # print(normal_hist[1].tolist())
    # print(normal_hist[0].tolist())
    
    print("AUROC:", roc_auc_score(gts, preds))