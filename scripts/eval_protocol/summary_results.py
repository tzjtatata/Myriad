import os
import jsonlines
import argparse
import numpy as np
from sklearn.metrics import precision_score, accuracy_score, recall_score, roc_auc_score, confusion_matrix


def get_model_answer(text, mode=0):
    if mode == 0:
        abnormal_words = [
            'has defect',
            'have defect',
            'have a defect',
            'has a defect',
            'has 1 defect',
            'has 2 defect', 
            'is damaged',
            'has a small defect', 
            'is broken',
            'has some defect',
            'has some anomalies',
            'looks damaged',
            'looks a little bit disfigured',
            'a bit blurry', 
            'bit distorted',
            'bit irregular',
            'a bit weird',
            'a flaw',
            'signs of defects', 
            'shows defects',
            'defect-like', 
            'sort of defect',
            'a number of defect', 
            'some kind of defect',
            'a bit odd',
            'show a crack',
            'show defect',
            'have some kind',
            'show some kind', 
            'certainly defect',
            'a little bitter', 
            'a bit unusual',
            'a bit strange',
            'has a scratch', 
            'have some defect',
            'a series of small defect', 
            'have some issues',
            'show some defect', 
            'have a crack',
            'has some problems', 
            'has a hole',
            'have a scratch', 
            'Yes', 
            'There are 2 defect', 
            'There is an anomaly', 
            'There are two defect',
            'There are three', 
            'have two anomalies', 
            'There are two anomalies',
            'has an anomaly', 
            'contains an anomaly', 
        ]
        normal_words = [
            'has no defect',
            'have no defect',
            'be undamaged', 
            'looks good',
            'look good',
            'looks fine',
            'look fine', 
            'looks perfect', 
            'look perfect', 
            'is perfect', 
            'is normal',
            'looks normal', 
            'look normal', 
            'looks defect free', 
            'looks defect-free',
            'looks okay',
            'No,',  
            'There is no anomaly', 
            'There are no defect', 
            'There is no defect', 
            'There are no',
            'has no anomalies', 
            'has 0 defect', 
            'contains no defect', 
            'contains no anomal', 
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


def get_performance(f_path):
    with jsonlines.open(f_path, 'r') as reader:
        records = list(reader)
    
    # image level 结果
    image_info = {}
    unknown_in_abnormal = 0
    has_anomaly_score = False
    scene_results = {}
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
        pred = get_model_answer(out_text)
        if pred == -1:
            if r['is_anomaly']: unknown_in_abnormal += 1
        image_info[image_id]['pred'].append(pred)
        
        
        
        scene = r['scene'] if 'scene' in r else r['image_path'].split('/')[1]
        if scene not in scene_results:
            scene_results[scene] = {
                'gt': [], 
                'pred': [],
                'score': []
            }
        
        score_key = 'anomaly_map_scores' if 'anomaly_map_scores' in r else 'anomaly_score'
        if score_key in r:
            has_anomaly_score = True
            image_info[image_id]['anomaly_map_scores'] = float(r[score_key])

        if pred != -1:
            scene_results[scene]['gt'].append(1 if r['is_anomaly'] else 0)
            scene_results[scene]['pred'].append(pred)
            scene_results[scene]['score'].append(float(r[score_key]))

    print("#"*16, f_path, "#"*16)
    accuracy_ = []
    auroc_ = []
    th_acc_ = []
    for scene in scene_results.keys():
        gt, pred = np.array(scene_results[scene]['gt']), np.array(scene_results[scene]['pred'])
        score = np.array(scene_results[scene]['score'])

        th = score[gt==0].max().item()
        th_score = np.zeros_like(score)
        th_score[score > th] = 1

        acc = accuracy_score(gt, pred)
        accuracy_.append(acc)
        auroc = roc_auc_score(gt, score)
        auroc_.append(auroc)
        th_acc = accuracy_score(gt, th_score)
        th_acc_.append(th_acc)

    return np.mean(accuracy_), np.mean(auroc_), np.mean(th_acc_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, required=True)
    args = parser.parse_args()

    json_files = { }
    for f in os.listdir(args.result_path):
        if f.endswith('jsonl'):
            ckpt_ord = int(f.split('_')[1].replace('ckpt', ''))
            if ckpt_ord not in json_files:
                json_files[ckpt_ord] = {
                    '1cls': [], 
                    '1shot': [], 
                    '2shot': [],
                    '4shot': [],
                }
            
            if '1cls' in f:
                json_files[ckpt_ord]['1cls'].append(f)
            else:
                assert 'shot' in f
                few = int(f.split('kshot=')[-1].split('_')[0])
                json_files[ckpt_ord][f'{few}shot'].append(f)

    k_order = ['1cls', '1shot', '2shot', '4shot']
    with open(os.path.join(args.result_path, 'summary_v2.txt'), 'w') as f:
        f.write("Head 1cls(Myriad) 1shot(Myriad) 2shot(Myriad) 4shot(Myriad)   1cls(Expert) 1shot(Expert) 2shot(Expert) 4shot(Expert)\n")
        result_keys = sorted(list(json_files.keys()))
        for k in result_keys:
            r_str = f'{k:03d} '
            r_record = {}
            for proc, f_names in json_files[k].items():
                if len(f_names) == 0:
                    r_record[proc] = ['-', '-']
                elif len(f_names) == 1:
                    try:
                        acc, auroc, _ = get_performance(os.path.join(args.result_path, f_names[0]))
                        r_record[proc] = [f'{acc:.4f}', f'{auroc:.4f}']
                    except:
                        print("skip.")
                        r_record[proc] = ['-', '-']
                        continue
                else:
                    best_acc, best_auroc = -1, 0.0
                    for f_name in f_names:
                        try:
                            acc, auroc, _ = get_performance(os.path.join(args.result_path, f_names[0]))
                            if acc > best_acc:
                                best_acc = acc
                                best_auroc = auroc
                        except:
                            print("skip.")
                            continue
                    if best_acc == -1:
                        print(f"skip ckpt{k} {proc}")
                        r_record[proc] = ['-', '-']
                        continue
                    r_record[proc] = [f'{best_acc:.4f}', f'{best_auroc:.4f}']
            r_str += '{} {} {} {}   {} {} {} {}\n'.format(*([r_record[_k][0] for _k in k_order] + [r_record[_k][1] for _k in k_order]))
            f.write(r_str)
        f.write('#' * 32 + '\n')
        