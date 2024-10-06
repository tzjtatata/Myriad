import os
import sys
import cv2
import json
import numpy as np
import datetime
import jsonlines
from typing import List, Dict

sys.path.insert(1, "/mnt/vdb1/datasets/EvalADDataset/meta")
sys.path.insert(1, "/home/lyz/vdb/codes/minigpt")
# 这个的逗号之间不能有空格=。=
from minigpt4.processors.transform import PlainBoxFormatter
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise


def box_formater(box):
    # from xywh -> xyxy
    x, y, w, h= box
    return [x, y, x+w, y+h]


class ALEvaluator:
    """
        Evaluator for Anomaly Location(Fine-Grained Perception).
    """
    def __init__(self, root) -> None:
        self.root = root
        self.coco_path = os.path.join(root, 'val_coco.json')
        anno_path = os.path.join(root, 'anomaly_location.json')
        assert os.path.exists(anno_path) and os.path.isfile(anno_path), f"{anno_path} does not exist or is not a file."
        with open(anno_path, 'r', encoding='utf-8') as f:
            self.anno = json.load(f)  # anomaly_location annotaion file
    
    @staticmethod
    def _check_format(preds: List[Dict]):
        # eval format: {"image_id": 10, "category_id": 3, "bbox": [187.0, 37.0, 16.0, 81.0], "score": 0.9}
        for k in ["image_id", 'category_id', 'bbox', 'score']:
            if k not in preds[0]:
                return 'self'
        return 'coco'
    
    @staticmethod
    def find_coco_id(img_info: Dict, coco_annos: List[Dict], strict=False):
        img_annos = coco_annos['images']
        rel_path = os.path.join(img_info['scene'], 'test', img_info['is_anomaly'], img_info['image_name'])
        for img in img_annos:
            if img['rel_path'] == rel_path:
                return img['id']
        if strict:
            raise ValueError(f"Image {rel_path} not in coco annotations.")
        return None
    
    @staticmethod
    def find_image_info(img_info: Dict, coco_annos: List[Dict], strict=False):
        """
            根据相对路径，查找到对应的图片信息。包括其coco的image id和其他信息。
        """
        img_annos = coco_annos['images']
        rel_path = os.path.join(img_info['scene'], 'test', img_info['is_anomaly'], img_info['image_name'])
        for img in img_annos:
            if img['rel_path'] == rel_path:
                return img
        if strict:
            raise ValueError(f"Image {rel_path} not in coco annotations.")
        return None

    @staticmethod
    def _coco_formatter(preds: List[Dict or List], coco_annos: Dict):
        # 假定的输出格式是:  {"image_id": path, "coco_id": image_id, "output": sentences.}
        if isinstance(preds[0], List):
            preds = ALEvaluator._csv2dict(preds)
        # print(preds)
        pbf = PlainBoxFormatter()
        try:
            coco_results = []
            for item in preds:
                # 这里过滤掉正常的样本。
                if item['is_anomaly'] != 'bad': continue 
                # 检查coco_id
                info = ALEvaluator.find_image_info(item, coco_annos, strict=True)
                item['coco_id'] = info['id']
                w = info['width']
                h = info['height']

                _, boxes = pbf.extract(item['output'])  # 不能包含空格。
                if len(boxes) == 0: continue  # 直接跳过就行
                boxes = np.array(boxes).reshape(-1, 4).tolist()  # 三层结构我们可能暂时不需要，就直接展开就行
                for box in boxes:
                    box = [box[1], box[0], box[3], box[2]]  # 模型的x轴是纵轴，需要改换格式 -> x为横轴
                    box = [box[0] * w, box[1] * h, box[2] * w, box[3] * h]  # 乘以w和h，其中w是横轴长度，h为纵轴长度
                    box = [box[0], box[1], box[2]-box[0], box[3] - box[1]]  # xyxy -> xywh
                    coco_results.append(
                        {
                            "image_id": item['coco_id'],
                            "category_id": 1,  # 现在全是anomaly类
                            "bbox": box,
                            "score": 0.9  # 不好评分，就暂定0.9吧……
                        }
                    )
            return coco_results
        except KeyError:
            raise KeyError("Wrong Predictions which do not contain sufficient information for COCO format.")
    
    # 兼容@whl 2023-07-19的输出格式: [is_anomaly: str, gt_defect_types: List[str], gt_boxes: List[str], image_name: str, output: str]
    @staticmethod
    def _csv2dict(lines: List[List]):
        template = {"is_anomaly": 0, "gt_defect_types": 1, "gt_boxes": 2, "image_name": 3, "output": 4, "scene": 5}

        def construct_via_template(l):
            _ret = {}
            for k, v in template.items():
                _ret[k] = l[v]
            return _ret

        results = []
        for line in lines:
            results.append(construct_via_template(line))
        return results
    
    def extract4AL(self, results):
        coco_results = []
        pbf = PlainBoxFormatter(use_small_brackets=True)
        for r in results:
            w = r['width']
            h = r['height']
            print(r['image_id'])

            _, boxes = pbf.extract(r['output'])
            # print(r['output'], boxes)
            if len(boxes) == 0: continue  # 直接跳过就行
            boxes = np.array(boxes).reshape(-1, 4).tolist()  # 三层结构我们可能暂时不需要，就直接展开就行
            for box in boxes:
                box = [box[0] * w, box[1] * h, box[2] * w, box[3] * h]  # 乘以w和h，其中w是横轴长度，h为纵轴长度
                box = [box[0], box[1], box[2]-box[0], box[3] - box[1]]  # xyxy -> xywh
                coco_results.append(
                    {
                        "image_id": r['coco_id'],
                        "category_id": 1,  # 现在全是anomaly类
                        "bbox": box,
                        "score": 0.9  # 不好评分，就暂定0.9吧……
                    }
                )
        return coco_results

    def eval_coco(self, preds:List[Dict] or str):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        with open(self.coco_path, 'r', encoding='utf-8') as f:
            coco_annos = json.load(f)

        # 进行格式处理
        if isinstance(preds, str):
            # 这里代表一个路径，包含了一系列csv文件
            if os.path.isdir(preds):
                # 如果是一个文件夹，则是@whl的格式。
                import csv
                results_dir = preds
                assert os.path.exists(results_dir) and os.path.isdir(results_dir), f"The path {results_dir} does not exist or not is a directory."
                preds = []
                for f_name in os.listdir(results_dir):
                    if not f_name.endswith('.csv'): continue
                    f_path = os.path.join(results_dir, f_name) 
                    with open(f_path, 'r', encoding='utf-8') as f:
                        lines = list(csv.reader(f))
                    for line in lines: line.append(f_name[:-4])  # 加上scene
                    preds.extend(lines)
                # 这种格式下要进行复杂的处理才能转到coco格式。
                results = ALEvaluator._coco_formatter(preds, coco_annos)
            else:
                # 否则，可能是一个json格式的文件。
                with open(preds, 'r', encoding='utf-8') as f:
                    _res = json.load(f)
                results = self.extract4AL(_res)
        
        # print(results)
        # 可视化测试
        sys.path.insert(1, "/home/ubuntu/lyz/MiniGPT-4")
        from visual_utils import annotate
        pred_count = 0
        gt_count = 0
        for img_info in coco_annos['images']:
            coco_id = img_info['id']
            # if coco_id != 3: continue
            bboxes = [box_formater(r['bbox']) for r in results if r['image_id'] == coco_id]
            gt_boxes = [box_formater(gt['bbox']) for gt in coco_annos['annotations'] if gt['image_id'] == coco_id]
            # print(gt_boxes)
            if len(gt_boxes) == 0: continue
            gt_count += 1
            if len(bboxes) != 0: pred_count += 1
            frame = annotate(os.path.join("/home/ubuntu/whl/data/EvalADDataset/2cls_highshot", img_info['rel_path']), bboxes+gt_boxes, ['pred' for _ in bboxes] + ['gt' for _ in gt_boxes])
            cv2.imwrite(os.path.join('./test', f"{img_info['scene']}_{img_info['rel_path'].split('/')[-1][:-4]}.jpg"), frame)
        
        print(u"漏检率", pred_count * 1.0 / gt_count)
        
        with open('./temp_file.json', 'w', encoding='utf-8') as f:
            json.dump(results, f)
        
        cocoGt = COCO(self.coco_path)
        cocoDt = cocoGt.loadRes('./temp_file.json')
        cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

    def eval_ad(self, result_file: str):
        pbf = PlainBoxFormatter(use_small_brackets=True)
        if result_file.endswith('.json'):
            with open(result_file, 'r', encoding='utf-8') as f:
                _res = json.load(f)
        else:
            with jsonlines.open(result_file, 'r') as reader:
                _res = list(reader)
        pred = []
        gt = []
        for r in _res:
            _, boxes = pbf.extract(r['output'])
            # print(r['output'], boxes)
            boxes = np.array(boxes).reshape(-1, 4)
            if boxes.shape[0] == 0:
                # 证明预测是无异常
                _pred = 0
            else:
                # print(r['output'])
                _pred = 1
            _gt = 1 if r['is_anomaly'] else 0
            gt.append(_gt)
            pred.append(_pred)
        from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix
        cfm = confusion_matrix(gt, pred)

        precisions, recalls, thresholds = precision_recall_curve(gt, pred)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        f1_sp = np.max(f1_scores[np.isfinite(f1_scores)])

        print(cfm)
        print(u"过杀:", cfm[0, 1] / (cfm[0, 0]+ cfm[0,1]) )
        print(u"漏检:", cfm[1, 0] / (cfm[1, 0]+ cfm[1,1]) )
        print("Acc: ", accuracy_score(gt, pred))
        print("Recall: ", recall_score(gt, pred))
        print("Precision: ", precision_score(gt, pred))
        print("AUROC: ", roc_auc_score(gt, pred))
        print('AP-cls:',average_precision_score(gt,pred))
        print('F1-max-cls:',f1_sp)

    @staticmethod
    def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
        from sklearn.metrics import auc
        from skimage import measure
        # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
        binary_amaps = np.zeros_like(amaps, dtype=bool)
        min_th, max_th = amaps.min(), amaps.max()
        delta = (max_th - min_th) / max_step
        pros, fprs, ths = [], [], []
        for th in np.arange(min_th, max_th, delta):
            binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
            pro = []
            for binary_amap, mask in zip(binary_amaps, masks):
                for region in measure.regionprops(measure.label(mask)):
                    tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                    pro.append(tp_pixels / region.area)
            inverse_masks = 1 - masks
            fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
            fpr = fp_pixels / inverse_masks.sum()
            pros.append(np.array(pro).mean())
            fprs.append(fpr)
            ths.append(th)
        pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
        idxes = fprs < expect_fpr
        fprs = fprs[idxes]
        fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
        pro_auc = auc(fprs, pros[idxes])
        return pro_auc

    def eval_seg(self, preds):
        from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
        assert "expert_mask" in preds[0], "You should provide the mask of a visual expert's output, like patchcore or April GAN. "
        assert "gt_mask" in preds[0], f"You should provide the Ground Truth as gt_mask, now there is only {list(preds.keys())}. "

        # metrics
        table = []
        gt_px = []
        pr_px = []
        for item in preds:
            gt_px.append(item['gt_mask'].squeeze(1).numpy())
            pr_px.append(item['anomaly_maps'])
        gt_px = np.array(gt_px)
        pr_px = np.array(pr_px)

        auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())
        ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())
        # f1_px
        precisions, recalls, thresholds = precision_recall_curve(gt_px.ravel(), pr_px.ravel())
        f1_scores = (2 * precisions * recalls) / (precisions + recalls)
        f1_px = np.max(f1_scores[np.isfinite(f1_scores)])
        # aupro
        if len(gt_px.shape) == 4:
            gt_px = gt_px.squeeze(1)
        if len(pr_px.shape) == 4:
            pr_px = pr_px.squeeze(1)
        aupro = ALEvaluator.cal_pro_score(gt_px, pr_px)

        ret = {
            "auroc_px": np.round(auroc_px * 100, decimals=1),
            "f1_px": np.round(f1_px * 100, decimals=1),
            "ap_px": np.round(ap_px * 100, decimals=1),
            "aupro": np.round(aupro * 100, decimals=1),
        }
        return ret


class EvalADGPT:

    def __init__(self, root) -> None:
        self.root = root
        self.dd_anno = os.path.join(root, 'defects_detection_test.json')  # defect detection annotation file.
        self.sc_anno = os.path.join(root, 'object_description_test.json')  # scene caption annotation file.
        self.al_anno = os.path.join(root, 'anomaly_location.json')  # anomaly_location annotaion file
    
    def _check_exist(self, task_name):
        task_annos = getattr(self, task_name, None)
        if task_annos is not None:
            if os.path.exists(task_annos):
                return True
        return False


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, required=True)
    args = parser.parse_args()
    evaluator = ALEvaluator("/mnt/vdb1/datasets/EvalADDataset")
    # evaluator.eval_ad(result_file)
    # evaluator.eval_coco(result_file)
    evaluator.eval_ad(args.result_path)