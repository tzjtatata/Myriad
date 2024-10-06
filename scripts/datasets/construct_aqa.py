"""
    从AprilGAN等Vision Expert提供的输出mask提取出相应的test数据文件
"""
import os
import sys
import jsonlines
from tqdm import tqdm

# 处理图像的库
import cv2
from PIL import Image
import numpy as np

sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from minigpt4.visual_utils.visualize_tools import draw_box
from minigpt4.datasets.datasets.aqa_dataset import get_random_normal_box


def check_box_valid(box, width, height):
    x1, y1, x2, y2 = box
    if (x1 > width) or (x1 < 0) or (x2 > width) or (x2 < 0):
        return False
    if (y1 > height) or (y1 < 0) or (y2 > height) or (y2 < 0):
        return False
    return True


if __name__ == '__main__':
    root = "/mnt/vdb1/datasets/EvalADDataset"
    ve_root = "/mnt/vdb1/datasets/aprilgan_processresults"
    vis_root = os.path.join(root, '2cls_highshot')
    anno_path = os.path.join(root, "AL_VisA_test.jsonl")

    # 4487 lines, 每行都是对应一张测试图片
    with jsonlines.open(anno_path, 'r') as reader:
        annos = list(reader)

    sent_id = 0
    miss_count = 0
    have_done = {}
    with jsonlines.open(os.path.join(root, 'AQA_test.jsonl'), 'w') as writer:
        for ann in tqdm(annos, total=len(annos), desc="Processing To AQA format"):
            img_path = os.path.join(vis_root, ann['img_path'])
            # if 'good' in img_path: continue
            img_id = ann['image_id']
            if img_id in have_done: continue
            width, height = ann['width'], ann['height']

            ve_path = ann['aprilgan_path']
            # 这里需要做处理，因为原来的数据集里是硬编码的，现在时代变了
            ve_path = os.path.join(ve_root, *ve_path.split('/')[6:])
            # 检验是否有错误的ve_path；结果：全对
            # print(ve_path, os.path.exists(ve_path))
            # if not os.path.exists(ve_path): 
            #     raise ValueError(u"存在Vision Expert地址错误")
            ve = cv2.imread(ve_path)
            # cv2的resize函数脑子有问题，格式是(width, height)而不是numpy的shape
            ve = cv2.resize(ve, (width, height), interpolation=cv2.INTER_NEAREST)[:, :, 0] # (height, width), range: (0, 1)
            # print(ve.max(), ve.min())  # 确定了值域没问题

            # 获取gt mask
            if 'good' in ann['img_path']:  # Normal
                gt = np.zeros((height, width)).astype(float)
            else:
                prefixes = ann['img_path'].split('/')  # 场景/split/bad/具体图像号.JPG
                gt_path = os.path.join(vis_root, prefixes[0], 'ground_truth', *prefixes[1:])
                gt_path = gt_path[:-3]+'png'  # 和图像不一样，gt是png结尾
                gt = np.array(Image.open(gt_path).convert('L')) > 0
                gt = gt.astype(float)
            # print(gt.shape, gt.max(), gt.min())

            # 从Vision Expert的mask里获取proposals
            _contours = []
            _boxes = []
            _, ve_threshold = cv2.threshold(ve, 127, 255, 0)
            # print(ve_threshold.shape, ve_threshold.max(), ve_threshold.min())
            contours, hierarchy = cv2.findContours(ve_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) <= width * height / (224. * 224): continue  # 输入时一个像素都不到的异常就忽略掉。
                _contours.append(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                _boxes.append([x, y, x+w, y+h])
            
            # 判断各个框是否是Normal
            normals = []
            abnormals = []
            for box in _boxes:
                x1, y1, x2, y2 = box
                if np.sum(gt[y1:y2, x1:x2]) == 0:  # Normal
                    normals.append(box)
                else:
                    abnormals.append(box)

            # 绘制Normal的图。
            # print("Normal:", len(normals))
            printed_img = draw_box(Image.open(img_path), np.array(normals))
            
            # 补全Normal图
            add_normal = []
            if 'good' in ann['img_path']:  # Normal
                # Normal时，就三个三个为一组，形成问题
                to_be_filled = 3 - (len(normals) % 3) if len(normals) % 3 != 0 else 0
                
            else:
                # Abnormal时就两个两个一组，补全
                if 2 * len(abnormals) > len(normals):  # 如果normals不足abnormals的两倍，需要补足
                    to_be_filled = 2 * len(abnormals) - len(normals)
                elif 2 * len(abnormals) == len(normals):  # 如果正好等于，就不需要补足
                    to_be_filled = 0
                else:
                    rest_normal = len(normals) - 2*len(abnormals)
                    to_be_filled = 3 - (rest_normal % 3) if rest_normal % 3 != 0 else 0
            for i in range(to_be_filled):
                add_normal.append(get_random_normal_box(width, height, gt, scales=[32, 48, 64, 80, 96, 112, 128]))
                assert check_box_valid(add_normal[-1], width, height), u"随机生成的box超出了界了"
            # print("Added Normal:", len(add_normal))
            printed_img = draw_box(printed_img, np.array(add_normal), color=(0, 0, 255))  # 用蓝色标记新增加的normal
            normals += add_normal
            
            # print("Abnormals:", len(abnormals))
            draw_box(printed_img, np.array(abnormals), color=(255, 0, 0), save_path=os.path.join(os.path.dirname(__file__), 'visualize',f"{'_'.join(ann['img_path'][:-4].split('/'))}.png"))

            if ('bad' in img_path) and (len(abnormals) == 0):
                print(ann['img_path'])
                miss_count += 1

            # 依次组建问题:
            if len(abnormals) != 0:  # 如果存在abnormals，就1配3解决问题
                # print(u"存在异常")
                for box in abnormals:
                    options = [box] + normals[:2]
                    normals = normals[2:]
                    item = {
                        'img_path': ann['img_path'],
                        'image_id': img_id,
                        've_path': os.path.join(*ann['aprilgan_path'].split('/')[6:]),  # 这里存的是相对路径
                        'expression': 'defect',  # TODO: 这里有GT, 其实完全可以提取类型的
                        'is_anomaly': True, 
                        'options': options, 
                        'dataset_name': 'VisA', 
                        'height': height, 
                        'width': width, 
                        'sent_id': sent_id, 
                        'split': 'test', 
                    }
                    # print(item)
                    writer.write(item)
                    sent_id += 1

            if len(normals) != 0:  # 如果还剩下normals，就三个为一组组成问题
                for ind in range(len(normals) // 3):
                    options = normals[:3]
                    normals = normals[3:]
                    item = {
                        'img_path': ann['img_path'],
                        'image_id': img_id,
                        've_path': os.path.join(*ann['aprilgan_path'].split('/')[6:]),  # 这里存的是相对路径
                        'expression': 'normal',  # TODO: 这里有GT, 其实完全可以提取类型的
                        'is_anomaly': False, 
                        'options': options, 
                        'dataset_name': 'VisA', 
                        'height': height, 
                        'width': width, 
                        'sent_id': sent_id, 
                        'split': 'test', 
                    }
                    sent_id += 1
                    writer.write(item)
                    # print(item)
            have_done[img_id] = True
            # break
    print(u"处理的图片数:", len(have_done))
    print(u"AprilGAN没有判断异常的图片个数为:", miss_count)
        
        
        


