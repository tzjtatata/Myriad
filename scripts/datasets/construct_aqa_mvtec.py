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

def rescale_box(coor, scale_factor):
    x1, y1, x2, y2 = coor
    c_x, c_y = (x1 + x2) / 2, (y1+y2) /2
    half_w, half_h = c_x - x1, c_y - y1
    rescale_hw, rescale_hh = half_w * scale_factor, half_h * scale_factor

    return int(c_x - rescale_hw), int(c_y - rescale_hh), int(c_x + rescale_hw), int(c_y + rescale_hh)


if __name__ == '__main__':
    root = "/mnt/vdb1/datasets/EvalADDataset"
    ve_root = "/mnt/vdb1/datasets/aprilgan_processresults"
    vis_root = root
    anno_path = os.path.join(root, "DC_MVTEC_test_normal.jsonl")

    # 4487 lines, 每行都是对应一张测试图片
    with jsonlines.open(anno_path, 'r') as reader:
        annos = list(reader)

    sent_id = 0
    miss_count = 0
    have_done = {}
    img_id = 0
    with jsonlines.open(os.path.join(root, 'DC_MVTEC_test_gt.jsonl'), 'w') as writer:
        for ann in tqdm(annos, total=len(annos), desc="Processing To AQA format"):
            img_path = os.path.join(vis_root, ann['img_path'])
            # if 'good' in img_path: continue
            if img_id in have_done: continue

            img = Image.open(img_path)
            width, height = img.size

            ve_path = ann['ve_path']
            ve_path = os.path.join(ve_root, ve_path[:-3]+'png')
            # 检验是否有错误的ve_path；结果：全对
            # print(ve_path, os.path.exists(ve_path))
            if not os.path.exists(ve_path): 
                raise ValueError(u"存在Vision Expert地址错误: "+f"{ve_path}")
            ve = cv2.imread(ve_path)
            # cv2的resize函数脑子有问题，格式是(width, height)而不是numpy的shape
            ve = cv2.resize(ve, (width, height), interpolation=cv2.INTER_NEAREST)[:, :, 0] # (height, width), range: (0, 1)
            # print(ve.max(), ve.min())  # 确定了值域没问题

            # 获取gt mask
            if 'good' in ann['img_path']:  # Normal
                gt = np.zeros((height, width)).astype(float)
            else:
                prefixes = ann['img_path'].split('/')  # 场景/split/bad/具体图像号.JPG
                gt_path = os.path.join(vis_root, prefixes[0], prefixes[1], 'ground_truth', *prefixes[3:])
                gt_path = gt_path[:-4]+'_mask.png'  # 和图像不一样，gt是png结尾
                gt = np.array(Image.open(gt_path).convert('L')) > 0
                gt = gt.astype(float)
            # print(gt.shape, gt.max(), gt.min())

            # 从Vision Expert的mask里获取proposals
            _contours = []
            _boxes = []
            _, ve_threshold = cv2.threshold(ve, 77, 255, 0)
            # print(ve_threshold.shape, ve_threshold.max(), ve_threshold.min())
            contours, hierarchy = cv2.findContours(ve_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                if cnt_area <= width * height / (224. * 224): # 输入时一个像素都不到的异常就放大
                    rescale_factor = (width * height) / (224 * 224.)  # 其实就是224*224的图像中，一个像素映射过来的面积
                    x1, y1, x2, y2 = rescale_box((x, y, x+w, y+h), rescale_factor * 4)  # 这里采用4像素的版本
                    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, width), min(y2, height)
                else:
                    x1, y1, x2, y2 = x, y, x+w, y+h
                _boxes.append([x1, y1, x2, y2, cnt_area])
                # print(_boxes)
            assert len(_boxes) != 0, f"There is no proposal in {img_path}"

            _boxes = sorted(_boxes, key=lambda x: x[-1], reverse=True)
            _boxes = [item[:-1] for item in _boxes]
            
            # 判断各个框是否是Normal
            normals = []
            abnormals = []
            for box in _boxes:
                x1, y1, x2, y2 = box
                if np.sum(gt[y1:y2, x1:x2]) == 0:  # Normal
                    normals.append(box)
                else:
                    abnormals.append(box)
            if ann['is_anomaly'] == '1':
                if len(abnormals) == 0:
                    print(f"Abnormal with no abnormal proposal in {img_path}")
                    miss_count += 1
            
            print("Normal:", len(normals))
            print("AbNormal:", len(abnormals))
            abnormals = abnormals[:1]
            normals = normals[:3-len(abnormals)]

            # 绘制Normal的图。
            printed_img = draw_box(Image.open(img_path).convert('RGB'), np.array(normals))
            
            # 补全Normal图
            add_normal = []
            to_be_filled = 3-len(normals)
            for i in range(to_be_filled):
                add_normal.append(get_random_normal_box(width, height, gt, scales=[32, 48, 64, 80, 96, 112, 128]))
                assert check_box_valid(add_normal[-1], width, height), u"随机生成的box超出了界了"
            # print("Added Normal:", len(add_normal))
            printed_img = draw_box(printed_img, np.array(add_normal), color=(0, 0, 255))  # 用蓝色标记新增加的normal
            normals += add_normal
            
            # print("Abnormals:", len(abnormals))
            draw_box(printed_img, np.array(abnormals), color=(255, 0, 0), save_path=os.path.join(os.path.dirname(__file__), 'visualize',f"{'_'.join(ann['img_path'][:-4].split('/'))}.png"))

            # 依次组建问题:
            item = {
                'img_path': ann['img_path'],
                'image_id': img_id,
                've_path': ann['ve_path'],  # 这里存的是相对路径
                'expression': 'defect',  # TODO: 这里有GT, 其实完全可以提取类型的
                'is_anomaly': ann['is_anomaly'], 
                'caption': ann['caption'], 
                'abnormal_boxes': abnormals,
                'normal_boxes': normals,  
                'dataset_name': 'MVTEC', 
                'height': height, 
                'width': width, 
                'sent_id': sent_id, 
                'split': 'test', 
            }
            sent_id += 1
            writer.write(item)

            # print(item)
            have_done[img_id] = True
            img_id += 1
            # break
    print(u"处理的图片数:", len(have_done))
    print(u"AprilGAN没有判断异常的图片个数为:", miss_count)
        
        
        


