import supervision as sv
from typing import List, Tuple
import numpy as np
import cv2
from PIL import Image, ImageDraw
import colorsys,random


def annotate(image_source: np.ndarray or str, boxes: np.ndarray or List[List or Tuple], phrases: List[str]) -> np.ndarray:
    # preprocess
    if isinstance(image_source, str):
        image_source = np.array(Image.open(image_source).convert('RGB'))
    if isinstance(boxes, List):
        boxes = np.array(boxes, dtype=float)
    
    h, w, _ = image_source.shape
    detections = sv.Detections(xyxy=boxes)

    box_annotator = sv.BoxAnnotator()
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=phrases)
    return annotated_frame


def draw_box(img: Image.Image, boxes: np.ndarray or List, save_path:str=None, color: tuple =(0, 255, 0)):
    draw = ImageDraw.Draw(img)
    if isinstance(boxes, np.ndarray):
        boxes = boxes.tolist()

    for box in boxes:
        draw.rectangle(box, outline=color)  # 默认是绿色
    if save_path is None:
        return img
    img.save(save_path)


def apply_ad_scoremap(image: Image.Image, scoremap:np.ndarray, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


# Thanks to Chunpu Liu.
def get_n_hls_colors(num):

    hls_colors=[]
    i=0
    step=360.0/num

    while i<360:
        h=i
        s=90+random.random()*10
        l=50+random.random()*10
        _hlsc=[h/360.0,l/100.0,s/100.0]
        hls_colors.append(_hlsc)
        i+=step

    return hls_colors


def ncolors(num):

    rgb_colors=[]
    if num<1:
        return rgb_colors
    
    hls_colors=get_n_hls_colors(num)
    
    for hlsc in hls_colors:
        _r,_g,_b=colorsys.hls_to_rgb(hlsc[0],hlsc[1],hlsc[2])
        r,g,b=[int(x*255.0) for x in (_r,_g,_b)]
        rgb_colors.append([r,g,b])

    return rgb_colors


def mask2RGB(mask: np.ndarray, num_classes: int, save_path=None) -> None or Image.Image:
    # preprocess
    color_list=ncolors(num_classes)
    color_dict={}
    for i in range(len(color_list)):
        color_dict[i]=color_list[i]
    color_mask_npy=np.zeros((mask.shape[0],mask.shape[1],3),dtype=np.uint8)
    for label,color in color_dict.items():
        color_mask_npy[mask==label]=color
    color_mask=Image.fromarray(color_mask_npy)
    if save_path is not None:
        color_mask.save(save_path)
    return color_mask


def draw_mask(image_source: Image.Image, mask: np.ndarray, num_classes:int, alpha:float = 0.5, save_path=None) -> None or Image.Image:
    image = image_source.convert('RGBA')
    mask_img = mask2RGB(mask, num_classes).convert('RGBA')
    fusion = Image.blend(image, mask_img, alpha=alpha)
    if save_path is not None:
        fusion.save(save_path)
    else:
        return fusion