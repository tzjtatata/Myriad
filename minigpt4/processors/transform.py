from typing import Dict, Any, Tuple, Optional, List, Union
import re

from PIL import Image
Box = List[Union[float, int]]
Boxes = List[Box]
BoxesSeq = List[Boxes]


def de_norm_box_xyxy(box, *, w, h):
    x1, y1, x2, y2 = box
    x1 = x1 * w
    x2 = x2 * w
    y1 = y1 * h
    y2 = y2 * h
    box = x1, y1, x2, y2
    return box


def box_xywh_to_xyxy(box, *, w=None, h=None):
    x, y, bw, bh = box
    x2 = x + bw
    y2 = y + bh
    if w is not None:
        x2 = min(x2, w)
    if h is not None:
        y2 = min(y2, h)
    box = x, y, x2, y2
    return box


def norm_box_xyxy(box, *, w, h):
    x1, y1, x2, y2 = box

    # Calculate the normalized coordinates with min-max clamping
    norm_x1 = max(0.0, min(x1 / w, 1.0))
    norm_y1 = max(0.0, min(y1 / h, 1.0))
    norm_x2 = max(0.0, min(x2 / w, 1.0))
    norm_y2 = max(0.0, min(y2 / h, 1.0))

    # Return the normalized box coordinates
    normalized_box = (round(norm_x1, 3), round(norm_y1, 3), round(norm_x2, 3), round(norm_y2, 3))
    return normalized_box


def norm_point_xyxy(point, *, w, h):
    x, y = point
    norm_x = max(0.0, min(x / w, 1.0))
    norm_y = max(0.0, min(y / h, 1.0))
    point = norm_x, norm_y
    return point


def expand2square(pil_img, background_color=(255, 255, 255)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def box_xyxy_expand2square(box, *, w, h):
    if w == h:
        return box
    if w > h:
        x1, y1, x2, y2 = box
        y1 += (w - h) // 2
        y2 += (w - h) // 2
        box = x1, y1, x2, y2
        return box
    assert w < h
    x1, y1, x2, y2 = box
    x1 += (h - w) // 2
    x2 += (h - w) // 2
    box = x1, y1, x2, y2
    return box


def point_xy_expand2square(point, *, w, h):
    pseudo_box = (point[0], point[1], point[0], point[1])
    expanded_box = box_xyxy_expand2square(box=pseudo_box, w=w, h=h)
    expanded_point = (expanded_box[0], expanded_box[1])
    return expanded_point


class Expand2square:
    def __init__(self, background_color=(255, 255, 255)):
        self.background_color = background_color

    def __call__(self, image: Image.Image, labels: Dict[str, Any] = None) -> Tuple[Image.Image, Optional[Dict[str, Any]]]:
        width, height = image.size
        processed_image = expand2square(image, background_color=self.background_color)
        if labels is None:
            return processed_image, labels
        if 'boxes' in labels:
            bboxes = [box_xyxy_expand2square(bbox, w=width, h=height) for bbox in labels['boxes']]
            labels['boxes'] = bboxes
        if 'points' in labels:
            points = [point_xy_expand2square(point, w=width, h=height) for point in labels['points']]
            labels['points'] = points
        return processed_image, labels


class PlainBoxFormatter:

    def __init__(self, use_small_brackets=False):
        small_brackets_pat = re.compile(r'\(\d(?:\.\d*)?(?:,[ ]?\d(?:\.\d*)?){3}(?:;\d(?:\.\d*)?(?:,[ ]?\d(?:\.\d*)?){3})*\)')

        middle_brackets_pat = re.compile(r'\[\d(?:\.\d*)?(?:,[ ]?\d(?:\.\d*)?){3}(?:;\d(?:\.\d*)?(?:,[ ]?\d(?:\.\d*)?){3})*\]')

        self.pat = small_brackets_pat if use_small_brackets else middle_brackets_pat

    def format_box(self, boxes: Boxes) -> str:
        box_strs = []
        for box in boxes:
            box_strs.append(','.join([f"{elem:.{self.precision}f}" for elem in box]))
        box_str = ';'.join(box_strs)
        return "[" + box_str + "]"

    def extract(self, string: str) -> List[Boxes]:
        """ balabala<boxes>balabala<boxes> -> [boxes, boxes] """
        ret = []
        for bboxes_str in self.pat.findall(string):
            bboxes = []
            bbox_strs = bboxes_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(";")
            for bbox_str in bbox_strs:
                bbox = list(map(float, bbox_str.split(',')))
                bboxes.append(bbox)
            ret.append(bboxes)
        new_string = re.sub(self.pat, '{}', string)
        return new_string, ret


if __name__ == '__main__':
    import numpy as np
    pbf = PlainBoxFormatter()
    format_str, boxes = pbf.extract('Examining this chewinggum, there exist anomoly defects of Corner Missing[0.420, 0.229, 0.782, 0.785],Small Cracks[0.534, 0.713, 0.689, 0.792].')
    boxes = np.array(boxes).reshape(-1, 4)
    whwh = np.array([1100, 900, 1100, 900])
    print(boxes)
    print(boxes * whwh)
    print(boxes * whwh /whwh)
    print(format_str.format(*boxes.tolist()))