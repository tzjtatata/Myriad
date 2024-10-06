import jsonlines
import os


if __name__ == '__main__':
    with jsonlines.open('/mnt/vdb1/datasets/TrainADDataset/AQA_balanced_train.jsonl', 'r') as reader:
        records = list(reader)
    
    with jsonlines.open('/mnt/vdb1/datasets/TrainADDataset/AQA_balanced_train_correct.jsonl', 'w') as writer:
        for item in records:
            opts = item['options']
            width, height = item['width'], item['height']
            new_opts = []
            for opt in opts:
                x1, y1, x2, y2 = opt
                if x1 > width: x1 = width - 1
                if x2 > width: x2 = width - 1
                if y1 > height: y1 = height - 1
                if y2 > height: y2 = height - 1
                new_opts.append([x1, y1, x2, y2])
            item['options'] = new_opts
            writer.write(item)