import os
import shutil
from collections import Counter

def calculate_iou(box1, box2):
    # Extract values from YOLO annotation format
    _, x1, y1, w1, h1 = box1
    _, x2, y2, w2, h2 = box2

    # Calculate coordinates of the intersection rectangle
    x_intersection = max(x1 - w1 / 2, x2 - w2 / 2)
    y_intersection = max(y1 - h1 / 2, y2 - h2 / 2)
    w_intersection = min(x1 + w1 / 2, x2 + w2 / 2) - x_intersection
    h_intersection = min(y1 + h1 / 2, y2 + h2 / 2) - y_intersection

    # Calculate area of intersection rectangle
    area_intersection = max(0, w_intersection) * max(0, h_intersection)

    # Calculate areas of individual bounding boxes
    area_box1 = w1 * h1
    area_box2 = w2 * h2

    # Calculate union of bounding boxes
    area_union = area_box1 + area_box2 - area_intersection

    # Calculate IOU
    iou = area_intersection / area_union if area_union > 0 else 0

    return iou


def parse_line(line):
    # 라인에서 정보 추출
    parts = line.split()
    class_id = int(parts[0])
    center_x, center_y, width, height, confidence_score = map(float, parts[1:])
    return class_id, center_x, center_y, width, height, confidence_score


def calculate_iou(box1, box2):
    # 두 상자 간의 IoU 계산
    # box = (center_x, center_y, width, height)
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    intersection_x = max(0, min(x1 + w1 / 2, x2 + w2 / 2) - max(x1 - w1 / 2, x2 - w2 / 2))
    intersection_y = max(0, min(y1 + h1 / 2, y2 + h2 / 2) - max(y1 - h1 / 2, y2 - h2 / 2))
    intersection = intersection_x * intersection_y
    
    area1 = w1 * h1
    area2 = w2 * h2
    
    iou = intersection / (area1 + area2 - intersection)
    
    return iou


txt_files = [file for file in os.listdir('./label_dino/') if file.split('.')[-1] == 'txt']
file_paths = [
    [os.path.join('label_yolo', file), os.path.join('label_detr', file), os.path.join('label_dino', file), os.path.join('../dataset15/val2017', file)] for file in txt_files
]

t, f = 0, 0
print(f"total {len(txt_files)} dino files")
for i, file_name in enumerate(txt_files):
    file_number = file_name.split('.')[0]

    with open(file_paths[i][0], 'r') as f1, open(file_paths[i][1], 'r') as f2, open(file_paths[i][2], 'r') as f3, open(file_paths[i][3], 'r') as f4:
        
        lines_yolo = f1.readlines()
        lines_detr = f2.readlines()
        lines_dino = f3.readlines()
        lines_gt = f4.readlines()
        class_counter_yolo = Counter([line.split()[0] for line in lines_yolo])
        class_counter_detr = Counter([line.split()[0] for line in lines_detr])
        class_counter_dino = Counter([line.split()[0] for line in lines_dino])
        class_counter_gt = Counter([line.split()[0] for line in lines_gt])

        if class_counter_yolo == class_counter_dino == class_counter_detr:
            t += 1
        else:
            shutil.copy(f'./label_yolo/{file_number}.jpg', f'./hard_negative/class_diff/{file_number}.jpg')
            shutil.copy(f'./label_yolo/{file_number}.txt', f'./hard_negative/class_diff/{file_number}.txt')
            f += 1

print(t, f)    

        # detect class difference
        
        # if len(lines_yolo) > len(lines_detr):
        #     if len(lines_yolo) == len(lines_gt):
        #         yolo_true_1 += 1
        #     elif len(lines_detr) == len(lines_gt):
        #         yolo_false_1 += 1
        #     # shutil.copy(f'./label_yolo/{file_number}.jpg', f'./hard_negative/class_diff/yolo>detr/{file_number}_yolo.jpg')
        #     # shutil.copy(f'./label_detr/{file_number}.png', f'./hard_negative/class_diff/yolo>detr/{file_number}_detr.png')
        #     continue

        
        # elif len(lines_yolo) < len(lines_detr):
        #     if len(lines_detr) == len(lines_gt):
        #         yolo_false_2 += 1
        #     elif len(lines_yolo) == len(lines_gt):
        #         yolo_true_2 += 1
        #     shutil.copy(f'./label_yolo/{file_number}.jpg', f'./hard_negative/class_diff/yolo<detr/{file_number}_yolo.jpg')
        #     shutil.copy(f'./label_detr/{file_number}.png', f'./hard_negative/class_diff/yolo<detr/{file_number}_detr.png')
        #     continue

        # if len(lines_yolo) > len(lines_dino):
        #     if len(lines_yolo) == len(lines_gt):
        #         yolo_true_3 += 1
        #     elif len(lines_dino) == len(lines_gt):
        #         yolo_false_3 += 1
        #     shutil.copy(f'./label_yolo/{file_number}.jpg', f'./hard_negative/class_diff/yolo>dino/{file_number}_yolo.jpg')
        #     shutil.copy(f'./label_dino/image/{file_number}.png', f'./hard_negative/class_diff/yolo>dino/{file_number}_dino.png')
        #     continue

        # elif len(lines_yolo) < len(lines_dino):
        #     if len(lines_dino) == len(lines_gt):
        #         yolo_false_4 += 1
        #     elif len(lines_yolo) == len(lines_gt):
        #         yolo_true_4 += 1
        #     shutil.copy(f'./label_yolo/{file_number}.jpg', f'./hard_negative/class_diff/yolo<dino/{file_number}_yolo.jpg')
        #     shutil.copy(f'./label_dino/image/{file_number}.png', f'./hard_negative/class_diff/yolo<dino/{file_number}_dino.png')
        #     continue

        # detect bounding box difference
        # for line_yolo in lines_yolo:
        #     class_id_yolo, *box_yolo = parse_line(line_yolo)

        #     # matching_lines_detr = [line_detr for line_detr in lines_detr if parse_line(line_detr)[0] == class_id_yolo]
        #     matching_lines_dino = [line_dino for line_dino in lines_dino if parse_line(line_dino)[0] == class_id_yolo]

        #     # f = False
        #     # for line_detr in matching_lines_detr:
        #     #     class_id_detr, *box_detr = parse_line(line_detr)
        #     #     iou_detr = calculate_iou(box_yolo[:4], box_detr[:4])
        #     #     if iou_detr >= 0.5: # 한번이라도 맞으면 flag True로
        #     #         f = True
        #     #         break
        #     # if f == False:
        #     #     shutil.copy(f'./label_yolo/{file_number}.jpg', f'./hard_negative/bbox_diff/yolo_detr/{file_number}_yolo.jpg')
        #     #     shutil.copy(f'./label_detr/{file_number}.png', f'./hard_negative/bbox_diff/yolo_detr/{file_number}_detr.png')

        #     f = False
        #     for line_dino in matching_lines_dino:
        #         class_id_dino, *box_dino = parse_line(line_dino)
        #         iou_dino = calculate_iou(box_yolo[:4], box_dino[:4])
        #         if iou_dino >= 0.5:
        #             f = True
        #             break
        #     if f == False:
        #         print(f"{file_number} {class_id_yolo} {box_yolo} {box_dino}")
        #         shutil.copy(f'./label_yolo/{file_number}.jpg', f'./hard_negative/bbox_diff/yolo_dino/{file_number}_yolo.jpg')
        #         shutil.copy(f'./label_dino/image/{file_number}.png', f'./hard_negative/bbox_diff/yolo_dino/{file_number}_dino.png')