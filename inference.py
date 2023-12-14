import os
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter

import torch
import torchvision.transforms as T
import torch.nn.functional as F
torch.set_grad_enabled(False)

import argparse

from models_detr import build_model
from util_detr import args_parser

from models_dino.main import build_model_main
from util_dino.slconfig import SLConfig
from datasets import build_dataset
from util_dino.visualizer import COCOVisualizer
from util_dino import box_ops

threshold = 0.3

CLASSES = ['background', 'Face_Shield', 'Fire_Gloves', 'Gas_Mask_Full', 'Gas_Mask_Half', 
           'Glasses', 'Hazmat_Coat', 'Lab_Coat', 'Latex_Gloves', 'Mask', 'No_Gloves', 
           'Normal_Shoes', 'Safety_Glasses', 'Safety_Hat', 'Sandal', 'Work_Gloves']

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]



# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def detect(im, model, transform, filename, plot=True):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    if img.shape[-2] > 1600 or img.shape[-1] > 1600:

        # Calculate the aspect ratio
        aspect_ratio = img.shape[-1] / img.shape[-2]

        # Calculate the target size based on the maximum size and aspect ratio
        if img.shape[-2] > img.shape[-1]:
            target_height = 1600
            target_width = int(target_height * aspect_ratio)
        else:
            target_width = 1600
            target_height = int(target_width / aspect_ratio)

        # Resize the tensor using interpolate
        resized_img = F.interpolate(img, size=(target_height, target_width), mode='bilinear', align_corners=False)
        img = resized_img

    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'
    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.7

    detections = []
    prob = probas[keep]
    boxes = outputs['pred_boxes'][0, keep]

    file_contents = ""
    for p, (x1, y1, x2, y2) in zip(prob, boxes.tolist()):
        cl = p.argmax()
        inference_class = CLASSES[cl]
        conf = p[cl]
        detections.append([x1, y1, x2, y2, conf, inference_class])
        file_contents += f"{cl - 1} {x1} {y1} {x2} {y2} {conf}\n"

    # save inference result
    file_path = os.path.join('label_detr', filename.split('.')[0] + '.txt')
    with open(file_path, 'w') as file:
        file.write(file_contents)

    if plot:
        plt.figure(figsize=(16,10))
        plt.imshow(im)
        ax = plt.gca()
        # convert boxes from [0; 1] to image scales
        bboxes_scaled = rescale_bboxes(boxes, im.size)
        for p, (xmin, ymin, xmax, ymax), c in zip(prob, bboxes_scaled.tolist(), COLORS * 100):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=2))
            cl = p.argmax()
            text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
            ax.text(xmin, ymin, text, fontsize=12, color=c)
            # ax.text(xmin, ymin, text, fontsize=10, color=c,
            #         bbox=dict(facecolor='white', alpha=0.5))
        plt.axis('off')
        plt.savefig(f'./label_detr/image/{filename}.png')
        plt.close()

    return detections


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[args_parser.get_args_parser()])
    args = parser.parse_args()

    if not os.path.exists(args.inference_image_dir):
        raise Exception(f"❌ {args.inference_image_dir} does not exist")
    os.makedirs('label_detr', exist_ok=True)
    os.makedirs('label_dino', exist_ok=True)

    # load detr
    detr, criterion, postprocessors = build_model(args)
    state_dict = torch.load(args.detr_weights)
    detr.load_state_dict(state_dict['model'])
    detr.eval()
    print("✅ load DETR")

    # load dino
    model_config_path = "config/DINO/DINO_4scale.py"
    model_checkpoint_path = args.dino_weights

    args_dino = SLConfig.fromfile(model_config_path)
    args_dino.device = 'cuda:0'
    args_dino.dataset_file = 'coco'
    args_dino.coco_path = '../dataset15/'
    args_dino.fix_size = False

    dino, criterion, postprocessors = build_model_main(args_dino)
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
    dino.load_state_dict(checkpoint['model'])
    _ = dino.eval()
    print("✅ load DINO")

    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Inference from YOLO
    os.system("git clone https://github.com/WongKinYiu/yolov7")
    os.system(f"python ./yolov7/detect.py --weights {args.yolo_weights} --conf 0.25 --source {args.inference_image_dir} --project label_yolo --name '' --exist-ok --save-txt --save-conf")

    # Inference from DETR
    # directory_path = './label_detr/image/'
    # os.makedirs(directory_path, exist_ok=True)
    image_paths = [os.path.join(args.inference_image_dir, file) 
                for file in os.listdir(args.inference_image_dir)
                if os.path.splitext(file)[1].lower() in ('.png', '.jpg', '.jpeg')]
    total_image_count = len(os.listdir(args.inference_image_dir))
    for idx, file_path in enumerate(image_paths):
        with Image.open(file_path) as img:
            filename = os.path.splitext(file_path)[0].split('/')[-1]
            detections_detr = detect(img, detr, transform, filename, plot=True) # inference from detr
            cnt = Counter([d[-1] for d in detections_detr])
            print(f"👀 (DETR {idx}/{len(image_paths)}) Detect {filename}: {[f'{k}: {v}' for k, v in cnt.items()]}")

    # Inference from DINO
    dataset_val = build_dataset(image_set='val', args=args_dino)
    vslzr = COCOVisualizer()

    for idx, (image, targets) in enumerate(dataset_val):
        filename = dataset_val.get_file_name(idx)
        output = dino.cuda()(image[None].cuda())
        output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]
        scores = output['scores']
        labels = output['labels']
        boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
        select_mask = scores > threshold
        box_label = [CLASSES[int(item)] for item in labels[select_mask]]
        cnt = Counter(box_label)
        print(f"🦕 (DINO {idx}/{len(dataset_val)}) Detect {filename}: {[f'{k}: {v}' for k, v in cnt.items()]}")

        # save inference result
        file_path = os.path.join('label_dino', filename.split('.')[0] + '.txt')
        with open(file_path, 'w') as file:
            for i in range(len(box_label)):
                file.write(f"{labels[select_mask][i] - 1} {boxes[i][0]} {boxes[i][1]} {boxes[i][2]} {boxes[i][3]} {scores[select_mask][i]}\n")
                vslzr = COCOVisualizer()
                pred_dict = {
                    'boxes': boxes[select_mask],
                    'size': targets['size'],
                    'box_label': box_label,
                    'image_id': targets['image_id'],
                    'scores': scores[select_mask]
                }
                vslzr.visualize(image, pred_dict, savedir="./label_dino/image", filename=filename.split('.')[0])

        # compare with ground truth
        # gt_file_path = os.path.join(args.inference_image_dir, filename.split('.')[0] + '.txt')
        # with open(gt_file_path, 'r') as gt:
        #     gt_lines = gt.readlines()
        #     gt_label = [CLASSES[int(line.split()[0]) + 1] for line in gt_lines]
        #     gt_cnt = Counter(gt_label)
        #     if gt_cnt != cnt:
        #         # bounding box 그려서 저장
        #         # thershold = 0.3 # set a thershold
        #         vslzr = COCOVisualizer()
        #         pred_dict = {
        #             'boxes': boxes[select_mask],
        #             'size': targets['size'],
        #             'box_label': box_label,
        #             'image_id': targets['image_id'],
        #             'scores': scores[select_mask]
        #         }
        #         vslzr.visualize(image, pred_dict, savedir="detr_wrong_inference", filename=filename.split('.')[0])
        #         print(f"    *Wrong Inference (DINO {idx}/{len(dataset_val)}) Ground Truth {filename}:  {[f'{k}: {v}' for k, v in gt_cnt.items()]}")
