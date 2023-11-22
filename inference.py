import os, sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

import torch
import torchvision.transforms as T
torch.set_grad_enabled(False)

import argparse

from models_detr import build_model
from models_dino.main import build_model_main

from util_dino.slconfig import SLConfig
from datasets import build_dataset
from util_dino.visualizer import COCOVisualizer
from util_dino import box_ops

threshold = 0.3

CLASSES = ['background', 'Face', 'Face_Shield', 'Fire_Gloves', 'Gas_Mask_Full', 'Gas_Mask_Half', 
           'Glasses', 'Hazmat_Coat', 'Lab_Coat', 'Latex_Gloves', 'Mask', 'No_Gloves', 
           'Normal_Shoes', 'Safety_Glasses', 'Safety_Hat', 'Sandal', 'Work_Gloves']

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
 
    parser.add_argument('--detr_weights', type=str)
    parser.add_argument('--dino_weights', type=str)
    parser.add_argument('--inference_image_save', type=bool, default=False,
                        help="True if you want to save the inference image")
    parser.add_argument('--inference_image_dir', type=str,
                        help="Path to the image directory to inference")

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

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


def detect(im, model, transform, filename, plot=False):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
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
        file_contents += f"{cl - 1} {x1} {y1} {x2} {y2}\n"

    # save inference result
    file_path = os.path.join(args.inference_image_dir, 'label_detr', filename.split('.')[0] + '.txt')
    with open(file_path, 'w') as file:
        file.write(file_contents)
        # file.write(f"{cl} {x1} {y1} {x2} {y2}\n")

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
        plt.savefig(f'inference_{filename}.png')
        plt.close()

    return detections


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if not os.path.exists(args.inference_image_dir):
        raise Exception(f"❌ {args.inference_image_dir} does not exist")
    os.makedirs(args.inference_image_dir + '/label_detr', exist_ok=True)
    os.makedirs(args.inference_image_dir + '/label_dino', exist_ok=True)

    # load detr
    detr, criterion, postprocessors = build_model(args)
    # detr.to('cpu')
    state_dict = torch.load(args.detr_weights)
    detr.load_state_dict(state_dict['model'])
    detr.eval()
    print("✅ load DETR")

    # load dino model
    model_config_path = "config/DINO/DINO_4scale.py"
    model_checkpoint_path = args.dino_weights

    args_dino = SLConfig.fromfile(model_config_path)
    args_dino.device = 'cpu' 
    args_dino.dataset_file = 'coco'
    args_dino.coco_path = '../훈련셋/실험복/' # 수정
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

    # Inference from DINO
    dataset_val = build_dataset(image_set='val', args=args_dino)

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
        file_path = os.path.join(args.inference_image_dir, 'label_dino', filename.split('.')[0] + '.txt')
        with open(file_path, 'w') as file:
            for i in range(len(box_label)):
                file.write(f"{labels[select_mask][i] - 1} {boxes[i][0]} {boxes[i][1]} {boxes[i][2]} {boxes[i][3]}\n")

    # Inference from DETR
    image_paths = [os.path.join(args.inference_image_dir, file) 
                for file in os.listdir(args.inference_image_dir)
                if os.path.splitext(file)[1].lower() in ('.png', '.jpg', '.jpeg')]    
    total_image_count = len(os.listdir(args.inference_image_dir))
    for idx, file_path in enumerate(image_paths):
        with Image.open(file_path) as img:
            filename = os.path.splitext(file_path)[0].split('/')[-1]
            detections_detr = detect(img, detr, transform, filename, args.inference_image_save) # inference from detr
            cnt = Counter([d[-1] for d in detections_detr])
            print(f"👀 (DETR {idx}/{len(image_paths)}) Detect {filename}: {[f'{k}: {v}' for k, v in cnt.items()]}")