"""This script can be used to extract roi features from the annotated bounding boxes"""
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import pickle

import torch
from detectron2.config import get_cfg
from os.path import join
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.structures import ImageList, Boxes
import cv2
import numpy as np
import os

parser = ArgumentParser()
parser.add_argument('path_to_checkpoint', type=Path)
parser.add_argument('path_to_annotations', type=Path)
parser.add_argument('path_to_images', type=Path)
parser.add_argument('path_to_output', type=Path)
parser.add_argument('--num_classes', type=int, default=306)
parser.add_argument('--fname_format', type=str, default="{clip_uid:s}_{clip_frame_number:06d}.jpg")

args = parser.parse_args()
args.path_to_output.mkdir(exist_ok=True, parents=True)

annts = {}
annts['train'] = pickle.load(open(args.path_to_annotations / 'easg_train.pkl', 'rb'))
annts['val'] = pickle.load(open(args.path_to_annotations / 'easg_val.pkl', 'rb'))

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))
cfg.MODEL.WEIGHTS = str(args.path_to_checkpoint)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes

predictor = DefaultPredictor(cfg)
model = predictor.model.eval()
pixel_mean = torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
pixel_std = torch.tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)

for split in annts:
    roi_feats_all = {}
    for graph_uid in tqdm(annts[split]):
        if graph_uid not in roi_feats_all:
            roi_feats_all[graph_uid] = {}
        clip_uid = '_'.join(graph_uid.split('_')[:-1])
        for aid in annts[split][graph_uid]['annotations']:
            if aid not in roi_feats_all[graph_uid]:
                roi_feats_all[graph_uid][aid] = []
            for annt in annts[split][graph_uid]['annotations'][aid]:
                roi_feats = {}
                for frameType in annt['bbox']:
                    fname = args.fname_format.format(clip_uid=clip_uid, clip_frame_number=annts[split][graph_uid]['clip_frame_number'][frameType])
                    img_path = str(args.path_to_images / fname)
                    if not os.path.isfile(img_path):
                        print (f'cannot find {img_path}')
                        continue
                    img = cv2.imread(img_path)
                    with torch.no_grad():
                        if predictor.input_format == 'RGB':
                            img = img[..., ::-1]
                        img = predictor.aug.get_transform(img).apply_image(img)
                        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
                        img = (img-pixel_mean)/pixel_std
                        img = img.to(cfg.MODEL.DEVICE)
                        img = ImageList.from_tensors([img], model.backbone.size_divisibility).tensor
                        H, W = img.shape[-2:]
                        boxes = torch.tensor(annt['bbox'][frameType]).to(cfg.MODEL.DEVICE)
                        boxes[:, ::2] *= W
                        boxes[:, 1::2] *= H
                        boxes[:, 2] += boxes[:, 0]
                        boxes[:, 3] += boxes[:, 1]

                        features = model.backbone(img)
                        features = [features[f] for f in cfg.MODEL.ROI_HEADS.IN_FEATURES]
                        outputs = model.roi_heads.box_pooler(features, [Boxes(boxes)]) # (N, 256, 7, 7)
                        outputs = model.roi_heads.box_head(outputs) # (N, 1024)
                        outputs = outputs.to('cpu')

                    roi_feats[frameType] = outputs

                if roi_feats:
                    roi_feats_all[graph_uid][aid].append(roi_feats)

            if len(roi_feats_all[graph_uid][aid]) == 0:
                del roi_feats_all[graph_uid][aid]

        if len(roi_feats_all[graph_uid]) == 0:
            del roi_feats_all[graph_uid]

    with open(args.path_to_output / f'roi_feats_{split}.pkl', 'wb') as f:
        pickle.dump(roi_feats_all, f)
