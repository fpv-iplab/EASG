"""This script can be used to extract object detections from the annotated frames"""
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import pickle
import json

from detectron2.config import get_cfg
from os.path import join
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import cv2
import numpy as np
import os

parser = ArgumentParser()
parser.add_argument('path_to_checkpoint', type=Path)
parser.add_argument('path_to_annotations', type=Path)
parser.add_argument('path_to_images', type=Path)
parser.add_argument('path_to_output_json', type=Path)
parser.add_argument('--fname_format', type=str, default="{video_uid:s}_{frame_number:07d}.jpg")
parser.add_argument('--num_classes', type=int, default=None)

args = parser.parse_args()
args.path_to_output_json.parent.mkdir(exist_ok=True, parents=True)

train = json.load(open(args.path_to_annotations / 'fho_scod_train.json'))
val = json.load(open(args.path_to_annotations / 'fho_scod_val.json'))
test = json.load(open(args.path_to_annotations / 'fho_scod_test_unannotated.json'))
noun_categories = json.load(open('noun_categories.json'))

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))
cfg.MODEL.WEIGHTS = str(args.path_to_checkpoint)
if args.num_classes is not None:
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
else:
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(noun_categories)

predictor = DefaultPredictor(cfg)

detections = {}

for anns in [train, val, test]:
    for ann in tqdm(anns['clips']):
        video_id = ann["video_uid"]
        for frameType in ['pre', 'pnr', 'post']:
            fname = args.fname_format.format(video_uid=video_id, frame_number=ann[f'{frameType}_frame']["frame_number"])
            uid = os.path.splitext(fname)[0]
            img_path = str(args.path_to_images / fname)
            if not os.path.isfile(img_path):
                print (f'cannot find {img_path}')
                continue
            img = cv2.imread(img_path)
            outputs = predictor(img)['instances'].to('cpu')

            dets = []
            for box, score, noun in zip(outputs.pred_boxes.tensor, outputs.scores, outputs.pred_classes):
                box = box.tolist()
                box = [float(x) for x in box]
                score = score.item()
                noun = noun.item()
                dets.append(
                    {
                        'box': box,
                        'score': score,
                        'noun_category_id': noun
                    }
                )
            detections[uid] = dets

json.dump(detections, open(args.path_to_output_json, 'w'))
