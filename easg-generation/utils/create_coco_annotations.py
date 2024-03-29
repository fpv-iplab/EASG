"""This script creates a dataset to train an object detector in COCO format, so that it can be used with Detectron2"""

from argparse import ArgumentParser
from pathlib import Path
import json
from datetime import date
from tqdm import tqdm
import os
from detectron2.structures import BoxMode

parser = ArgumentParser()
parser.add_argument('path_to_annotations', type=Path)
parser.add_argument('path_to_noun_categories', type=Path)
parser.add_argument('path_to_images', type=Path)
parser.add_argument('path_to_output_json', type=Path)
parser.add_argument('--fname_format', type=str, default="{clip_uid:s}_{clip_frame_number:06d}.jpg")

args = parser.parse_args()
args.path_to_output_json.parent.mkdir(exist_ok=True, parents=True)

labels = json.load(open(args.path_to_annotations))['clips']
noun_categories = json.load(open(args.path_to_noun_categories))
idx_to_obj = {x['id']: x['name'] for x in noun_categories}
obj_to_idx = {v : k for k, v in idx_to_obj.items()}

progressive_image_id = 0
progressive_annotation_id = 0
filenames_to_ids = {}

images = []
annotations = []

# Filter out the annotations having nouns that we won't use
for i in range(len(labels)-1, -1, -1):
    ann = labels[i]
    clip_id = ann["clip_uid"]
    for frameType in ['pre', 'pnr', 'post']:
        fname = args.fname_format.format(clip_uid=clip_id, clip_frame_number=ann[f'{frameType}_frame']["clip_frame_number"])
        for j in range(len(ann[f'{frameType}_frame']['bbox'])-1, -1, -1):
            obj = ann[f'{frameType}_frame']['bbox'][j]
            noun = obj['object_type']
            if noun in ['left_hand', 'right_hand']:
                # merge all the hand labels into a single class
                noun = 'hand_(finger,_hand,_palm,_thumb)'
            elif 'structured_noun' in obj:
                snoun = obj['structured_noun']
                if snoun is not None:
                    noun = snoun

            if noun not in obj_to_idx:
                del labels[i][f'{frameType}_frame']['bbox'][j]

        if len(labels[i][f'{frameType}_frame']['bbox']) == 0:
            del labels[i][f'{frameType}_frame']


for ann in tqdm(labels):
    clip_id = ann["clip_uid"]
    for frameType in ['pre', 'pnr', 'post']:
        if f'{frameType}_frame' not in ann:
            continue

        fname = args.fname_format.format(clip_uid=clip_id, clip_frame_number=ann[f'{frameType}_frame']["clip_frame_number"])
        if not os.path.isfile(args.path_to_images / fname):
            print ('no image file:', str(args.path_to_images / fname))
            continue

        if fname not in filenames_to_ids:
            filenames_to_ids[fname] = progressive_image_id
            images.append({
                "file_name": fname,
                "width": ann[f'{frameType}_frame']['width'],
                "height": ann[f'{frameType}_frame']['height'],
                "id": progressive_image_id
            })
            progressive_image_id += 1

        for obj in ann[f'{frameType}_frame']['bbox']:
            noun = obj['object_type']
            if noun in ['left_hand', 'right_hand']:
                # merge all the hand labels into a single class
                noun = 'hand_(finger,_hand,_palm,_thumb)'
            elif 'structured_noun' in obj:
                snoun = obj['structured_noun']
                if snoun is not None:
                    noun = snoun

            bbox = obj['bbox']
            annotations.append({
                "segmentation": [],
                "area": bbox['width']*bbox['height'],
                "iscrowd": 0,
                "image_id": filenames_to_ids[fname],
                "bbox": [bbox['x'], bbox['y'], bbox['width'], bbox['height']],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": obj_to_idx[noun],
                "id": progressive_annotation_id
            })
            progressive_annotation_id += 1

info = {
    "description" : "Egocentric Action Scene Graph Annotations in COCO format",
    "version": "1.0",
    "date_created": str(date.today())
}

categories = [{
    "supercategory": "hand" if "_hand" in v else "object",
    "id": k,
    "name": v
} for k, v in idx_to_obj.items()]

coco_annotations = {
    "info": info,
    "images": images,
    "annotations": annotations,
    "categories": categories
}

with open(args.path_to_output_json, 'w') as f:
    json.dump(coco_annotations, f)
