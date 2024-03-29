from argparse import ArgumentParser
import json
from pathlib import Path
import numpy as np
import cv2
import multiprocessing
from tqdm import tqdm
import os
from ego4d_forecasting.datasets.short_term_anticipation import PyAVVideoReader

parser = ArgumentParser()
parser.add_argument('path_to_annotations', type=Path)
parser.add_argument('path_to_videos', type=Path)
parser.add_argument('path_to_output', type=Path)
parser.add_argument('--fname_format', type=str, default="{clip_uid:s}_{clip_frame_number:06d}.jpg")
parser.add_argument('--clips', action='store_true')
parser.add_argument('--num_split', type=int, default=10)
parser.add_argument('--split', type=int, default=0)

NUM_FRAME_PER_BATCH = 1000
args = parser.parse_args()
args.path_to_output.mkdir(exist_ok=True, parents=True)

train = json.load(open(args.path_to_annotations / 'fho_scod_train.json'))
val = json.load(open(args.path_to_annotations / 'fho_scod_val.json'))
test = json.load(open(args.path_to_annotations / 'fho_scod_test_unannotated.json'))

names = []
clip_ids = []
frame_numbers = []

for ann in [train, val, test]:
    for x in ann['clips']:
        clip_id = x["clip_uid"]
        for frameType in ['pre', 'pnr', 'post']:
            fname = args.fname_format.format(clip_uid=clip_id, clip_frame_number=x[f'{frameType}_frame']["clip_frame_number"])
            names.append(fname)
            clip_ids.append(x['clip_uid'])
            frame_numbers.append(x[f'{frameType}_frame']['clip_frame_number'])

if args.split > 0:
    print (f'Processing split {args.split} out of {args.num_split}')

    unique_cids = sorted(set(clip_ids))
    num_split = args.num_split
    size_split = len(unique_cids) // num_split
    if args.split < num_split:
        unique_cids = unique_cids[size_split*(args.split-1):size_split*(args.split)]
    elif args.split == num_split:
        unique_cids = unique_cids[size_split*(args.split-1):]

    selected = []
    for idx, cid in enumerate(clip_ids):
        if cid in unique_cids:
            selected.append(idx)

    names = [names[i] for i in selected]
    clip_ids = [clip_ids[i] for i in selected]
    frame_numbers = [frame_numbers[i] for i in selected]
else:
    print ('Processing all annotations')

print(f"Found {len(set(names))} frames to extract") # 1363204 for all / 1361338 for all_vid

#missing = []
#for idx, im in enumerate(names):
#    if not os.path.isfile(args.path_to_output / im):
#        missing.append(idx)
#print (f"Number of missing idx: {len(missing)}")
#if len(missing)==0:
#    exit(0)

#print(f"Skipping {len(names)-len(missing)} frames already extracted")
#names = [names[i] for i in missing]
#clip_ids = [clip_ids[i] for i in missing]
#frame_numbers = [frame_numbers[i] for i in missing]
#if len(names)==0:
#    exit(0)


dict_cid_vf = {}
for cid, fn in zip(clip_ids, frame_numbers):
    fn = int(fn)
    if cid not in dict_cid_vf:
        dict_cid_vf[cid] = []
    dict_cid_vf[cid].append(fn)

"""
def process_video(cid, frames, path_to_videos, path_to_output, fname_format):
    vr = PyAVVideoReader(str(path_to_videos / (cid+'.mp4')))

    num_batch = len(frames) // NUM_FRAME_PER_BATCH
    if len(frames) % NUM_FRAME_PER_BATCH:
        num_batch += 1

    for b in range(num_batch):
        if b != num_batch-1:
            frames_b = frames[b*NUM_FRAME_PER_BATCH:(b+1)*NUM_FRAME_PER_BATCH]
        else:
            frames_b = frames[b*NUM_FRAME_PER_BATCH:]

        video_frames = vr[frames_b]

        for vf, fn in zip(video_frames, frames_b):
            imname = str(path_to_output / fname_format.format(clip_uid=cid, clip_frame_number=fn))
            cv2.imwrite(imname, vf)
"""

def process_video(cid, frames, path_to_videos, path_to_output, fname_format):
    vr = PyAVVideoReader(str(path_to_videos / (cid+'.mp4')))
    if vr.num_frames > 0:
        frames = [fn for fn in frames if fn < vr.num_frames]

    video_frames = vr[frames]
    for vf, fn in zip(video_frames, frames):
        imname = str(path_to_output / fname_format.format(clip_uid=cid, clip_frame_number=fn))
        cv2.imwrite(imname, vf)

clip_ids = sorted(set(clip_ids))
for cid in tqdm(clip_ids):
    process_video(cid, sorted(set(dict_cid_vf[cid])), args.path_to_videos, args.path_to_output, args.fname_format)
