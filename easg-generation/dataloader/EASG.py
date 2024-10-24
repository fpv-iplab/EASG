import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import random
#from scipy.misc import imread
import cv2
import numpy as np
import pickle
import json
import os
from fasterRCNN.lib.model.utils.blob import prep_im_for_blob, im_list_to_blob

class EASG(Dataset):

    def __init__(self, split, datasize, data_path=None):

        root_path = data_path
        self.frames_path = os.path.join(root_path, 'frames/')

        self.obj_classes = ['__background__']
        with open('fasterRCNN/data/easg/objects_vocab.txt') as f:
            for l in f:
                self.obj_classes.append(l.strip('\n'))

        self.verb_feats = []
        self.verb_classes = []
        self.edge_classes = []
        self.video_list = []
        self.video_size = []
        self.gt_groundings = []

        print('-------loading annotations---------slowly-----------')

        feats = torch.load(os.path.join(root_path, 'features_verb.pt'))

        with open(os.path.join(root_path, 'EASG_unict_master_final_upd.json'), 'rb') as f:
            annts = json.load(f)

        for clip_id in annts:
            for graph in annts[clip_id]['graphs']:
                for triplet in graph['triplets']:
                    n1, e, n2 = triplet
                    if n1 == 'CW':
                        assert e == 'verb'
                        if n2 not in self.verb_classes:
                            self.verb_classes.append(n2)
                    else:
                        if ':' in n2:
                            n2 = n2.split(':')[0]

                        if n2 not in self.obj_classes:
                            continue
                        if e not in self.edge_classes:
                            self.edge_classes.append(e)

        for clip_id in annts:
            if annts[clip_id]['split'] != split: # train or val
                continue

            video_size = (annts[clip_id]['W'], annts[clip_id]['H'])

            num_frames = 0
            video = []
            feat = []
            gt_grounding = []
            for graph in annts[clip_id]['graphs']:
                graph_uid = graph['graph_uid']
                obj_to_edge = {}
                for triplet in graph['triplets']:
                    n1, e, n2 = triplet
                    if n1 == 'CW':
                        verb = n2
                    else:
                        if ':' in n2:
                            n2 = n2.split(':')[0]

                        if n2 not in self.obj_classes:
                            continue
                        if n2 not in obj_to_edge:
                            obj_to_edge[n2] = []

                        if e not in obj_to_edge[n2]:
                            obj_to_edge[n2].append(e)

                grounding_t = {}
                grounding_t['pre'] = []
                grounding_t['pnr'] = []
                grounding_t['post'] = []
                for t in ['pre', 'pnr', 'post']:
                    if t not in graph['groundings']:
                        continue

                    for n in graph['groundings'][t]:
                        if n not in obj_to_edge:
                            # Here we ignore the mismatched graphs/groundings
                            continue
                        x, y, w, h = graph['groundings'][t][n]['left'], graph['groundings'][t][n]['top'], graph['groundings'][t][n]['width'], graph['groundings'][t][n]['height']
                        bbox = np.array([x, y, x+w, y+h], dtype=np.float32)
                        grounding_t[t].append({'obj': self.obj_classes.index(n)-1,
                                               'bbox': bbox,
                                               'verb': self.verb_classes.index(verb),
                                               'edge': sorted([self.edge_classes.index(e) for e in obj_to_edge[n]])})
                        feat.append(feats[f'{graph_uid}_{clip_id}'])

                for t in ['pre', 'pnr', 'post']:
                    if not grounding_t[t]:
                        continue

                    video.append('{}/{}_{}.jpg'.format(graph_uid, clip_id, t))
                    gt_grounding.append(grounding_t[t])
                    num_frames += 1

                if num_frames >= 100:
                    self.video_list.append(video)
                    self.video_size.append(video_size)
                    self.verb_feats.append(torch.stack(feat))
                    self.gt_groundings.append(gt_grounding)
                    video = []
                    feat = []
                    gt_grounding = []
                    num_frames = 0

            if num_frames > 0:
                self.video_list.append(video)
                self.video_size.append(video_size)
                self.verb_feats.append(torch.stack(feat))
                self.gt_groundings.append(gt_grounding)

        print('There are {} videos and {} maximum number of frames'.format(len(self.video_list), max([len(v) for v in self.video_list])))
        print('--------------------finish!-------------------------')


    def __getitem__(self, index):

        frame_names = self.video_list[index]
        processed_ims = []
        im_scales = []

        for idx, name in enumerate(frame_names):
            #im = imread(os.path.join(self.frames_path, name)) # channel h,w,3
            #im = im[:, :, ::-1] # rgb -> bgr
            im = cv2.imread(os.path.join(self.frames_path, name),cv2.IMREAD_UNCHANGED)
            im, im_scale = prep_im_for_blob(im, [[[102.9801, 115.9465, 122.7717]]], 600, 1000) #cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
            im_scales.append(im_scale)
            processed_ims.append(im)

        blob = im_list_to_blob(processed_ims)
        im_info = np.array([[blob.shape[1], blob.shape[2], im_scales[0]]],dtype=np.float32)
        im_info = torch.from_numpy(im_info).repeat(blob.shape[0], 1)
        img_tensor = torch.from_numpy(blob)
        img_tensor = img_tensor.permute(0, 3, 1, 2)

        return img_tensor, im_info, index

    def __len__(self):
        return len(self.video_list)

def cuda_collate_fn(batch):
    """
    don't need to zip the tensor

    """
    return batch[0]
