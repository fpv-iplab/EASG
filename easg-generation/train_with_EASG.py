import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
np.set_printoptions(precision=3)
import time
import os
import logging
import pandas as pd
import copy
from math import ceil

from dataloader.EASG import EASG, cuda_collate_fn
from lib.object_detector_EASG import detector
from lib.config import Config
from lib.AdamW import AdamW
from lib.sttran_EASG import STTran


def intersect_2d(out, gt):
    return (out[..., None] == gt.T[None, ...]).all(1)

def get_pred_triplets(mode, verb_idx, obj_indices, scores_rels, scores_objs, scores_verb, list_k, num_top_verb, num_top_rel_with, num_top_rel_no):
    triplets_with, triplets_no = [], []
    scores_with, scores_no = [], []
    if mode == 'edgecls':
        for obj_idx, scores_rel in zip(obj_indices, scores_rels):
            sorted_scores_rel = scores_rel.argsort(descending=True)
            for ri in sorted_scores_rel[:num_top_rel_with]:
                triplets_with.append((verb_idx, obj_idx, ri.item()))
                scores_with.append(scores_rel[ri].item())

            for ri in sorted_scores_rel[:ceil(max(list_k)/num_obj)]:
                triplets_no.append((verb_idx, obj_idx, ri.item()))
                scores_no.append(scores_rel[ri].item())
    elif mode == 'sgcls':
        num_top_obj_with = ceil(max(list_k)/(num_top_rel_with*num_obj))
        num_top_obj_no = ceil(max(list_k)/(num_top_rel_no*num_obj))
        for scores_obj, scores_rel in zip(scores_objs, scores_rels):
            sorted_scores_obj = scores_obj.argsort(descending=True)
            sorted_scores_rel = scores_rel.argsort(descending=True)
            for oi in sorted_scores_obj[:num_top_obj_with]:
                for ri in sorted_scores_rel[:num_top_rel_with]:
                    triplets_with.append((verb_idx, oi.item(), ri.item()))
                    scores_with.append((scores_obj[oi]+scores_rel[ri]).item())
            for oi in sorted_scores_obj[:num_top_obj_no]:
                for ri in sorted_scores_rel[:num_top_rel_no]:
                    triplets_no.append((verb_idx, oi.item(), ri.item()))
                    scores_no.append((scores_obj[oi]+scores_rel[ri]).item())
    elif mode == 'easgcls':
        num_top_obj_with = ceil(max(list_k)/(num_top_verb*num_top_rel_with*num_obj))
        num_top_obj_no = ceil(max(list_k)/(num_top_verb*num_top_rel_no*num_obj))
        for vi in scores_verb.argsort(descending=True)[:num_top_verb]:
            for scores_obj, scores_rel in zip(scores_objs, scores_rels):
                sorted_scores_obj = scores_obj.argsort(descending=True)
                sorted_scores_rel = scores_rel.argsort(descending=True)
                for oi in sorted_scores_obj[:num_top_obj_with]:
                    for ri in sorted_scores_rel[:num_top_rel_with]:
                        triplets_with.append((vi.item(), oi.item(), ri.item()))
                        scores_with.append((scores_verb[vi]+scores_obj[oi]+scores_rel[ri]).item())
                for oi in sorted_scores_obj[:num_top_obj_no]:
                    for ri in sorted_scores_rel[:num_top_rel_no]:
                        triplets_no.append((vi.item(), oi.item(), ri.item()))
                        scores_no.append((scores_verb[vi]+scores_obj[oi]+scores_rel[ri]).item())

    triplets_with = torch.tensor(triplets_with, dtype=torch.long)
    triplets_no = torch.tensor(triplets_no, dtype=torch.long)

    triplets_with = triplets_with[torch.argsort(torch.tensor(scores_with), descending=True)]
    triplets_no = triplets_no[torch.argsort(torch.tensor(scores_no), descending=True)]

    return triplets_with, triplets_no


"""------------------------------------some settings----------------------------------------"""
conf = Config()
conf.save_path = os.path.join(conf.save_path, f'train_{conf.mode}')
os.makedirs(conf.save_path, exist_ok=True)

logging.basicConfig(format='%(asctime)s.%(msecs)03d %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.DEBUG,
                    handlers=[logging.StreamHandler(),
                    logging.FileHandler(filename=os.path.join(conf.save_path, 'log'), mode='w')])

logger = logging.getLogger()

logger.info('spatial encoder layer num: {} / temporal decoder layer num: {}'.format(conf.enc_layer, conf.dec_layer))
for i in conf.args:
    logger.info(f'{i}: {conf.args[i]}')
"""-----------------------------------------------------------------------------------------"""

EASG_dataset_train = EASG("train", datasize=conf.datasize, data_path=conf.data_path)
dataloader_train = torch.utils.data.DataLoader(EASG_dataset_train, shuffle=True, num_workers=4,
                                               collate_fn=cuda_collate_fn, pin_memory=False)
EASG_dataset_test = EASG("val", datasize=conf.datasize, data_path=conf.data_path)
dataloader_test = torch.utils.data.DataLoader(EASG_dataset_test, shuffle=False, num_workers=4,
                                              collate_fn=cuda_collate_fn, pin_memory=False)

gpu_device = torch.device("cuda:0")
# freeze the detection backbone
object_detector = detector(train=True, object_classes=EASG_dataset_train.obj_classes, use_SUPPLY=True, mode=conf.mode).to(device=gpu_device)
object_detector.eval()

model = STTran(mode=conf.mode,
               obj_classes=EASG_dataset_train.obj_classes,
               verb_classes=EASG_dataset_train.verb_classes,
               edge_class_num=len(EASG_dataset_train.edge_classes),
               enc_layer_num=conf.enc_layer,
               dec_layer_num=conf.dec_layer).to(device=gpu_device)


# loss function, default Multi-label margin loss
ce_loss = nn.CrossEntropyLoss()
mlm_loss = nn.MultiLabelMarginLoss()

# optimizer
if conf.optimizer == 'adamw':
    optimizer = AdamW(model.parameters(), lr=conf.lr)
elif conf.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=conf.lr)
elif conf.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=conf.lr, momentum=0.9, weight_decay=0.01)

scheduler = ReduceLROnPlateau(optimizer, "max", patience=1, factor=0.5, verbose=True, threshold=1e-4, threshold_mode="abs", min_lr=1e-7)

# some parameters
tr = []
freq = 10

for epoch in range(1, conf.nepoch+1):
    model.train()
    object_detector.is_train = True
    train_iter = iter(dataloader_train)
    test_iter = iter(dataloader_test)
    for b in range(1, len(dataloader_train)+1):
        data = next(train_iter)

        im_data = copy.deepcopy(data[0].cuda(0))
        im_info = copy.deepcopy(data[1].cuda(0))
        gt_grounding = EASG_dataset_train.gt_groundings[data[2]]

        # prevent gradients to FasterRCNN
        with torch.no_grad():
            entry = object_detector(im_data, im_info, gt_grounding, im_all=None)

        entry['features_verb'] = copy.deepcopy(EASG_dataset_train.verb_feats[data[2]].cuda(0))

        pred = model(entry)

        edge_distribution = pred["edge_distribution"]

        losses = {}
        if conf.mode != 'edgecls':
            losses['obj_loss'] = ce_loss(pred['distribution'], pred['labels'])

        if conf.mode == 'easgcls':
            losses['verb_loss'] = ce_loss(pred['distribution_verb'], pred['labels_verb'])

        edge_label = -torch.ones([len(pred['edge']), len(EASG_dataset_train.edge_classes)], dtype=torch.long).to(device=edge_distribution.device)
        for i in range(len(pred['edge'])):
            edge_label[i, :len(pred['edge'][i])] = torch.tensor(pred['edge'][i])

        losses["edge_loss"] = mlm_loss(edge_distribution, edge_label)

        optimizer.zero_grad()
        loss = sum(losses.values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
        optimizer.step()

        tr.append(pd.Series({x: y.item() for x, y in losses.items()}))

        if b % freq == 0:
            losses_mean = pd.concat(tr[-freq:], axis=1).mean(1)
            msg = 'epoch [{:2d}/{:2d}] | batch [{:3d}/{:3d}] | edge_loss: {:.3f}'.format(epoch, conf.nepoch, b, len(dataloader_train), losses_mean['edge_loss'])
            if conf.mode != 'edgecls':
                msg += ', obj_loss: {:.3f}'.format(losses_mean['obj_loss'])
            if conf.mode == 'easgcls':
                msg += ', verb_loss: {:.3f}'.format(losses_mean['verb_loss'])

            logger.info(msg)

    torch.save({"state_dict": model.state_dict()}, os.path.join(conf.save_path, "model_{}.tar".format(epoch)))

    num_top_verb = 5
    num_top_rel_with = 1
    num_top_rel_no = 5
    list_k = [10, 20, 50]

    recall_with = {k: [] for k in list_k}
    recall_no = {k: [] for k in list_k}

    model.eval()
    object_detector.is_train = False
    with torch.no_grad():
        for b in range(1, len(dataloader_test)+1):
            data = next(test_iter)

            im_data = copy.deepcopy(data[0].cuda(0))
            im_info = copy.deepcopy(data[1].cuda(0))
            gt_grounding = EASG_dataset_test.gt_groundings[data[2]]
            video_list = EASG_dataset_test.video_list[data[2]]

            entry = object_detector(im_data, im_info, gt_grounding, im_all=None)
            entry['features_verb'] = copy.deepcopy(EASG_dataset_test.verb_feats[data[2]].cuda(0))
            pred = model(entry)

            gcid_to_f_idx = {}
            for f_idx, fname in enumerate(video_list):
                gcid = fname.split('_')[0]
                if gcid not in gcid_to_f_idx:
                    gcid_to_f_idx[gcid] = []
                gcid_to_f_idx[gcid].append(f_idx)

            gcid_to_b_idx = {}
            gcid_obj_idx_to_b_idx = {}
            b_idx = 0
            for gcid in gcid_to_f_idx:
                if gcid not in gcid_to_b_idx:
                    gcid_to_b_idx[gcid] = []

                if gcid not in gcid_obj_idx_to_b_idx:
                    gcid_obj_idx_to_b_idx[gcid] = {}

                for f_idx in gcid_to_f_idx[gcid]:
                    for g in gt_grounding[f_idx]:
                        if g['obj'] not in gcid_obj_idx_to_b_idx[gcid]:
                            gcid_obj_idx_to_b_idx[gcid][g['obj']] = []

                        gcid_obj_idx_to_b_idx[gcid][g['obj']].append(b_idx)
                        gcid_to_b_idx[gcid].append(b_idx)
                        b_idx += 1

            for gcid in gcid_to_f_idx:
                f_indices = gcid_to_f_idx[gcid]
                verb_idx = gt_grounding[f_indices[0]][0]['verb']
                obj_indices = [g['obj'] for g in gt_grounding[f_indices[0]]]
                triplets_gt = []
                for g in gt_grounding[f_indices[0]]:
                    for e in g['edge']:
                        triplets_gt.append((verb_idx, g['obj'], e))

                triplets_gt = torch.LongTensor(triplets_gt)

                num_obj = len(obj_indices)
                scores_rels = []
                scores_objs = []
                for obj_idx in obj_indices:
                    scores_rels.append(pred['edge_distribution'][gcid_obj_idx_to_b_idx[gcid][obj_idx]].mean(dim=0))
                    scores_objs.append(pred['distribution'][gcid_obj_idx_to_b_idx[gcid][obj_idx]].mean(dim=0))

                scores_rels = torch.stack(scores_rels)
                scores_objs = torch.stack(scores_objs)
                scores_verb = pred['distribution_verb'][gcid_to_b_idx[gcid]].mean(dim=0)

                triplets_with, triplets_no = get_pred_triplets(conf.mode, verb_idx, obj_indices, scores_rels, scores_objs, scores_verb, list_k, num_top_verb, num_top_rel_with, num_top_rel_no)

                out_to_gt_with = intersect_2d(triplets_gt, triplets_with)
                out_to_gt_no = intersect_2d(triplets_gt, triplets_no)

                num_gt = triplets_gt.shape[0]
                for k in list_k:
                    recall_with[k].append(out_to_gt_with[:, :k].any(dim=1).sum().item() / num_gt)
                    recall_no[k].append(out_to_gt_no[:, :k].any(dim=1).sum().item() / num_gt)

    for k in list_k:
        recall_with[k] = sum(recall_with[k]) / len(recall_with[k])*100
        recall_no[k] = sum(recall_no[k]) / len(recall_no[k])*100

    msg = 'epoch [{:2d}/{:2d}] | With: ('.format(epoch, conf.nepoch)
    for k in list_k:
        msg += '{:.1f}, '.format(recall_with[k])
    msg = msg[:-2] + ') | No: ('

    for k in list_k:
        msg += '{:.1f}, '.format(recall_no[k])
    msg = msg[:-2] + ')'

    logger.info(msg)
    logger.info("*" * 40)

    score = (recall_with[20] + recall_no[20])/2
    scheduler.step(score)
