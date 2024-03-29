import os
import torch
from torch.nn import Module, Linear, CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import Dataset
from argparse import ArgumentParser
from pathlib import Path
import pickle
import random
import logging
from math import ceil


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('path_to_annotations', type=Path)
    parser.add_argument('path_to_data', type=Path)
    parser.add_argument('path_to_output', type=Path)
    parser.add_argument('--gpu_id', type=int, default=0, help='which gpu to run')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--sch_param', type=int, default=10, help='parameter for lr scheduler')
    parser.add_argument('--num_epoch', type=int, default=100, help='total number of epochs')
    parser.add_argument('--random_guess', action='store_true', help='for random guessing')
    args = parser.parse_args()
    return args


class EASGData(Dataset):
    def __init__(self, path_annts, path_data, split, verbs, objs, rels):
        self.path_annts = path_annts
        self.path_data = path_data
        self.split = split
        with open(path_annts / f'easg_{split}.pkl', 'rb') as f:
            annts = pickle.load(f)

        with open(path_data / f'roi_feats_{split}.pkl', 'rb') as f:
            roi_feats = pickle.load(f)

        clip_feats = torch.load(path_data / 'verb_features.pt')

        """
        graph:
            dict['verb_idx']: index of its verb
            dict['clip_feat']: 2304-D clip-wise feature vector
            dict['objs']: dict of obj_idx
                dict[obj_idx]: dict
                    dict['obj_feat']: 1024-D ROI feature vector
                    dict['rels_vec']: multi-hot vector of relationships

        graph_batch:
            dict['verb_idx']: index of its verb
            dict['clip_feat']: 2304-D clip-wise feature vector
            dict['obj_indices']: batched version of obj_idx
            dict['obj_feats']: batched version of obj_feat
            dict['rels_vecs']: batched version of rels_vec
            dict['triplets']: all the triplets consisting of (verb, obj, rel)
        """
        graphs = []
        for graph_uid in annts:
            graph = {}
            for aid in annts[graph_uid]['annotations']:
                for i, annt in enumerate(annts[graph_uid]['annotations'][aid]):
                    verb_idx = verbs.index(annt['verb'])
                    if verb_idx not in graph:
                        graph[verb_idx] = {}
                        graph[verb_idx]['verb_idx'] = verb_idx
                        graph[verb_idx]['objs'] = {}

                    graph[verb_idx]['clip_feat'] = clip_feats[aid]

                    obj_idx = objs.index(annt['obj'])
                    if obj_idx not in graph[verb_idx]['objs']:
                        graph[verb_idx]['objs'][obj_idx] = {}
                        graph[verb_idx]['objs'][obj_idx]['obj_feat'] = torch.zeros((0, 1024), dtype=torch.float32)
                        graph[verb_idx]['objs'][obj_idx]['rels_vec'] = torch.zeros(len(rels), dtype=torch.float32)

                    rel_idx = rels.index(annt['rel'])
                    graph[verb_idx]['objs'][obj_idx]['rels_vec'][rel_idx] = 1

                    for frameType in roi_feats[graph_uid][aid][i]:
                        graph[verb_idx]['objs'][obj_idx]['obj_feat'] = torch.cat((graph[verb_idx]['objs'][obj_idx]['obj_feat'], roi_feats[graph_uid][aid][i][frameType]), dim=0)

            for verb_idx in graph:
                for obj_idx in graph[verb_idx]['objs']:
                    graph[verb_idx]['objs'][obj_idx]['obj_feat'] = graph[verb_idx]['objs'][obj_idx]['obj_feat'].mean(dim=0)

                graphs.append(graph[verb_idx])

        self.graphs = []
        for graph in graphs:
            graph_batch = {}
            verb_idx = graph['verb_idx']
            graph_batch['verb_idx'] = torch.tensor([verb_idx], dtype=torch.long)
            graph_batch['clip_feat'] = graph['clip_feat']
            graph_batch['obj_indices'] = torch.zeros(0, dtype=torch.long)
            graph_batch['obj_feats'] = torch.zeros((0, 1024), dtype=torch.float32)
            graph_batch['rels_vecs'] = torch.zeros((0, len(rels)), dtype=torch.float32)
            graph_batch['triplets'] = torch.zeros((0, 3), dtype=torch.long)

            for obj_idx in graph['objs']:
                graph_batch['obj_indices'] = torch.cat((graph_batch['obj_indices'], torch.tensor([obj_idx], dtype=torch.long)), dim=0)
                graph_batch['obj_feats'] = torch.cat((graph_batch['obj_feats'], graph['objs'][obj_idx]['obj_feat'].unsqueeze(0)), dim=0)

                rels_vec = graph['objs'][obj_idx]['rels_vec']
                graph_batch['rels_vecs'] = torch.cat((graph_batch['rels_vecs'], rels_vec.unsqueeze(0)), dim=0)

                triplets = []
                for rel_idx in torch.where(rels_vec)[0]:
                    triplets.append((verb_idx, obj_idx, rel_idx.item()))
                graph_batch['triplets'] = torch.cat((graph_batch['triplets'], torch.tensor(triplets, dtype=torch.long)), dim=0)

            self.graphs.append(graph_batch)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


class EASG(Module):
    def __init__(self, verbs, objs, rels, dim_clip_feat=2304, dim_obj_feat=1024):
        super().__init__()
        self.fc_verb = Linear(dim_clip_feat, len(verbs))
        self.fc_objs = Linear(dim_obj_feat, len(objs))
        self.fc_rels = Linear(dim_clip_feat+dim_obj_feat, len(rels))

    def forward(self, clip_feat, obj_feats):
        out_verb = self.fc_verb(clip_feat)
        out_objs = self.fc_objs(obj_feats)

        clip_feat_expanded = clip_feat.expand(obj_feats.shape[0], -1)
        out_rels = self.fc_rels(torch.cat((clip_feat_expanded, obj_feats), dim=1))
        return out_verb, out_objs, out_rels


def intersect_2d(out, gt):
    return (out[..., None] == gt.T[None, ...]).all(1)


def evaluation(dataset_val, model, device, args):
    model.eval()

    num_top_verb = 5
    num_top_rel_with = 1
    num_top_rel_no = 5
    list_k = [10, 20, 50]

    recall_predcls_with = {k: [] for k in list_k}
    recall_predcls_no = {k: [] for k in list_k}
    recall_sgcls_with = {k: [] for k in list_k}
    recall_sgcls_no = {k: [] for k in list_k}
    recall_easgcls_with = {k: [] for k in list_k}
    recall_easgcls_no = {k: [] for k in list_k}
    for idx in range(len(dataset_val)):
        graph = dataset_val[idx]
        clip_feat = graph['clip_feat'].unsqueeze(0).to(device)
        obj_feats = graph['obj_feats'].to(device)

        with torch.no_grad():
            out_verb, out_objs, out_rels = model(clip_feat, obj_feats)
            scores_verb = out_verb[0].detach().cpu().softmax(dim=0)
            scores_objs = out_objs.detach().cpu().softmax(dim=1)
            scores_rels = out_rels.detach().cpu().sigmoid()

        if args.random_guess:
            scores_verb = torch.rand(scores_verb.shape)
            scores_verb /= scores_verb.sum()
            scores_objs = torch.rand(scores_objs.shape)
            scores_objs /= scores_objs.sum()
            scores_rels = torch.rand(scores_rels.shape)

        verb_idx = graph['verb_idx']
        obj_indices = graph['obj_indices']
        rels_vecs = graph['rels_vecs']
        triplets_gt = graph['triplets']
        num_obj = obj_indices.shape[0]

        # make triplets for precls
        triplets_pred_with = []
        scores_pred_with = []
        triplets_pred_no = []
        scores_pred_no = []
        for obj_idx, scores_rel in zip(obj_indices, scores_rels):
            sorted_scores_rel = scores_rel.argsort(descending=True)
            for ri in sorted_scores_rel[:num_top_rel_with]:
                triplets_pred_with.append((verb_idx.item(), obj_idx.item(), ri.item()))
                scores_pred_with.append(scores_rel[ri].item())

            for ri in sorted_scores_rel[:ceil(max(list_k)/num_obj)]:
                triplets_pred_no.append((verb_idx.item(), obj_idx.item(), ri.item()))
                scores_pred_no.append(scores_rel[ri].item())

        # make triplets for sgcls
        triplets_sg_with = []
        scores_sg_with = []
        triplets_sg_no = []
        scores_sg_no = []
        num_top_obj_with = ceil(max(list_k)/(num_top_rel_with*num_obj))
        num_top_obj_no = ceil(max(list_k)/(num_top_rel_no*num_obj))
        for scores_obj, scores_rel in zip(scores_objs, scores_rels):
            sorted_scores_obj = scores_obj.argsort(descending=True)
            sorted_scores_rel = scores_rel.argsort(descending=True)
            for oi in sorted_scores_obj[:num_top_obj_with]:
                for ri in sorted_scores_rel[:num_top_rel_with]:
                    triplets_sg_with.append((verb_idx.item(), oi.item(), ri.item()))
                    scores_sg_with.append((scores_obj[oi]+scores_rel[ri]).item())
            for oi in sorted_scores_obj[:num_top_obj_no]:
                for ri in sorted_scores_rel[:num_top_rel_no]:
                    triplets_sg_no.append((verb_idx.item(), oi.item(), ri.item()))
                    scores_sg_no.append((scores_obj[oi]+scores_rel[ri]).item())

        # make triplets for easgcls
        triplets_easg_with = []
        scores_easg_with = []
        triplets_easg_no = []
        scores_easg_no = []
        num_top_obj_with = ceil(max(list_k)/(num_top_verb*num_top_rel_with*num_obj))
        num_top_obj_no = ceil(max(list_k)/(num_top_verb*num_top_rel_no*num_obj))
        for vi in scores_verb.argsort(descending=True)[:num_top_verb]:
            for scores_obj, scores_rel in zip(scores_objs, scores_rels):
                sorted_scores_obj = scores_obj.argsort(descending=True)
                sorted_scores_rel = scores_rel.argsort(descending=True)
                for oi in sorted_scores_obj[:num_top_obj_with]:
                    for ri in sorted_scores_rel[:num_top_rel_with]:
                        triplets_easg_with.append((vi.item(), oi.item(), ri.item()))
                        scores_easg_with.append((scores_verb[vi]+scores_obj[oi]+scores_rel[ri]).item())
                for oi in sorted_scores_obj[:num_top_obj_no]:
                    for ri in sorted_scores_rel[:num_top_rel_no]:
                        triplets_easg_no.append((vi.item(), oi.item(), ri.item()))
                        scores_easg_no.append((scores_verb[vi]+scores_obj[oi]+scores_rel[ri]).item())

        triplets_pred_with = torch.tensor(triplets_pred_with, dtype=torch.long)
        triplets_pred_no = torch.tensor(triplets_pred_no, dtype=torch.long)
        triplets_sg_with = torch.tensor(triplets_sg_with, dtype=torch.long)
        triplets_sg_no = torch.tensor(triplets_sg_no, dtype=torch.long)
        triplets_easg_with = torch.tensor(triplets_easg_with, dtype=torch.long)
        triplets_easg_no = torch.tensor(triplets_easg_no, dtype=torch.long)

        # sort the triplets using the averaged scores
        triplets_pred_with = triplets_pred_with[torch.argsort(torch.tensor(scores_pred_with), descending=True)]
        triplets_pred_no = triplets_pred_no[torch.argsort(torch.tensor(scores_pred_no), descending=True)]
        triplets_sg_with = triplets_sg_with[torch.argsort(torch.tensor(scores_sg_with), descending=True)]
        triplets_sg_no = triplets_sg_no[torch.argsort(torch.tensor(scores_sg_no), descending=True)]
        triplets_easg_with = triplets_easg_with[torch.argsort(torch.tensor(scores_easg_with), descending=True)]
        triplets_easg_no = triplets_easg_no[torch.argsort(torch.tensor(scores_easg_no), descending=True)]

        out_to_gt_pred_with = intersect_2d(triplets_gt, triplets_pred_with)
        out_to_gt_pred_no = intersect_2d(triplets_gt, triplets_pred_no)
        out_to_gt_sg_with = intersect_2d(triplets_gt, triplets_sg_with)
        out_to_gt_sg_no = intersect_2d(triplets_gt, triplets_sg_no)
        out_to_gt_easg_with = intersect_2d(triplets_gt, triplets_easg_with)
        out_to_gt_easg_no = intersect_2d(triplets_gt, triplets_easg_no)

        num_gt = triplets_gt.shape[0]
        for k in list_k:
            recall_predcls_with[k].append(out_to_gt_pred_with[:, :k].any(dim=1).sum().item() / num_gt)
            recall_predcls_no[k].append(out_to_gt_pred_no[:, :k].any(dim=1).sum().item() / num_gt)
            recall_sgcls_with[k].append(out_to_gt_sg_with[:, :k].any(dim=1).sum().item() / num_gt)
            recall_sgcls_no[k].append(out_to_gt_sg_no[:, :k].any(dim=1).sum().item() / num_gt)
            recall_easgcls_with[k].append(out_to_gt_easg_with[:, :k].any(dim=1).sum().item() / num_gt)
            recall_easgcls_no[k].append(out_to_gt_easg_no[:, :k].any(dim=1).sum().item() / num_gt)

    for k in list_k:
        recall_predcls_with[k] = sum(recall_predcls_with[k]) / len(recall_predcls_with[k])*100
        recall_predcls_no[k] = sum(recall_predcls_no[k]) / len(recall_predcls_no[k])*100
        recall_sgcls_with[k] = sum(recall_sgcls_with[k]) / len(recall_sgcls_with[k])*100
        recall_sgcls_no[k] = sum(recall_sgcls_no[k]) / len(recall_sgcls_no[k])*100
        recall_easgcls_with[k] = sum(recall_easgcls_with[k]) / len(recall_easgcls_with[k])*100
        recall_easgcls_no[k] = sum(recall_easgcls_no[k]) / len(recall_easgcls_no[k])*100

    return recall_predcls_with, recall_predcls_no, recall_sgcls_with, recall_sgcls_no, recall_easgcls_with, recall_easgcls_no


def main():
    args = parse_args()
    args.path_to_output.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.DEBUG,
                        handlers=[logging.StreamHandler(), logging.FileHandler(filename=args.path_to_output / 'log', mode='w')],
                        )
    logger = logging.getLogger()

    with open(args.path_to_annotations / 'verbs.txt') as f:
        verbs = [l.strip() for l in f.readlines()]

    with open(args.path_to_annotations / 'objects.txt') as f:
        objs = [l.strip() for l in f.readlines()]

    with open(args.path_to_annotations / 'relationships.txt') as f:
        rels = [l.strip() for l in f.readlines()]

    dataset_train = EASGData(args.path_to_annotations, args.path_to_data, 'train', verbs, objs, rels)
    dataset_val = EASGData(args.path_to_annotations, args.path_to_data, 'val', verbs, objs, rels)

    device = ('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    model = EASG(verbs, objs, rels)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.sch_param)
    criterion = CrossEntropyLoss()
    criterion_rel = BCEWithLogitsLoss()

    for epoch in range(1, args.num_epoch+1):
        model.train()

        list_index = list(range(len(dataset_train)))
        random.shuffle(list_index)
        loss_train = 0.
        for idx in list_index:
            if args.random_guess:
                break

            optimizer.zero_grad()

            graph = dataset_train[idx]
            clip_feat = graph['clip_feat'].unsqueeze(0).to(device)
            obj_feats = graph['obj_feats'].to(device)
            out_verb, out_objs, out_rels = model(clip_feat, obj_feats)

            verb_idx = graph['verb_idx'].to(device)
            obj_indices = graph['obj_indices'].to(device)
            rels_vecs = graph['rels_vecs'].to(device)
            loss = criterion(out_verb, verb_idx) + criterion(out_objs, obj_indices) + criterion_rel(out_rels, rels_vecs)
            loss.backward()
            loss_train += loss.item()
            optimizer.step()

        scheduler.step()

        loss_train /= len(dataset_train)
        recall_predcls_with, recall_predcls_no, recall_sgcls_with, recall_sgcls_no, recall_easgcls_with, recall_easgcls_no = evaluation(dataset_val, model, device, args)
        logger.info(f'Epoch [{epoch:03d}|{args.num_epoch:03d}] loss_train: {loss_train:.4f}, with: [({recall_predcls_with[10]:.2f}, {recall_predcls_with[20]:.2f}, {recall_predcls_with[50]:.2f}), ({recall_sgcls_with[10]:.2f}, {recall_sgcls_with[20]:.2f}, {recall_sgcls_with[50]:.2f}), ({recall_easgcls_with[10]:.2f}, {recall_easgcls_with[20]:.2f}, {recall_easgcls_with[50]:.2f})], no: [({recall_predcls_no[10]:.2f}, {recall_predcls_no[20]:.2f}, {recall_predcls_no[50]:.2f}), ({recall_sgcls_no[10]:.2f}, {recall_sgcls_no[20]:.2f}, {recall_sgcls_no[50]:.2f}), ({recall_easgcls_no[10]:.2f}, {recall_easgcls_no[20]:.2f}, {recall_easgcls_no[50]:.2f})]')


if __name__ == "__main__":
    main()
