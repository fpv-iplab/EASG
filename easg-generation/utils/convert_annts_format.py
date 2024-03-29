import os
import glob
import json
import pickle
import torch
import numpy as np


def read_annt_file(annt_file):
    """
    Read annotation file in json format
    """
    with open(annt_file, 'r') as f:
        data = json.load(f)

    return data


def read_ego4d_annts(annt_file):
    """
    Read the Ego4D annotation file and return its clip annotations
    Return: List of dicts
        dict['split']
        dict['clip_uid']
        dict['pre_frame']: dict
            dict['frame_number']
            dict['clip_frame_number']
            dict['width']
            dict['height']
            dict['bbox']: list of dicts
                dict['object_type']
                dict['structured_noun']
                dict['instance_number']
                dict['bbox']: dict
                    dict['x']
                    dict['y']
                    dict['width']
                    dict['height']
        dict['pnr_frame']: same as pre_frame
        dict['post_frame']: same as pre_frame
    """
    data = read_annt_file(annt_file)
    split = data['split']
    for clip in data['clips']:
        clip['split'] = split

    return data['clips']


def process_bboxes(data, frameType):
    """
    Process the bounding box annotations
    Return: Dict
        dict['split']
        dict['width']
        dict['height']
        dict['bbox']: dict of noun
            dict[noun]: list of normalized (x, y, width, height)
    """
    split = data['split']
    data = data[f'{frameType}_frame']

    data_bbox = {}
    data_bbox['split'] = split
    data_bbox['clip_frame_number'] = data['clip_frame_number']
    W = data['width']
    H = data['height']
    data_bbox['bbox'] = {}
    for bbox in data['bbox']:
        noun = bbox['object_type']
        if noun not in ['left_hand', 'right_hand'] and 'structured_noun' in bbox:
            snoun = bbox['structured_noun']
            if noun == 'object_of_change' and snoun is None:
                continue

            if snoun is not None:
                noun = snoun

        if noun not in data_bbox['bbox']:
            data_bbox['bbox'][noun] = []

        x = bbox['bbox']['x']/W
        y = bbox['bbox']['y']/H
        width = bbox['bbox']['width']/W
        height = bbox['bbox']['height']/H
        data_bbox['bbox'][noun].append((x, y, width, height))

    return data_bbox


def preprocess_from_list_to_dict(list_data):
    """
    Preprocess the Ego4D annotations from a list to a dictionary for better access
    Return: Dict of clip_uid
        dict[clip_uid]: dict of pnrFrameNumber
            dict[pnrFrameNumber]: dict of 'pre', 'pnr', 'post'
                dict['pre']: dict
                    dict['split']
                    dict['bbox']: dict of noun
                        dict[noun]: list of dicts
                            dict['x']
                            dict['x']
                            dict['width']
                            dict['height']
                dict['pnr']: same as dict['pre']
                dict['post']: same as dict['pre']
    """
    dict_data = {}
    for data in list_data:
        clip_uid = data.pop('clip_uid')
        if clip_uid not in dict_data:
            dict_data[clip_uid] = {}

        pnrFrameNumber = data['pnr_frame']['clip_frame_number']
        if pnrFrameNumber not in dict_data[clip_uid]:
            dict_data[clip_uid][pnrFrameNumber] = {}

        for frameType in ['pre', 'pnr', 'post']:
            dict_data[clip_uid][pnrFrameNumber][frameType] = process_bboxes(data, frameType)

    return dict_data


def get_matched_noun(dobj, annt_nouns):
    """
    Get a dobj corresponding to Ego4D's annotation
    """
    matched_noun = None
    for noun in annt_nouns:
        if dobj in noun:
            matched_noun = noun
            break

    return matched_noun


if __name__ == "__main__":
    """
    Convert the annotations to the same format as Action Genome
    """

    file_annts_ego4d_train = '/datasets/ego4d/v1/annotations/fho_scod_train.json'
    file_annts_ego4d_val = '/datasets/ego4d/v1/annotations/fho_scod_val.json'

    file_feats = '/datasets/ego4d/EASG_files/verb_features.pt'
    path_annts = '/datasets/ego4d/EASG_files/SG_all_first_round'
    path_new_annts = '/datasets/ego4d/EASG_files/annts_in_new_format'
    os.makedirs(path_new_annts, exist_ok=True)


    data_ego4d = read_ego4d_annts(file_annts_ego4d_train) + read_ego4d_annts(file_annts_ego4d_val)
    data_ego4d = preprocess_from_list_to_dict(data_ego4d)
    feats = torch.load(file_feats)

    new_annts_train = {}
    new_annts_val = {}
    """
    New set of annotations in AG format: Dict of graph_uid
        dict[graph_uid]: dict
            dict['clip_frame_number']: dict of 'pre', 'pnr', 'post'
            dict['annotations']: dict of aid
                dict[aid]: list of dicts
                    dict['obj']: dobj or indir_obj
                    dict['bbox']: dict of a subset of {'pre', 'pnr', 'post'}
                        dict['pnr']: normalized (x, y, w, h)
                    dict['verb']
                    dict['rel']: 'dobj', 'with', 'in', ...
    """

    num_annts_with_no_verb = 0
    num_new_annts = 0
    annt_files = sorted(glob.glob(f'{path_annts}/*/*.json'))
    for annt_file in annt_files:
        aid = os.path.basename(os.path.dirname(annt_file))
        data = read_annt_file(annt_file)
        annts = json.loads(data['annotationData']['content'])
        clip_uid = annts['clip_uid']
        pnrFrameNumber = int(annts['pnrFrameNumber'])

        if aid not in feats:
            print ('no corresponding feats')
            continue

        annts_ego4d = data_ego4d[clip_uid][pnrFrameNumber]
        split = annts_ego4d['pnr']['split']
        assert split == annts_ego4d['pre']['split'] and split == annts_ego4d['post']['split']
        if split == 'train':
            new_annts = new_annts_train
        elif split == 'val':
            new_annts = new_annts_val

        graph_uid = f'{clip_uid}_{pnrFrameNumber:06d}'
        if graph_uid not in new_annts:
            new_annts[graph_uid] = {}
            new_annts[graph_uid]['annotations'] = {}
            new_annts[graph_uid]['clip_frame_number'] = {}
            for frameType in ['pre', 'pnr', 'post']:
                new_annts[graph_uid]['clip_frame_number'][frameType] = annts_ego4d[frameType]['clip_frame_number']

        if aid not in new_annts[graph_uid]['annotations']:
            new_annts[graph_uid]['annotations'][aid] = []

        if 'verb' in annts:
            verb = annts['verb']
            dobj = annts['dobj']
            if ':' in dobj:
                dobj = dobj.split(':')[0]
            newverb = annts['newverb'] if 'newverb' in annts and annts['newverb'] is not None else None
            newnoun = annts['newnoun'] if 'newnoun' in annts and annts['newnoun'] is not None else None
            #if newverb not in ['none', None] and verb != newverb:
            #    if newnoun not in ['none', None] and dobj != newnoun:
            #        print (f'verb: {verb}, newverb: {newverb}, dobj: {dobj}, newnoun: {newnoun}')
            #    else:
            #        print (f'verb: {verb}, newverb: {newverb}')
        else:
            verb = annts['newverb']
            dobj = annts['newnoun']

        W, H = float(annts['imageWidth']), float(annts['imageHeight'])

        # If bounding box information of dobj is found in Ego4D's annotations, add it to new_annts
        num_bbox = 0
        bbox_dict = {}
        for frameType in ['pre', 'pnr', 'post']:
            annt_noun_bbox = annts_ego4d[frameType]['bbox']
            matched_noun = get_matched_noun(dobj, annt_noun_bbox.keys())
            if matched_noun:
                bbox_dict[frameType] = []
                for bbox in annt_noun_bbox[matched_noun]:
                    bbox_dict[frameType].append(bbox)
                    num_bbox += 1
                assert len(bbox_dict[frameType]) > 0

        if num_bbox:
            new_annts[graph_uid]['annotations'][aid].append({'obj': dobj, 'bbox': bbox_dict, 'verb': verb, 'rel': 'dobj'})

        roles_objects = json.loads(annts['annotations'])
        for role_obj in roles_objects:
            if len(role_obj['frames']):
                # If new annotation for indir_obj is provided, add it to new_annts
                bbox_dict = {}
                rel = role_obj['role']
                for frame in role_obj['frames']:
                    frameType = frame['frameType']
                    bbs = frame['bbs']
                    x = bbs['left']/W
                    y = bbs['top']/H
                    width = bbs['width']/W
                    height = bbs['height']/H
                    indir_obj = bbs['object']
                    if ':' in indir_obj:
                        #if indir_obj.split(':')[1] != '0':
                        #    print (indir_obj)
                        indir_obj = indir_obj.split(':')[0]

                    if indir_obj not in bbox_dict:
                        bbox_dict[indir_obj] = {}

                    if frameType not in bbox_dict[indir_obj]:
                        bbox_dict[indir_obj][frameType] = []

                    bbox_dict[indir_obj][frameType].append((x, y, width, height))

                for indir_obj in bbox_dict:
                    new_annts[graph_uid]['annotations'][aid].append({'obj': indir_obj, 'bbox': bbox_dict[indir_obj], 'verb': verb, 'rel': rel})
            else:
                # Default annotation is 'with both_hands'
                rel = 'with'
                for frameType in ['pre', 'pnr', 'post']:
                    num_bbox = 0
                    bbox_dict = {}
                    bbox_dict[frameType] = []
                    for indir_obj in ['left_hand', 'right_hand', 'hand_(finger,_hand,_palm,_thumb)']:
                        matched_noun = get_matched_noun(indir_obj, annt_noun_bbox.keys())
                        if matched_noun:
                            for bbox in annt_noun_bbox[matched_noun]:
                                bbox_dict[frameType].append(bbox)
                                num_bbox += 1

                    if num_bbox:
                        new_annts[graph_uid]['annotations'][aid].append({'obj': 'both_hands', 'bbox': bbox_dict, 'verb': verb, 'rel': rel})

        if len(new_annts[graph_uid]['annotations'][aid]) == 0:
            del new_annts[graph_uid]['annotations'][aid]

        if len(new_annts[graph_uid]['annotations']) == 0:
            del new_annts[graph_uid]

    """
    Save the new annotations
    """
    list_obj = []
    list_verb = []
    list_rel = []
    for split, new_annts in zip(['train', 'val'], [new_annts_train, new_annts_val]):
        for graph_uid in new_annts:
            assert len(new_annts[graph_uid]['annotations']) > 0
            for aid in new_annts[graph_uid]['annotations']:
                assert len(new_annts[graph_uid]['annotations'][aid]) > 0

        for graph_uid in new_annts:
            for aid in new_annts[graph_uid]['annotations']:
                for i in range(len(new_annts[graph_uid]['annotations'][aid])):
                    for frameType in new_annts[graph_uid]['annotations'][aid][i]['bbox']:
                        new_annts[graph_uid]['annotations'][aid][i]['bbox'][frameType] = np.array(new_annts[graph_uid]['annotations'][aid][i]['bbox'][frameType], dtype=np.float32)

        with open(os.path.join(path_new_annts, f'easg_{split}.pkl'), 'wb') as f:
            pickle.dump(new_annts, f)

        for graph_uid in new_annts:
            for aid in new_annts[graph_uid]['annotations']:
                for annt in new_annts[graph_uid]['annotations'][aid]:
                    list_obj.append(annt['obj'])
                    list_verb.append(annt['verb'])
                    list_rel.append(annt['rel'])

    with open(os.path.join(path_new_annts, f'objects.txt'), 'w') as f:
        for obj in sorted(set(list_obj)):
            f.write(obj + '\n')

    with open(os.path.join(path_new_annts, f'verbs.txt'), 'w') as f:
        for verb in sorted(set(list_verb)):
            f.write(verb + '\n')

    with open(os.path.join(path_new_annts, f'relationships.txt'), 'w') as f:
        for rel in sorted(set(list_rel)):
            f.write(rel + '\n')
