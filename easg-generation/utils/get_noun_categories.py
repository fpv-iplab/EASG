import os
import json
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument('path_to_annotations', type=Path)
parser.add_argument('path_to_output_json', type=Path)

args = parser.parse_args()
args.path_to_output_json.parent.mkdir(exist_ok=True, parents=True)

train = json.load(open(args.path_to_annotations / 'fho_scod_train.json'))
val = json.load(open(args.path_to_annotations / 'fho_scod_val.json'))

def get_noun_categories(annotations):
    noun_categories = []
    for ann in annotations:
        for frameType in ['pre', 'pnr', 'post']:
            for bbox in ann[f'{frameType}_frame']['bbox']:
                noun = bbox['object_type']
                if noun in ['left_hand', 'right_hand']:
                    # merge all the hand labels into a single class
                    noun = 'hand_(finger,_hand,_palm,_thumb)'
                elif 'structured_noun' in bbox:
                    snoun = bbox['structured_noun']
                    if noun == 'object_of_change' and snoun is None:
                        continue

                    if snoun is not None:
                        noun = snoun

                noun_categories.append(noun)
    return set(noun_categories)

noun_categories_train = get_noun_categories(train['clips'])
noun_categories_val = get_noun_categories(val['clips'])
noun_categories = sorted(noun_categories_train.intersection(noun_categories_val))

noun_categories = [{'id': idx, 'name': noun} for idx, noun in enumerate(noun_categories)]

with open(args.path_to_output_json, 'w') as f:
    json.dump(noun_categories, f)
