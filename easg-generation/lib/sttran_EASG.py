"""
Let's get the relationships yo
"""

import numpy as np
import torch
import torch.nn as nn

from lib.word_vectors import obj_edge_vectors, verb_edge_vectors
from lib.transformer import transformer
from lib.fpn.box_utils import center_size
from fasterRCNN.lib.model.roi_layers import ROIAlign, nms


class ObjectClassifier(nn.Module):
    """
    Module for computing the object contexts and edge contexts
    """

    def __init__(self, mode='edgecls', obj_classes=None):
        super(ObjectClassifier, self).__init__()
        self.mode = mode
        self.obj_classes = obj_classes

        embed_vecs = obj_edge_vectors(self.obj_classes, wv_type='glove.6B', wv_dir='data', wv_dim=200)
        self.obj_embed = nn.Embedding(len(self.obj_classes), 200)
        self.obj_embed.weight.data = embed_vecs.clone()

        # This probably doesn't help it much
        self.pos_embed = nn.Sequential(nn.BatchNorm1d(4, momentum=0.01 / 10.0),
                                       nn.Linear(4, 128),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.1))
        self.obj_dim = 2048
        self.decoder_lin = nn.Sequential(nn.Linear(self.obj_dim + 200 + 128, 1024),
                                         nn.BatchNorm1d(1024),
                                         nn.ReLU(),
                                         nn.Linear(1024, len(self.obj_classes)))

    def forward(self, entry):
        if self.mode  == 'edgecls':
            entry['pred_labels'] = entry['labels']
            entry['distribution'] = torch.nn.functional.one_hot(entry['pred_labels'], len(self.obj_classes)).to(torch.float32)
        else:
            obj_embed = entry['distribution'] @ self.obj_embed.weight
            pos_embed = self.pos_embed(center_size(entry['boxes'][:, 1:]))
            obj_features = torch.cat((entry['features'], obj_embed, pos_embed), 1)
            if self.training:
                entry['distribution'] = self.decoder_lin(obj_features)
                entry['pred_labels'] = entry['labels']
            else:
                entry['distribution'] = self.decoder_lin(obj_features)

                entry['distribution'] = torch.softmax(entry['distribution'], dim=1)
                entry['pred_labels'] = torch.max(entry['distribution'], dim=1)[1]

        return entry


class ActionClassifier(nn.Module):
    def __init__(self, mode='edgecls', verb_classes=None):
        super(ActionClassifier, self).__init__()
        self.mode = mode
        self.verb_classes = verb_classes

        self.decoder_lin = nn.Sequential(nn.Linear(2304, 1024),
                                         nn.BatchNorm1d(1024),
                                         nn.ReLU(),
                                         nn.Linear(1024, len(self.verb_classes)))

    def forward(self, entry):
        if self.mode != 'easgcls':
            entry['pred_labels_verb'] = entry['labels_verb']
            entry['distribution_verb'] = torch.nn.functional.one_hot(entry['pred_labels_verb'], len(self.verb_classes)).to(torch.float32)
        else:
            if self.training:
                entry['distribution_verb'] = self.decoder_lin(entry['features_verb'])
                entry['pred_labels_verb'] = entry['labels_verb']
            else:
                entry['distribution_verb'] = self.decoder_lin(entry['features_verb'])

                entry['distribution_verb'] = torch.softmax(entry['distribution_verb'], dim=1)
                entry['pred_labels_verb'] = torch.max(entry['distribution_verb'], dim=1)[1]

        return entry


class STTran(nn.Module):

    def __init__(self, mode='edgecls',
                 obj_classes=None, verb_classes=None, edge_class_num=None,
                 enc_layer_num=None, dec_layer_num=None):

        super(STTran, self).__init__()
        self.obj_classes = [cls for cls in obj_classes if cls != '__background__']
        self.verb_classes = verb_classes
        self.edge_class_num = edge_class_num
        assert mode in ('easgcls', 'sgcls', 'edgecls')
        self.mode = mode

        self.object_classifier = ObjectClassifier(mode=self.mode, obj_classes=self.obj_classes)
        self.action_classifier = ActionClassifier(mode=self.mode, verb_classes=self.verb_classes)

        ###################################
        self.conv = nn.Sequential(
            nn.Conv2d(2, 256 //2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256//2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256 // 2, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, momentum=0.01),
        )
        self.obj_fc = nn.Linear(2048, 512)
        self.verb_fc = nn.Linear(2304, 512)

        embed_vecs = obj_edge_vectors(self.obj_classes, wv_type='glove.6B', wv_dir='data', wv_dim=200)
        self.obj_embed = nn.Embedding(len(self.obj_classes), 200)
        self.obj_embed.weight.data = embed_vecs.clone()

        embed_vecs = verb_edge_vectors(self.verb_classes, wv_type='glove.6B', wv_dir='data', wv_dim=200)
        self.verb_embed = nn.Embedding(len(self.verb_classes), 200)
        self.verb_embed.weight.data = embed_vecs.clone()

        self.glocal_transformer = transformer(enc_layer_num=enc_layer_num, dec_layer_num=dec_layer_num, embed_dim=1424, nhead=8,
                                              dim_feedforward=2048, dropout=0.1, mode='latter')

        self.rel_compress = nn.Linear(1424, self.edge_class_num)

    def forward(self, entry):

        entry = self.object_classifier(entry)
        entry = self.action_classifier(entry)

        # visual part
        obj_rep = entry['features']
        obj_rep = self.obj_fc(obj_rep)
        verb_rep = entry['features_verb']
        verb_rep = self.verb_fc(verb_rep)
        x_visual = torch.cat((obj_rep, verb_rep), 1)

        # semantic part
        obj_class = entry['pred_labels']
        obj_emb = self.obj_embed(obj_class)
        verb_class = entry['pred_labels_verb']
        verb_emb = self.verb_embed(verb_class)
        x_semantic = torch.cat((obj_emb, verb_emb), 1)

        rel_features = torch.cat((x_visual, x_semantic), dim=1)
        # Spatial-Temporal Transformer
        global_output, global_attention_weights, local_attention_weights = self.glocal_transformer(features=rel_features, im_idx=entry['im_idx'])

        entry["edge_distribution"] = self.rel_compress(global_output)
        entry["edge_distribution"] = torch.sigmoid(entry["edge_distribution"])

        return entry

