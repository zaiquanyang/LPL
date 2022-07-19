import os
import sys
import json
from urllib.request import proxy_bypass

import numpy as np
import torch as t
from torch.utils.data import Dataset
import torch.nn.functional as F



class Data_loader_Mixup_Feature(Dataset):
    def __init__(self, args, feats, atts, b_labels, ways, shots, s_labels=None, maml=False, prob=None):
        # self.ways = ways * 2
        self.shots = shots
        self.args = args
        self.maml = maml

        self.feats = t.Tensor(feats).float()
        self.atts = atts
        self.labels = b_labels # b_labels
        self.classes = np.unique(b_labels)        # 1, 3, 5, 6, 11...., 199
        # self.VWSW_CLASS = self.args.VWSW_CLASS_DICT
        # assert np.max(self.classes) == 199

        # calculate the class prototype
        self.class_prototype = t.zeros(args.classes_num, 2048).cuda()
        self.gbl_class_prototype = t.zeros(args.classes_num, 2048).cuda()
        
        self.prob = prob


    def __getitem__(self, idx, ways=None):

        if ways is None:
            ways = self.args.sub_ways
            shots = self.args.sub_shots
        else:
            ways = ways
            shots =  self.args.sub_shots
        select_feats, select_atts, select_meta_labels, select_labels = M_ways_N_shots(ways=ways,
                                                                                    shots=shots,
                                                                                    classes=self.classes,
                                                                                    labels=self.labels,
                                                                                    feats=self.feats,
                                                                                    atts=self.atts,
                                                                                    prob=self.prob)

        real_feat = select_feats[:ways*shots, :]
        real_attr = select_atts[:ways * shots, :]
        real_meta_label = select_meta_labels[:ways * shots]
        real_label = select_labels[:ways * shots]


        return real_feat, real_attr,  real_meta_label, real_label


def M_ways_N_shots(ways, shots, classes, labels, feats, atts, prob):

    is_first = True
    select_feats = []
    select_atts = []
    select_labels = t.LongTensor(ways*2 * shots)
    select_meta_labels = t.LongTensor(ways*2 * shots)  # ways x shots
    # if prob is not None:
    #     prob = np.array(prob)
    #     select_classes = np.random.choice(list(classes), ways, False, p=prob)
    # else:
    #     select_classes = np.random.choice(list(classes), ways, False)
    select_classes = np.random.choice(list(classes), ways, False)
    # s0 = np.random.choice(list(classes['Sparrow']), 10, False)
    # s1 = np.random.choice(list(classes['Warbler']), 10, False)
    # s2 = np.random.choice(list(classes['Vireo']), 5, False)
    # s3 = np.random.choice(list(classes['Gull']), 5, False)
    # select_classes = np.concatenate([s0, s1], axis=0)
    # print(select_classes.shape)

    # assert 1==2

    for i in range(len(select_classes)):

        idx = (labels == select_classes[i]).nonzero()[0]
        # select_instances = np.random.choice(idx, shots, False)
        try:
            select_instances = np.random.choice(idx, shots, False)
        except:
            print('this class sample is too less to sample shots')
            sys.exit()

        for j in range(shots):
            feat = feats[select_instances[j], :]
            att = atts[select_instances[j], :]

            feat = feat.unsqueeze(0)
            att = att.unsqueeze(0)

            if is_first:
                is_first = False
                select_feats = feat
                select_atts = att
            else:
                select_feats = t.cat((select_feats, feat), 0)
                select_atts = t.cat((select_atts, att), 0)
            try:
                select_meta_labels[i * shots + j] = i
                select_labels[i * shots + j] = select_classes[i].item()
            except:
                print('error line 151')
                sys.exit()

    return select_feats, select_atts, select_meta_labels, select_labels


# class Data_loader_Syn_CLSWGAN(Dataset):
#     def __init__(self, args, feats, atts, b_labels):
#         self.args = args

#         self.feats = t.Tensor(feats).float()
#         self.atts = atts
#         self.labels = b_labels # b_labels
#         self.classes = np.unique(b_labels)        # 1, 3, 5, 6, 11...., 199

#         # calculate the class prototype
#         self.class_prototype = t.zeros(args.classes_num, 2048).cuda()
#         self.gbl_class_prototype = t.zeros(args.classes_num, 2048).cuda()


#     def __getitem__(self, idx):

#         ways = self.args.sub_ways
#         shots = self.args.sub_shots

#         select_feats, select_atts, select_meta_labels, select_labels = M_ways_N_shots(ways=ways,
#                                                                                       shots=shots,
#                                                                                       classes=self.classes,
#                                                                                       labels=self.labels,
#                                                                                       feats=self.feats,
#                                                                                       atts=self.atts)

#         real_feat = select_feats[:ways*shots, :]
#         real_attr = select_atts[:ways*shots, :]
#         real_meta_label = select_meta_labels[:ways * shots]
#         real_label = select_labels[:ways * shots]


#         return real_feat, real_attr,  real_meta_label, real_label
    
    