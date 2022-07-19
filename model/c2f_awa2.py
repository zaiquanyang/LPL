import os
import tqdm
import time
import sys
import copy
import random
import platform
import math

import torch as t
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn import datasets, linear_model, discriminant_analysis, model_selection

# from .Dirichlet_Net import Dirichlet_Net
from model.resnet import resnet101_features
from model.modules import *

from utils.tool import *

class Coarse_Align(nn.Module):

    def __init__(self,
                 args,
                 seen_att,
                 unseen_att,
                 vision_dim,
                 attr_dim,
                 ):
        super(Coarse_Align, self).__init__()
        self.args = args
        self.all_att = t.cat([seen_att, unseen_att], dim=0).to(args.device)
        self.seen_att = t.Tensor(seen_att).float().to(args.device)
        self.unseen_att = t.Tensor(unseen_att).float().to(args.device)
        self.scale_cls = 20.

        self.coarse_mix = Coarse_GMIX(args=args, ways=args.ways, shots=args.shots)
        self.sparse_softmax = Sparsemax(dim=1)

        self.backbone = resnet101_features(pretrained=True)
        if not args.FINE_TUNE:
            for p in self.backbone.parameters():
                p.requires_grad = False
        else:
            print('fine-tune the model!')
            for p in self.backbone.parameters():
                p.requires_grad = True

        self.V2S = nn.Linear(in_features=vision_dim, out_features=attr_dim, bias=False)
        self.CE_loss = nn.CrossEntropyLoss()


    def forward(self, x_img, x_attr, x_label):

        x_fea_map = self.backbone(x_img)
        x_fea = F.avg_pool2d(x_fea_map, kernel_size=(x_fea_map.size(-2), x_fea_map.size(-1))).squeeze()
        x_fea = x_fea - x_fea.mean(dim=-1).unsqueeze(dim=-1)

        if self.training:
            seen_att_norm = F.normalize(self.seen_att, p=2, dim=-1, eps=1e-12)

            pre_attr = self.V2S(x_fea)
            pre_attr_norm = F.normalize(pre_attr, p=2, dim=-1, eps=1e-12)
            pre_seen_scores = self.scale_cls * t.matmul(pre_attr_norm, seen_att_norm.t())
            v2s_loss = self.CE_loss(pre_seen_scores, x_label)

            return v2s_loss

        else:
            attr_norm = F.normalize(self.all_att, p=2, dim=-1, eps=1e-12)
            pre_attr = self.V2S(x_fea)
            pre_attr_norm = F.normalize(pre_attr, p=2, dim=-1, eps=1e-12)

            pre_scores = self.scale_cls * t.matmul(pre_attr_norm, attr_norm.t())

            return pre_scores


class Fine_Align(nn.Module):

    def __init__(self,
                 args,
                 seen_class,
                 seen_att,
                 unseen_att,
                 vision_dim,
                 attr_dim,
                 h_dim,
                 V2S_W,
                 Class_Center,
                 ):
        super(Fine_Align, self).__init__()
        self.args = args
        self.enc_shared_dim = attr_dim
        self.enc_exclusive_dim = attr_dim
        self.attr_dim = attr_dim
        self.h_dim = h_dim
        self.vision_dim = vision_dim
        self.seen_class = seen_class if seen_class is not None else None

        self.scale_cls = nn.Parameter(t.FloatTensor(1).fill_(args.scale_factor).to(args.device), requires_grad=True)
        self.bias = nn.Parameter(t.FloatTensor(1).fill_(0).to(args.device), requires_grad=True)
        self.V2S_W = V2S_W
        self.Class_Center = Class_Center
        self.all_att = t.cat([seen_att, unseen_att], dim=0).to(args.device)

        # S2V
        self.fc_s_1 = nn.Linear(in_features=attr_dim, out_features=self.h_dim, bias=True)
        self.fc_s_2 = nn.Linear(in_features=self.h_dim, out_features=vision_dim, bias=True)

        # GMIX
        self.GMIX = GMIX(args=args, ways=args.sub_ways, shots=args.sub_shots)

        if self.args.init_s2v==0:
            print('use pytorch default init.')

        if self.args.init_s2v == 1:
            print('init s2v by normal')
            self.fc_s_1.weight.data.normal_(0, 0.02)
            self.fc_s_1.bias.data.fill_(0)
            self.fc_s_2.weight.data.normal_(0, 0.02)
            self.fc_s_2.bias.data.fill_(0)

        if self.args.init_s2v == 2:
            print('init s2v by truncated_normal')
            self.fc_s_1.weight.data = truncated_normal_(self.fc_s_1.weight.data, mean=0, std=0.02)
            self.fc_s_1.bias.data.fill_(0)
            self.fc_s_2.weight.data = truncated_normal_(self.fc_s_2.weight.data, mean=0, std=0.02)
            self.fc_s_2.bias.data.fill_(0)


        self.CE_loss = nn.CrossEntropyLoss()
        self.MSE_loss = nn.MSELoss()


    def forward(self, x_fea, x_attr, x_meta_label, x_label=None, show=False, epoch=0):

        x_fea = F.normalize(x_fea, p=2, dim=-1)
        ways, shots = self.args.sub_ways, int(len(x_fea)/self.args.sub_ways)

        meta_task_x_label = x_meta_label

        # Adaptive attention
        # print(x_attr.size())


        if not self.training:
            x_attr = self.all_att
            x_attr = F.normalize(x_attr, dim=-1, p=2)
            x_cls_prototype = F.relu(self.fc_s_2(F.relu(self.fc_s_1(x_attr))))
            x_cls_prototype_norm = F.normalize(x_cls_prototype, p=2, eps=1e-12, dim=-1)

            x_fea_norm = F.normalize(x_fea, dim=-1, eps=1e-12)

            fine_pre_socres = self.scale_cls * t.mm(x_fea_norm, x_cls_prototype_norm.t())  # n x cls_num

            return fine_pre_socres

        else:
            # x_attr = (nn.Sigmoid()(
            #     t.mm((x_fea - self.Class_Center[x_label]), self.V2S_W.T) / 10.0) - 0.5 + 1.0) * x_attr
            # x_fea = x_fea.reshape(ways, shots, -1).transpose(1, 0)  # shots x ways x vision_dim
            # x_attr = x_attr.reshape(ways, shots, -1).transpose(1, 0)  # shots x ways x attr_dim
            gmix_feat, gmix_attr = self.GMIX(x_fea, x_attr, x_label, show=show, epoch=epoch)

            idx = t.tensor(list(range(0, ways * shots, shots))).to(self.args.device)
            gmix_attr = t.index_select(gmix_attr, 0, idx).float()

            gmix_attr_norm = F.normalize(gmix_attr, p=2, dim=-1)
            gmix_class_prototype = F.relu(self.fc_s_2(F.relu(self.fc_s_1(gmix_attr_norm))))
            gmix_class_prototype_norm = F.normalize(gmix_class_prototype, p=2, eps=1e-12, dim=1)

            gmix_fea_norm = F.normalize(gmix_feat, dim=-1, eps=1e-12)
            fine_pre_socres_s = self.scale_cls * t.mm(gmix_fea_norm, gmix_class_prototype_norm.t())
            fine_cls_ce_loss = self.CE_loss(fine_pre_socres_s, meta_task_x_label)

            return fine_cls_ce_loss

