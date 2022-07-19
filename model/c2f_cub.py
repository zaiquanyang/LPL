from cProfile import label

import torch
from .resnet import resnet101_features, save_grad
from .modules import *
from utils.tool import *
import torch.nn.functional as F
from model.wasserstein import get_CAM, Wasserstein_loss
from model.grl_layer import WarmStartGradientReverseLayer
import math, copy
import scipy.io as sio

# from data.mix_up_tool import  Data_loader_Syn_CLSWGAN


class Activation_Func(nn.Module):
    def __init__(self, args):
        super(Activation_Func, self).__init__()
        self.args = args
        self.alpha = nn.Parameter(t.FloatTensor(1).fill_(self.args.alpha).to(args.device), requires_grad=True)

    def forward(self, x):
        return 1. / (1. + t.exp(-1. * self.args.alpha * x)) - self.args.center
        # return 2 * (1. / (1. + t.exp(-1. * self.args.alpha * x)) - 1.)


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
        # weight = t.FloatTensor(1, seen_att.size(-1)).fill_(1.0)
        # self.channel_weight = nn.Parameter(weight, requires_grad=True)
        # self.V2S = nn.Sequential(nn.Linear(in_features=vision_dim, out_features=attr_dim, bias=False),
        #                          nn.LeakyReLU(negative_slope=self.args.negative_slope))
        # self.V2S = nn.Sequential(nn.Linear(in_features=vision_dim, out_features=attr_dim, bias=False),
        #                          nn.Sigmoid())
        # self.V2S = nn.Sequential(nn.Linear(in_features=vision_dim, out_features=attr_dim, bias=False),
        #                          Activation_Func(args))
        self.V2S = nn.Sequential(nn.Linear(in_features=vision_dim, out_features=attr_dim, bias=False))
        # self.S2V = nn.Linear(in_features=attr_dim, out_features=vision_dim, bias=False)
        # self.ADL = Attr_Decouple_Layer(attr_dim=self.all_att.size(-1), static=False, scale=1.0)     # 在 1.0的基础上更新学习
        # self.ADL = Attr_Decouple_Layer(attr_dim=self.all_att.size(-1), static=True, scale=2.0)        # 静态 2.0

        # self.Drop = nn.Dropout(p=0.4)
        # self.Adaptive_Drop = Adaptative_Dropout()

        # self.normal_clssifier = nn.Linear(in_features=vision_dim, out_features=len(seen_att), bias=False)
        # self.GRL_Layer = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)

        self.CE_loss = nn.CrossEntropyLoss()
        self.Reg_loss = nn.MSELoss()
        # self.Adv_loss = AdversarialLoss(cfg=self.args, classifier=self.V2S, seen_att=self.seen_att)
        # self.Relative_loss = relative_loss(cfg=self.args, classifier=self.V2S, seen_att=self.seen_att)

    def forward(self, x_img, x_attr, x_label, patch=False):

        x_fea_map  = self.backbone(x_img)
        x_fea = F.avg_pool2d(x_fea_map, kernel_size=(x_fea_map.size(-2), x_fea_map.size(-1))).squeeze()
        # x_fea = x_fea - x_fea.mean(dim=-1).unsqueeze(dim=-1) # 皮尔森系数

        if self.training:

            seen_att_norm = F.normalize(self.seen_att, p=2, dim=-1, eps=1e-12)
            pre_attr = self.V2S(x_fea)

            pre_attr_norm = F.normalize(pre_attr, p=2, dim=-1, eps=1e-12)
            pre_seen_scores = self.scale_cls * t.matmul(pre_attr_norm, seen_att_norm.t())
            V2S_loss = self.CE_loss(pre_seen_scores, x_label)

            # seen_att_norm = F.normalize(self.seen_att, p=2, dim=-1, eps=1e-12)
            # class_prototype = self.S2V(seen_att_norm)
            #
            # class_prototype_norm = F.normalize(class_prototype, p=2, dim=-1, eps=1e-12)
            # x_fea_norm = F.normalize(x_fea, p=2, dim=-1, eps=1e-12)
            #
            # pre_seen_scores = self.scale_cls * t.matmul(x_fea_norm, class_prototype_norm.t())
            # s2v_loss = self.CE_loss(pre_seen_scores, x_label)
            #
            # return s2v_loss


            # #########################normal classify loss  ################################################
            # x_fea = self.GRL_Layer(x_fea)
            # valina_classify_score = self.normal_clssifier(x_fea)
            # Valina_CLS_loss = self.CE_loss(valina_classify_score, x_label)

            # idx = t.tensor(list(range(0, self.args.ways * self.args.shots, self.args.shots))).to(self.args.device)
            # unique_attr = t.index_select(x_attr, 0, idx).float()
            #
            # s2v_label = t.Tensor(np.arange(self.args.ways).repeat(self.args.shots)).long().to(self.args.device)
            # print(s2v_label)

            # s2v_prototype = self.S2V(self.seen_att)
            # s2v_prototype_norm = F.normalize(s2v_prototype, dim=-1, p=2)
            #
            # x_fea = self.GRL_Layer(x_fea)
            #
            # x_fea_norm = F.normalize(x_fea, dim=-1, p=2)
            # s2v_pre_score = 20. * t.mm(x_fea_norm, s2v_prototype_norm.t())
            #
            # S2V_loss = self.CE_loss(s2v_pre_score, x_label)

            # Relative_loss = self.Relative_loss(x_f=x_fea, x_labels=x_label)
            # print(Relative_loss)
            # assert 1==2, print(Relative_loss)

            # if patch:
            #     # print("pre_atttr: ", pre_attr.size())
            #     # print(x_label)
            #     patch_kl_loss = Patch_KL_loss(patch_pre_attr=pre_attr, label=x_label, temp_factor=self.args.JS_Softmax_Temp)
            #     # print(patch_kl_loss)
            #     # print(V2S_loss)
            #     # assert 1==2
            #     return V2S_loss, patch_kl_loss
            # else:
            #     return V2S_loss

            return V2S_loss

        else:
            attr_norm = F.normalize(self.all_att, p=2, dim=-1, eps=1e-12)

            pre_attr = self.V2S(x_fea)
            pre_attr_norm = F.normalize(pre_attr, p=2, dim=-1, eps=1e-12)

            pre_scores = self.scale_cls * t.matmul(pre_attr_norm, attr_norm.t())

            # all_att_norm = F.normalize(self.all_att, p=2, dim=-1, eps=1e-12)
            # class_prototype = self.S2V(all_att_norm)
            #
            # class_prototype_norm = F.normalize(class_prototype, p=2, dim=-1, eps=1e-12)
            # x_fea_norm = F.normalize(x_fea, p=2, dim=-1, eps=1e-12)
            #
            # pre_scores = self.scale_cls * t.matmul(x_fea_norm, class_prototype_norm.t())

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
                 ):
        super(Fine_Align, self).__init__()
        self.args = args
        self.enc_shared_dim = attr_dim
        self.enc_exclusive_dim = attr_dim
        self.attr_dim = attr_dim
        self.h_dim = h_dim
        self.vision_dim = vision_dim
        self.seen_class = seen_class

        self.scale_cls = nn.Parameter(t.FloatTensor(1).fill_(args.scale_factor).to(args.device), requires_grad=True)
        self.bias = nn.Parameter(t.FloatTensor(1).fill_(0).to(args.device), requires_grad=True)
        self.seen_att = t.Tensor(seen_att).to(args.device)
        self.unseen_att = t.Tensor(unseen_att).to(args.device)
        self.all_att = t.cat([seen_att, unseen_att], dim=0).to(args.device)

        # S2V
        self.fc_s_1 = nn.Linear(in_features=attr_dim, out_features=1600, bias=True)
        self.fc_s_2 = nn.Linear(in_features=1600, out_features=vision_dim, bias=True)

        # GMIX
        self.GMIX = GMIX(args=args, ways=args.sub_ways, shots=args.sub_shots)

        # CL
        self.CL = Contrastive_Embedding_Loss(cfg=args, temperature=10.)
        
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
            self.fc_s_1.weight.data = truncated_normal_(self.fc_s_1.weight.data, mean=0, std=0.09)  # 0.09 不错
            self.fc_s_1.bias.data.fill_(0)
            self.fc_s_2.weight.data = truncated_normal_(self.fc_s_2.weight.data, mean=0, std=0.09)
            self.fc_s_2.bias.data.fill_(0)

        self.CE_loss = nn.CrossEntropyLoss()
        # self.MSE_loss = nn.MSELoss()


    def forward(self, x_fea, x_attr, x_meta_label, x_label=None, epoch=-1):

        x_fea = F.normalize(x_fea, p=2, dim=-1)
        ways, shots = self.args.sub_ways, self.args.sub_shots
        meta_task_x_label = x_meta_label


        if not self.training:
            x_attr = self.all_att
            # x_attr = F.normalize(x_attr, dim=-1, p=2)
            x_cls_prototype = F.relu(self.fc_s_2(F.relu(self.fc_s_1(x_attr))))
            # L2
            x_cls_prototype_norm = F.normalize(x_cls_prototype, p=2, eps=1e-12, dim=1)
            # L2
            x_fea_norm = F.normalize(x_fea, dim=-1, eps=1e-12)

            fine_pre_socres = self.scale_cls * t.mm(x_fea_norm, x_cls_prototype_norm.t())  # n x cls_num

            return fine_pre_socres

        else:
            if epoch < self.args.warm_epoch:
                # x_attr = nn.Dropout(0.2)(x_attr)
                gmix_feat, gmix_attr = self.GMIX(x_fea, x_attr, x_label, show=False, epoch=epoch)
                idx = t.tensor(list(range(0, ways * shots, shots))).to(self.args.device)

                gmix_attr = t.index_select(gmix_attr, 0, idx).float()
                gmix_attr_norm = F.normalize(gmix_attr, p=2, dim=-1)
                gmix_fea_norm = F.normalize(gmix_feat, dim=-1, eps=1e-12)

                gmix_class_prototype = F.relu(self.fc_s_2(F.relu(self.fc_s_1(gmix_attr_norm))))
                gmix_class_prototype_norm = F.normalize(gmix_class_prototype, p=2, eps=1e-12, dim=1)

                fine_pre_socres_s = self.scale_cls * t.mm(gmix_fea_norm, gmix_class_prototype_norm.t())
                fine_cls_ce_loss = self.CE_loss(fine_pre_socres_s, meta_task_x_label)

                mix_fine_cls_ce_loss = torch.tensor([0.0]).cuda()
                
            else:
                mix_feat, mix_attr = x_fea[ways*shots:], x_attr[ways*shots:]
                idx = t.tensor(list(range(0, int(ways * shots/2), shots))).to(self.args.device)
                mix_attr = t.index_select(mix_attr, 0, idx).float()
                mix_attr = F.normalize(mix_attr, dim=-1, p=2)
                
                x_cls_prototype = F.relu(self.fc_s_2(F.relu(self.fc_s_1(mix_attr))))
                x_cls_prototype_norm = F.normalize(x_cls_prototype, p=2, eps=1e-12, dim=1)  # N x D
                
                mix_feat_norm = F.normalize(mix_feat, dim=-1, eps=1e-12)                          # N x D
                mix_fine_pre_socres_s = self.scale_cls * t.mm(mix_feat_norm, x_cls_prototype_norm.t())
                mix_fine_cls_ce_loss = self.CE_loss(mix_fine_pre_socres_s, meta_task_x_label[:40])
                
                # CL_loss = self.CL([x_fea_norm, x_cls_prototype_norm], [x_meta_label, x_meta_label])
                
                
                # # x_attr = nn.Dropout(0.2)(x_attr)
                gmix_feat, gmix_attr = x_fea[:ways*shots], x_attr[:ways*shots]
                idx = t.tensor(list(range(0, ways * shots, shots))).to(self.args.device)

                gmix_attr = t.index_select(gmix_attr, 0, idx).float()
                gmix_attr_norm = F.normalize(gmix_attr, p=2, dim=-1)
                gmix_fea_norm = F.normalize(gmix_feat, dim=-1, eps=1e-12)

                gmix_class_prototype = F.relu(self.fc_s_2(F.relu(self.fc_s_1(gmix_attr_norm))))
                gmix_class_prototype_norm = F.normalize(gmix_class_prototype, p=2, eps=1e-12, dim=1)

                fine_pre_socres_s = self.scale_cls * t.mm(gmix_fea_norm, gmix_class_prototype_norm.t())
                fine_cls_ce_loss = self.CE_loss(fine_pre_socres_s, meta_task_x_label[:ways*shots])
                # # breakpoint()
                

                # gmix_feat, gmix_attr = x_fea, x_attr
                # idx = t.tensor(list(range(0, ways * shots, shots))).to(self.args.device)

                # gmix_attr = t.index_select(gmix_attr, 0, idx).float()
                # gmix_attr_norm = F.normalize(gmix_attr, p=2, dim=-1)
                # gmix_fea_norm = F.normalize(gmix_feat, dim=-1, eps=1e-12)

                # gmix_class_prototype = F.relu(self.fc_s_2(F.relu(self.fc_s_1(gmix_attr_norm))))
                # gmix_class_prototype_norm = F.normalize(gmix_class_prototype, p=2, eps=1e-12, dim=1)

                # fine_pre_socres_s = self.scale_cls * t.mm(gmix_fea_norm, gmix_class_prototype_norm.t())
                # fine_cls_ce_loss = self.CE_loss(fine_pre_socres_s, meta_task_x_label)

                # mix_fine_cls_ce_loss = torch.tensor([0.0]).cuda()

            return fine_cls_ce_loss, mix_fine_cls_ce_loss

      