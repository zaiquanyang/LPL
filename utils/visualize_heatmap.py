from sklearn.manifold import TSNE
import numpy as np
import torchvision.transforms as transforms
import torch as t
import torch.nn.functional as F

import os, time, random, tqdm, pickle, platform, sys
import seaborn as sns

if platform.system() == 'Linux':
    sys.path.append('/data/code/ZSL/C2F_ZSL_CVPR/')

import matplotlib.pyplot as plt

from tool import *

from model.c2f_awa2 import Fine_Align, Coarse_Align
from data.episode_data import ZSL_Episode_Img_Dataset

zsl_model_path_1 = '/data/code/ZSL/C2F_ZSL_CVPR/saved_model/CUB/v2s_loss/Fine/best_s2v_pkl_data_448_10_lr_0.0005_decay_5e-07_data.pkl_gama_1.0_temp_1.0_walk_num_6.pt'
zsl_model_path_2 = '/data/code/C2F_ZSL/UB/for_tsne_visualize/Fine/best_s2v_pkl_data_CUB_0_data_ck.pkl_gama_1.0_temp_1.0_walk_num_0.pt'


def visulize_sim_heat_map(pkl_data_path):


    data_loader = ZSL_Episode_Img_Dataset(cfg=cfg)
    train_loader = data_loader.train_dataloader
    test_s_loader = data_loader.test_seen_dataloader
    test_u_loader = data_loader.test_unseen_dataloader

    vision_dim = cfg.VISION_DIM  # 2048
    attr_dim = cfg.attr_num  # 312, 85, 102, 1024

    all_class_names = data_loader.mat_attr['allclasses_names']

    all_att = t.from_numpy(data_loader.raw_attr).float().to(cfg.device)
    seen_att = t.from_numpy(data_loader.seen_attr).float().to(cfg.device)
    unseen_att = t.from_numpy(data_loader.unseen_attr).float().to(cfg.device)

    seen_class = data_loader.seen_class_big_idx
    unseen_class = data_loader.unseen_class_big_idx

    saved_model_1 = t.load(zsl_model_path_2)['model']
    w1, w2, b1, b2 = saved_model_1['fc_s_1.weight'], saved_model_1['fc_s_2.weight'], saved_model_1['fc_s_1.bias'], \
                     saved_model_1['fc_s_2.bias']
    seen_att_norm = F.normalize(seen_att, dim=-1, p=2)
    unseen_att_norm = F.normalize(unseen_att, dim=-1, p=2)

    seen_prototype = t.mm((t.mm(seen_att_norm, w1.T) + b1), w2.T) + b2         # 150 x 2048
    seen_prototype_norm = F.normalize(seen_prototype, dim=-1, p=2)

    # # plot KDE
    # sim = t.mm(prototype_norm, prototype_norm.T)
    # for i in range(len(sim)):
    #     sim[i, i] = 0
    # sns.kdeplot(sim.view(-1).cpu().numpy(), shade=True, color="r")  # shade 设置曲线内部填充, color 设置颜色
    #
    # saved_model_2 = t.load(zsl_model_path_2)['model']
    # w1, w2, b1, b2 = saved_model_2['fc_s_1.weight'], saved_model_2['fc_s_2.weight'], saved_model_2['fc_s_1.bias'], \
    #                  saved_model_2['fc_s_2.bias']
    #
    unseen_prototype = t.mm((t.mm(unseen_att_norm, w1.T) + b1), w2.T) + b2  # 150 x 2048
    unseen_prototype_norm = F.normalize(unseen_prototype, dim=-1, p=2)

    # # plot KDE
    # sim = t.mm(prototype_norm, prototype_norm.T)
    # for i in range(len(sim)):
    #     sim[i, i] = 0
    # sns.kdeplot(sim.view(-1).cpu().numpy(), shade=True, color="g")  # shade 设置曲线内部填充, color 设置颜色
    plt.figure().set_size_inches(15, 6)
    sim = t.mm(unseen_prototype_norm, seen_prototype_norm.T)
    sns.heatmap(sim.cpu())
    # p1 = sns.heatmap(sim.cpu().numpy())
    # print('save img......')
    plt.savefig('/data/code/ZSL/C2F_ZSL_CVPR/visualize/CUB-Seen-Unseen-Similarity-Heat_LPL.png', dpi=800,  bbox_inches='tight',  pad_inches=0)
    plt.show()







if __name__ == "__main__":

    if platform.system() == 'Linux':
        cfg = parse_args(config_file='/data/code/ZSL/C2F_ZSL_CVPR/config/cub_hyp_para.yaml')
    else:
        print('config file error...')
        cfg = None


    # AWA2
    # /data/data/pkl_data/AWA2_448_4_data_lr_5e-4_decay_1e-4_iter_100_ck.pkl
    # zsl_model_path = '/data/code/ZSL/C2F_ZSL_CVPR/saved_model/CUB/v2s_loss/Fine/best_s2v_pkl_data_448_10_lr_0.0005_decay_5e-07_data.pkl_gama_1.0_temp_1.0_walk_num_6.pt'
    # zsl_model_path = '/data/code/ZSL/C2F_ZSL_CVPR/saved_model/CUB/V2S_loss+Attr_weight/Fine/best_s2v_pkl_data_448_10_lr_0.0005_decay_5e-07_data.pkl_gama_1.0_temp_1.0_walk_num_0.pt'
    visulize_sim_heat_map( pkl_data_path=None)


