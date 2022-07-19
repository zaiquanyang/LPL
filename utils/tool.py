from logging import handlers
import os, sys, time, logging, pickle, random, copy, tqdm, yaml, math
import argparse
from ast import literal_eval
import pickle as pkl
import scipy.io as sio

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data.mix_up_tool import Data_loader_Mixup_Feature
from torch.utils.data import DataLoader
from torch.backends import cudnn


np.set_printoptions(suppress=True)  # 取消科学计数法输出
np.set_printoptions(precision=4)    # 设置输出精度

t.set_printoptions(precision=1)



def time_():
    format_string = "%Y-%m-%d-%H-%M-%S"
    time_stamp = int(time.time())
    time_array = time.localtime(time_stamp)
    str_date = time.strftime(format_string, time_array)
    return str_date

def logger_(file_path, cfg):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)
    # log_root = os.path.join(cfg.log_dir, cfg.dataset_name)
    # if not os.path.exists(log_root):
    #     os.makedirs(log_root) 
    # log_root = os.path.join(log_root, cfg.train_stage)
    # if not os.path.exists(log_root):
    #     os.makedirs(log_root) 
    # log_root = os.path.join(log_root, cfg.key_word)
    # if not os.path.exists(log_root):
    #     os.makedirs(log_root)

    log_file_path = os.path.join(file_path, '{}_{}_lr_{}_decay_{}_temp_{}_pkl_data_{}_walk_num_{}.txt'.format(cfg.GMIX, \
        cfg.gama, cfg.coarse_lr, cfg.coarse_opt_decay, cfg.temp, cfg.pkl_data.split('/')[-1], cfg.walk_num))

    if os.path.exists(log_file_path):
        print("{} has existed.".format(log_file_path))
    
    fh = handlers.RotatingFileHandler(filename = log_file_path, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)                      # level
    formatter = logging.Formatter('%(asctime)s - %(message)s')   # format
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    print('Log file: {}'.format(log_file_path))
    return logger

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)

    # torch cuda
    t.cuda.empty_cache()
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    t.cuda.manual_seed_all(seed)

    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False



def record_hy_para(writer, opt):

    para_dic = dict()
    for key in opt.keys():
        if isinstance(opt[key], (int, float, bool)):
            para_dic[key] = opt[key]

    writer.add_hparams(para_dic, {'hy_para': True})


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(total_num,  trainable_num)


def Avg_pool2d(feat_map):
    w, h = feat_map.size(-2), feat_map.size(-1)

    return F.avg_pool2d(feat_map, kernel_size=(w, h)).squeeze()


def save_mat_data(cfg,
                  coarse_epoch,
                  model,
                  tr,
                  ts,
                  tu,
                  save_pkl_path='/data/data/pkl_data_c'):

    dir_data = {'train_seen': dict(), 'test_seen': dict(), 'test_unseen': dict(), 'patch_seen_data':dict()}
    tr_feat = []
    tr_b_label = []
    ts_feat = []
    ts_b_label = []
    tu_feat = []
    tu_b_label = []

    with t.no_grad():
        for x, s_label, b_label in tqdm.tqdm(tr):
            tmp_res_map = model.backbone(x.float().to(cfg.device))
            tmp_res_2048 = F.avg_pool2d(tmp_res_map, kernel_size=(tmp_res_map.size(-2), tmp_res_map.size(-1))).squeeze()
            tr_feat.append(tmp_res_2048)
            tr_b_label.append(b_label)

        for x, s_label, b_label in tqdm.tqdm(ts):
            tmp_res_map = model.backbone(x.float().to(cfg.device))
            tmp_res_2048 = F.avg_pool2d(tmp_res_map, kernel_size=(tmp_res_map.size(-2), tmp_res_map.size(-1))).squeeze()
            ts_feat.append(tmp_res_2048)
            ts_b_label.append(b_label)

        for x, s_label, b_label in tqdm.tqdm(tu):
            tmp_res_map = model.backbone(x.float().to(cfg.device))
            tmp_res_2048 = F.avg_pool2d(tmp_res_map, kernel_size=(tmp_res_map.size(-2), tmp_res_map.size(-1))).squeeze()
            tu_feat.append(tmp_res_2048)
            tu_b_label.append(b_label)


    # tr_feat = np.array(t.cat([t.cat(tr_feat, dim=0), tr_patch_feat], dim=1).cpu()).squeeze()
    tr_feat = np.array(t.cat(tr_feat, dim=0).cpu()).squeeze()
    tr_b_label = np.array(t.cat(tr_b_label, dim=0).cpu()).squeeze()
    ts_feat = np.array(t.cat(ts_feat, dim=0).cpu()).squeeze()
    ts_b_label = np.array(t.cat(ts_b_label, dim=0).cpu()).squeeze()
    tu_feat = np.array(t.cat(tu_feat, dim=0).cpu()).squeeze()
    tu_b_label = np.array(t.cat(tu_b_label, dim=0).cpu()).squeeze()

    dir_data['train_seen']['fea'] = tr_feat
    dir_data['train_seen']['labels'] = tr_b_label

    dir_data['test_seen']['fea'] = ts_feat
    dir_data['test_seen']['labels'] = ts_b_label

    dir_data['test_unseen']['fea'] = tu_feat
    dir_data['test_unseen']['labels'] = tu_b_label

    save_pkl_data_path = os.path.join(save_pkl_path, cfg.dataset_name)
    os.makedirs(save_pkl_data_path) if not os.path.exists(save_pkl_data_path) else print()
    save_pkl_data_path = os.path.join(save_pkl_data_path, cfg.key_word)
    os.makedirs(save_pkl_data_path) if not os.path.exists(save_pkl_data_path) else print()
    file_path = os.path.join(save_pkl_data_path, '{}_{}_lr_{}_decay_{}_data.pkl'.format(cfg.image_size,
                                                                                        coarse_epoch,
                                                                                        cfg.coarse_lr,
                                                                                        cfg.coarse_opt_decay))
    with open(file_path, 'wb') as fo:
        pickle.dump(dir_data, fo)

    print('save coarse pkl_data to file {}'.format(file_path))

    return file_path



def make_mat_dataloader(cfg,
                        attr,
                        save_pkl_path,
                        maml=False):

    pkl_dataset = pickle.load(open(save_pkl_path, 'rb'), encoding='utf-8')
    train_X, train_Y = pkl_dataset['train_seen']['fea'], pkl_dataset['train_seen']['labels']
    test_seen_X, test_seen_Y = pkl_dataset['test_seen']['fea'], pkl_dataset['test_seen']['labels']
    test_unseen_X, test_unseen_Y = pkl_dataset['test_unseen']['fea'], pkl_dataset['test_unseen']['labels']

    train_all_att = t.from_numpy(attr[train_Y]).float()  # n x attr_dim
    _, train_seen_small_label = np.unique(train_Y, return_inverse=True)
    seen_class_big_idx, test_seen_small_label = np.unique(test_seen_Y, return_inverse=True)
    unseen_class_big_idx, test_unseen_small_label = np.unique(test_unseen_Y, return_inverse=True)

    train_dataset = Data_loader_Mixup_Feature(args=cfg,
                                              feats=train_X,
                                              atts=train_all_att,
                                              b_labels=train_Y,
                                              s_labels=train_seen_small_label,
                                              ways=cfg.sub_ways,
                                              shots=cfg.sub_shots,
                                              maml=maml)

    test_seen_X = t.from_numpy(test_seen_X).float()
    test_unseen_X = t.from_numpy(test_unseen_X).float()

    test_seen_Y = t.Tensor(test_seen_Y.astype(np.int32)).long()
    test_unseen_Y = t.Tensor(test_unseen_Y.astype(np.int32)).long()

    test_seen_dataset_ = [(test_seen_X[i], test_seen_small_label[i], test_seen_Y[i])
                          for i in range(test_seen_Y.shape[0])]
    test_unseen_dataset_ = [(test_unseen_X[i], test_unseen_small_label[i], test_unseen_Y[i])
                            for i in range(test_unseen_Y.shape[0])]

    print('train size: {}, test_seen size: {}, test_unseen size: {}'.
          format(train_X.shape[0], len(test_seen_dataset_), len(test_unseen_dataset_)))

    test_seen_dataset = DataLoader(test_seen_dataset_, batch_size=80, shuffle=False)
    test_unseen_dataset = DataLoader(test_unseen_dataset_, batch_size=80, shuffle=False)

    seen_class_indices_inside_test = {c: [i for i in range(test_seen_Y.shape[0]) if test_seen_Y[i] == c]
                                      for c in seen_class_big_idx}
    unseen_class_indices_inside_test = {c: [i for i in range(test_unseen_Y.shape[0]) if test_unseen_Y[i] == c]
                                        for c in unseen_class_big_idx}

    Class_Center = []
    for lb in range(len(seen_class_big_idx)+len(unseen_class_big_idx)):
        idx = (train_Y == lb).nonzero()[0]
        if len(idx) > 0:
            Class_Center.append(t.Tensor(train_X[idx].mean(axis=0)).unsqueeze(dim=0))
        else:
            Class_Center.append(t.ones(1, 2048))

    Class_Center = t.cat(Class_Center, dim=0)

    return train_dataset, test_seen_dataset, test_unseen_dataset, seen_class_indices_inside_test, \
           unseen_class_indices_inside_test, train_dataset.classes



# def make_mat_dataloader(cfg,
#                         attr,
#                         save_pkl_path,
#                         maml=False):
#     pkl_dataset = pickle.load(open(save_pkl_path, 'rb'), encoding='utf-8')
#     train_X, train_Y = pkl_dataset['train_seen']['fea'], pkl_dataset['train_seen']['labels']
#     test_seen_X, test_seen_Y = pkl_dataset['test_seen']['fea'], pkl_dataset['test_seen']['labels']
#     test_unseen_X, test_unseen_Y = pkl_dataset['test_unseen']['fea'], pkl_dataset['test_unseen']['labels']
#
#
#     train_all_att = t.from_numpy(attr[train_Y]).float()        # n x attr_dim
#     _, train_seen_small_label = np.unique(train_Y, return_inverse=True)
#     seen_class_big_idx, test_seen_small_label = np.unique(test_seen_Y, return_inverse=True)
#     unseen_class_big_idx, test_unseen_small_label = np.unique(test_unseen_Y, return_inverse=True)
#
#     train_dataset = Data_loader_Mixup_Feature(args=cfg,
#                                               feats=train_X,
#                                               atts=train_all_att,
#                                               b_labels=train_Y,
#                                               s_labels=train_seen_small_label,
#                                               ways=cfg.sub_ways,
#                                               shots=cfg.sub_shots,
#                                               maml=maml)
#
#
#     test_seen_X = t.from_numpy(test_seen_X).float()
#     test_unseen_X = t.from_numpy(test_unseen_X).float()
#
#     test_seen_Y = t.Tensor(test_seen_Y.astype(np.int32)).long()
#     test_unseen_Y = t.Tensor(test_unseen_Y.astype(np.int32)).long()
#
#     test_seen_dataset = [(test_seen_X[i], test_seen_small_label[i], test_seen_Y[i]) for i in range(test_seen_Y.shape[0])]
#     test_unseen_dataset = [(test_unseen_X[i], test_unseen_small_label[i], test_unseen_Y[i]) for i in range(test_unseen_Y.shape[0])]
#     print('train size: {}, test_seen size: {}, '
#           'test_unseen size: {}'.format(train_X.shape[0], len(test_seen_dataset), len(test_unseen_dataset)))
#     test_seen_dataset = DataLoader(test_seen_dataset, batch_size=80, shuffle=False)
#     test_unseen_dataset = DataLoader(test_unseen_dataset, batch_size=80, shuffle=False)
#
#     seen_class_indices_inside_test = {c: [i for i in range(test_seen_Y.shape[0]) if test_seen_Y[i] == c] for c in seen_class_big_idx}
#     unseen_class_indices_inside_test = {c: [i for i in range(test_unseen_Y.shape[0]) if test_unseen_Y[i] == c] for c in unseen_class_big_idx}
#
#     return train_dataset, test_seen_dataset, test_unseen_dataset, seen_class_indices_inside_test, \
#            unseen_class_indices_inside_test, train_dataset.classes




def evaluate(log,
             cfg,
             tr_time,
             epoch,
             model,
             seen_cls_idx,
             unseen_cls_idx,
             ts_dataloader,
             tu_dataloader,
             seen_class_indices_inside_test,
             unseen_class_indices_inside_test):

    # eval mode !!!!
    model.eval()
    start = time.time()
    pre_ts_score = []
    pre_tu_score = []
    ss_label_gt = []
    sb_label_gt = []
    us_label_gt = []
    ub_label_gt = []

    with t.no_grad():
        for x, s_label, b_label in ts_dataloader:
            pre_tmp = model(x.float().to(cfg.device), s_label.to(cfg.device), b_label.to(cfg.device))
            pre_ts_score.append(pre_tmp)

            ss_label_gt.append(s_label.to(cfg.device).reshape(-1))
            sb_label_gt.append(b_label.to(cfg.device).reshape(-1))

        for x, s_label, b_label in tu_dataloader:
            pre_tmp = model(x.float().to(cfg.device), s_label.to(cfg.device), b_label.to(cfg.device))
            pre_tu_score.append(pre_tmp)
            us_label_gt.append(s_label.to(cfg.device).reshape(-1))
            ub_label_gt.append(b_label.to(cfg.device).reshape(-1))

    pre_ts_score = t.cat(pre_ts_score, dim=0)
    pre_tu_score = t.cat(pre_tu_score, dim=0)
    ss_label_gt = t.cat(ss_label_gt, dim=0)
    sb_label_gt = t.cat(sb_label_gt, dim=0)
    us_label_gt = t.cat(us_label_gt, dim=0)
    ub_label_gt = t.cat(ub_label_gt, dim=0)

    # print(pre_ts_score[:1000:10])
    pre_zsl_s = pre_ts_score[:, :seen_cls_idx.shape[0]].argmax(dim=1)
    pre_zsl_u = pre_tu_score[:, seen_cls_idx.shape[0]:].argmax(dim=1)

    gzsl_harmonic = -1.0
    gzsl_seen_acc = 0.0
    gzsl_unseen_acc = 0.0
    zsl_seen_acc = 0.0
    zsl_unseen_acc = 0.0
    calibration_factor = 0.0

    for i in range(0, 31, 1):

        tmp_calibration_factor = 1.0  # 0.7 + 0.01 * i

        pre_ts_score_copy, pre_tu_score_copy = copy.deepcopy(pre_ts_score), copy.deepcopy(pre_tu_score)
        pre_ts_score_copy[:, :seen_cls_idx.shape[0]] = pre_ts_score_copy[:, :seen_cls_idx.shape[0]] * tmp_calibration_factor
        pre_tu_score_copy[:, :seen_cls_idx.shape[0]] = pre_tu_score_copy[:, :seen_cls_idx.shape[0]] * tmp_calibration_factor

        seen_cls_idx_tensor = t.from_numpy(seen_cls_idx.astype(float)).to(cfg.device)
        unseen_cls_idx_tensor = t.from_numpy(unseen_cls_idx.astype(float)).to(cfg.device)
        all_cls_idx = t.cat([seen_cls_idx_tensor, unseen_cls_idx_tensor], dim=0)

        pre_gzsl_s = all_cls_idx[pre_ts_score_copy[:, :].argmax(dim=1)].long()
        pre_gzsl_u = all_cls_idx[pre_tu_score_copy[:, :].argmax(dim=1)].long()

        guessed_zsl_s = (pre_zsl_s == ss_label_gt)
        guessed_zsl_u = (pre_zsl_u == us_label_gt)
        guessed_gzsl_s = (pre_gzsl_s == sb_label_gt)
        guessed_gzsl_u = (pre_gzsl_u == ub_label_gt)


        # print(seen_class_indices_inside_test.keys())
        zsl_seen_acc = np.mean([t.Tensor.float(guessed_zsl_s[cls_idx]).mean().item() for cls_idx in [seen_class_indices_inside_test[c] for c in seen_cls_idx]])
        zsl_unseen_acc = np.mean([t.Tensor.float(guessed_zsl_u[cls_idx]).mean().item() for cls_idx in [unseen_class_indices_inside_test[c] for c in unseen_cls_idx]])
        tmp_gzsl_seen_acc = np.mean([t.Tensor.float(guessed_gzsl_s[cls_idx]).mean().item() for cls_idx in [seen_class_indices_inside_test[c] for c in seen_cls_idx]])
        tmp_gzsl_unseen_acc = np.mean([t.Tensor.float(guessed_gzsl_u[cls_idx]).mean().item() for cls_idx in [unseen_class_indices_inside_test[c] for c in unseen_cls_idx]])

        tmp_gzsl_harmonic = 2 * (tmp_gzsl_seen_acc * tmp_gzsl_unseen_acc) / (tmp_gzsl_seen_acc + tmp_gzsl_unseen_acc)

        if tmp_gzsl_harmonic > gzsl_harmonic:
            gzsl_harmonic = tmp_gzsl_harmonic
            gzsl_seen_acc = tmp_gzsl_seen_acc
            gzsl_unseen_acc = tmp_gzsl_unseen_acc
            calibration_factor = tmp_calibration_factor

    end = time.time()

    res_info = f'[{cfg.train_stage}|{epoch}]:::ZSL-S: {zsl_seen_acc * 100:.02f}:::ZSL-U: {zsl_unseen_acc * 100:.02f}:::' \
               f'GZSL-S: {gzsl_seen_acc * 100:.02f}:::GZSL-U: {gzsl_unseen_acc * 100:.02f}:::GZSL-H: {gzsl_harmonic * 100:.02f}:::' \
               f'Calibration_Factor:{calibration_factor:.02f}:::' \
               f'Train Time:{tr_time:.02f}:::Test Time: {end-start:.02f}'
    # if epoch % 10 == 0:
    print(res_info)
    log.info(res_info)

    # if epoch % 10 == 0:
    #     print(res_info)
    #     log.info(res_info)

    model.train()

    return gzsl_harmonic, zsl_unseen_acc, gzsl_seen_acc, gzsl_unseen_acc, zsl_seen_acc



class update_reliable_unseen_data(nn.Module):

    def __init__(self, cfg, saved_pkl_path, s2v_model, all_attr, seen_class_id, unseen_class_id, reliable_filter, unseen_class_indices_inside_test):
        """

        :param saved_pkl_path: the pkl_data after the coarse training. default use the inductive pkl_dataset
        :param s2v_model: fine_align model
        :param all_attr: novel unseen class semantic embeddings
        :param unseen_class_id: unseen class index in all class
        """
        super(update_reliable_unseen_data, self).__init__()

        self.cfg = cfg

        self.saved_pkl_path = saved_pkl_path      # possible to update when fine-tune the viusal backbone during the transductive training
        self.s2v_model = s2v_model                # s2v model architecture

        pkl_dataset = pickle.load(open(self.saved_pkl_path, 'rb'), encoding='utf-8')
        train_X, train_Y = pkl_dataset['train_seen']['fea'], pkl_dataset['train_seen']['labels']                # array
        test_unseen_X, test_unseen_Y = pkl_dataset['test_unseen']['fea'], pkl_dataset['test_unseen']['labels']  # array

        self.train_seen_data = train_X
        self.train_seen_label = train_Y

        self.test_unseen_data = test_unseen_X
        self.test_unseen_label = test_unseen_Y

        self.all_attr = all_attr
        self.unseen_attr = all_attr[unseen_class_id]
        self.unseen_class_id = unseen_class_id

        self.reliable_filter = reliable_filter
        self.unseen_class_indices_inside_test = unseen_class_indices_inside_test
        # print(self.unseen_class_id)

    def forward(self, model_dict):
        # s2v_model = model_dict
        s2v_model = self.s2v_model
        s2v_model.load_state_dict(model_dict)
        s2v_model.cuda().eval()

        test_unseen_feat = F.normalize(t.from_numpy(self.test_unseen_data).float().cuda(), dim=-1, p=2)

        # unseen_attr = F.normalize(t.from_numpy(self.unseen_attr).float().cuda(), dim=-1, p=2)
        unseen_attr = t.from_numpy(self.unseen_attr).float().cuda()        # 推断时是否对属性进行L2正则化对AWA2很关键
        unseen_prototype = F.relu(s2v_model.fc_s_2(F.relu(s2v_model.fc_s_1(unseen_attr))))
        unseen_prototype = F.normalize(unseen_prototype, dim=-1, p=2)

        pre_unseen_scores = t.mm(test_unseen_feat, unseen_prototype.t())    # N x 50

        max_val, pred_idx = pre_unseen_scores.max(dim=1)
        pred_idx = pred_idx.view(-1)

        pred_cls_id = self.unseen_class_id[pred_idx.cpu()]
        guessed_zsl_u = (pred_cls_id == self.test_unseen_label)
        zsl_unseen_acc_list = [round(t.Tensor(guessed_zsl_u[cls_idx]).float().mean().item(), 3) for cls_idx \
            in [self.unseen_class_indices_inside_test[c] for c in self.unseen_class_id]]
        zsl_unseen_acc = np.mean(zsl_unseen_acc_list)

        if self.reliable_filter == 'constant':
            selected_num = 100
            sorted_socre, sorted_idx = pre_unseen_scores.sort(dim=0, descending=True)    # column sorting
            selected_idx = sorted_idx[:selected_num].reshape(-1)                         # (selected_num x 50, )

            outpred = t.arange(len(self.unseen_class_id)).repeat(selected_num).view(-1).cpu()  # [0, 1, 2, 3, ...., 50, ... 0, 1, 2, 3, ...., 50]
            train_unseen_data_temp = self.test_unseen_data[selected_idx.cpu(), :]              # (selected_num x 50, 2048)


            train_unseen_label_temp = self.unseen_class_id[outpred]

            # 计算选中的样本预测的准确率
            # pre_label = pred_idx[selected_idx]
            # # gt_label = self.test_unseen_label[selected_idx.cpu()]
            # # guessed_zsl_u = (pre_label == gt_label)
            # # select_guessed_zsl_u = guessed_zsl_u
            # # select_guessed_zsl_u[i] = False if
            zsl_unseen_acc_list = []
            for k, c in enumerate(self.unseen_class_id):
                idx = selected_idx.reshape(selected_num, -1)[:, k].view(-1).cpu().numpy()
                zsl_unseen_acc_list.append((self.test_unseen_label[idx] == c).mean())
                # cls_idx = self.unseen_class_indices_inside_test[c]
                # zsl_unseen_acc = round(t.Tensor(guessed_zsl_u[np.intersect1d(cls_idx, selected_idx.reshape(selected_num, -1)[:, k].view(-1).cpu().numpy())]).float().mean().item(), 3)
                # zsl_unseen_acc_  = 0.0  if math.isnan(zsl_unseen_acc) else zsl_unseen_acc
                # zsl_unseen_acc_list.append(zsl_unseen_acc_)

            # zsl_unseen_acc_list = [round(t.Tensor(guessed_zsl_u[np.intersect1d(cls_idx, selected_idx.reshape(selected_num, -1)[self.unseen_class_id.index(c)])]).float().mean().item(), 3) for cls_idx in [self.unseen_class_indices_inside_test[c] for c in self.unseen_class_id]]
            # zsl_unseen_acc_list = [x for x in zsl_unseen_acc_list if math.isnan(x) == False]

            zsl_unseen_acc = np.mean(zsl_unseen_acc_list)
            print('{}/{}'.format(zsl_unseen_acc, zsl_unseen_acc_list))
            # breakpoint()

        elif self.reliable_filter == 'ratio':     # self.reliable_filter == 'ratio':
            ratio_reliable = 1.05
            selected_num = 10
            score_sorted, _ = pre_unseen_scores.sort(dim=1, descending=True)
            ratio_select_reliable_idx_list = [i  for i in range(score_sorted.size(0)) if ((score_sorted[i, 0] / (score_sorted[i, 1] + 1e-12)) > 1.1)]
            ratio_select_unreliable_idx_list = [i for i in range(score_sorted.size(0)) if (score_sorted[i, 0] / (score_sorted[i, 1] + 1e-12)) < 1.1 and ((score_sorted[i, 0] / (score_sorted[i, 1] + 1e-12)) > 1.05)]

            ratio_select_idx = t.Tensor(ratio_select_reliable_idx_list).long()
            ratio_select_pre = pred_idx[ratio_select_idx].cpu()
            ratio_select_pre_idx = self.unseen_class_id[ratio_select_pre]

            ratio_select_unreliable_idx = t.Tensor(ratio_select_unreliable_idx_list).long()
            ratio_select_unreliable_pre = pred_idx[ratio_select_unreliable_idx].cpu()
            ratio_select_unreliable_pre_idx = self.unseen_class_id[ratio_select_unreliable_pre]

            zsl_unseen_acc_list = []
            unseen_sample_num_list = []

            # breakpoint()
            for cls in range(len(self.unseen_class_id)):
                select_sample_idx = np.array(ratio_select_reliable_idx_list)[(ratio_select_pre == cls).numpy()]
                # cls_pre_idx = pred_idx.cpu().numpy()[select_sample_idx]
                if len(select_sample_idx) == 0:
                    zsl_unseen_acc_list.append(0.0)
                    unseen_sample_num_list.append(0)
                else:
                    unseen_sample_num_list.append(len(select_sample_idx))
                    gt_label = self.test_unseen_label[select_sample_idx]
                    zsl_unseen_acc_list.append(round((self.unseen_class_id[cls] == gt_label).mean().item(), 2))

            zsl_unseen_acc = np.mean(zsl_unseen_acc_list)
            print('{}/{}/{}'.format(zsl_unseen_acc, zsl_unseen_acc_list, unseen_sample_num_list))
            print('reliable/unreliable:{}/{}'.format(len(ratio_select_reliable_idx_list), len(ratio_select_unreliable_idx_list)))
            unseen_sample_num_list = np.array([i+10 for i in unseen_sample_num_list])
            pseudo_class_percent = 1. / (unseen_sample_num_list / (unseen_sample_num_list).sum())
            pseudo_class_percent = pseudo_class_percent / pseudo_class_percent.sum()
            # pseudo_class_percent = np.array(unseen_sample_num_list / sum(unseen_sample_num_list))
            pseudo_class_percent_dict = {self.unseen_class_id[i]: pseudo_class_percent[i] for i in range(len(self.unseen_class_id))}
            
            print(pseudo_class_percent_dict)
            
            
            sorted_score, sorted_idx = pre_unseen_scores.sort(dim=0, descending=True)
            stable_selected_idx = sorted_idx[:selected_num].view(-1)                                   # selected_num x 50
            outpred = t.arange(len(self.unseen_class_id)).repeat(selected_num).view(-1)
            stable_select_pre_idx = self.unseen_class_id[outpred.cpu()]
            # print(type(ratio_select_pre_idx))
            # print(type(stable_select_pre_idx))
            train_unseen_label_temp =  np.concatenate([ratio_select_pre_idx, stable_select_pre_idx], axis=0)
            train_unseen_list = ratio_select_reliable_idx_list + stable_selected_idx.tolist()
            train_unseen_data_temp = self.test_unseen_data[train_unseen_list, :]

            # train_unseen_label_temp = ratio_select_pre_idx
            # train_unseen_data_temp = self.test_unseen_data[ratio_select_reliable_idx_list, :]

            train_unseen_unreliable_label_temp = ratio_select_unreliable_pre_idx
            train_unseen_unreliable_data_temp = self.test_unseen_data[ratio_select_unreliable_idx_list, :]

            # breakpoint()

        else:
            pass

        # print('train unseen data size is {}/{}'.format(len(train_unseen_data_temp), len(self.test_unseen_data)))

        train_seen_unseen_data = np.concatenate([self.train_seen_data, train_unseen_data_temp], axis=0)
        train_seen_unseen_label = np.concatenate([self.train_seen_label, train_unseen_label_temp], axis=0)
        train_all_attr = t.from_numpy(self.all_attr[train_seen_unseen_label]).float()
        
        # train_seen_unseen_data_ = np.concatenate([self.train_seen_data, train_unseen_unreliable_data_temp], axis=0)
        # train_seen_unseen_label_ = np.concatenate([self.train_seen_label, train_unseen_unreliable_label_temp], axis=0)
        # train_all_attr_ = t.from_numpy(self.all_attr[train_seen_unseen_label_]).float()

        train_seen_data = self.train_seen_data
        train_seen_label = self.train_seen_label
        train_seen_attr = t.from_numpy(self.all_attr[train_seen_label]).float()
        
        train_unseen_data = train_unseen_data_temp
        train_unseen_label = train_unseen_label_temp
        train_unseen_attr = t.from_numpy(self.all_attr[train_unseen_label]).float()

        # generate dataloader
        train_dataset = Data_loader_Mixup_Feature(args=self.cfg,
                                                  feats=train_seen_unseen_data,
                                                  atts=train_all_attr,
                                                  b_labels=train_seen_unseen_label,
                                                  ways=self.cfg.sub_ways,
                                                  shots=self.cfg.sub_shots,
                                                  maml=False)

        # train_dataset_ = Data_loader_Mixup_Feature(args=self.cfg,
        #                                           feats=train_seen_unseen_data_,
        #                                           atts=train_all_attr_,
        #                                           b_labels=train_seen_unseen_label_,
        #                                           ways=self.cfg.sub_ways,
        #                                           shots=self.cfg.sub_shots,
        #                                           maml=False)

        train_dataset_s = Data_loader_Mixup_Feature(args=self.cfg,
                                                  feats=train_seen_data,
                                                  atts=train_seen_attr,
                                                  b_labels=train_seen_label,
                                                  ways=self.cfg.sub_ways,
                                                  shots=self.cfg.sub_shots,
                                                  maml=False)
        
        train_dataset_u = Data_loader_Mixup_Feature(args=self.cfg,
                                                  feats=train_unseen_data,
                                                  atts=train_unseen_attr,
                                                  b_labels=train_unseen_label,
                                                  ways=self.cfg.sub_ways,
                                                  shots=self.cfg.sub_shots,
                                                  maml=False,
                                                  prob=pseudo_class_percent)

        return train_dataset, train_dataset_s, train_dataset_u, pseudo_class_percent_dict

        # return [train_dataset]






def save_model(main_epoch, sub_epoch, cfg, model, U, f_name):

    saved_dir = {
        'main_epoch': main_epoch,
        'sub_epoch': sub_epoch,
        'cfg': cfg,
        'model': model.state_dict(),
        'cls_acc': U,
    }

    t.save(saved_dir, f_name)



class H_U_Update(object):
    def __init__(self):
        self.H_MAX = 0.0
        self.Epoch = 0
        self.U_MAX = 0.0


    def Update(self, H, U, E):

        # self.H_MAX = H if H > self.H_MAX else self.H_MAX
        self.U_MAX = U if U > self.U_MAX else self.U_MAX
        [self.H_MAX, self.Epoch] = [H, E] if H > self.H_MAX else [self.H_MAX, self.Epoch]
        # self.Epoch = E if H > self.H_MAX else self.Epoch


def make_optimizer(model, tp, opt):

    # model_para = filter(lambda p: id(p), model.parameters())
    model_para = filter(lambda p: id(p), model.parameters())
    optimizer = t.optim.Adam(model_para, lr=opt.fine_lr, weight_decay=opt.fine_opt_decay)

    return optimizer


def biased_random_walk(graph_edge_bool, walk_num):
    node_N = graph_edge_bool.float().size(0)  # ways

    diag_one_mask = t.eye(node_N).cuda()  # node_N x node_N
    diag_zero_mask = 1.0 - diag_one_mask  # node_N x node_N

    p = t.ones(node_N) * 10

    alpha = 1.0

    walk_mask = t.zeros(node_N, node_N).cuda()

    for i in range(node_N):
        p_tmp = copy.deepcopy(p)
        p_tmp[i] = 1e-20

        p_tmp_norm = np.float64(np.array(F.softmax(p_tmp, dim=0).cpu()))
        p_tmp_norm = p_tmp_norm / (p_tmp_norm.sum())
        idx = np.random.choice(np.arange(node_N), size=walk_num, replace=True, p=p_tmp_norm).astype(np.int32)
        walk_mask[i][idx] = 1.0

    return walk_mask*diag_zero_mask


def random_sample(graph_edge_bool, walk_num):
    node_N = graph_edge_bool.float().size(0)  # ways

    walk_mask = t.zeros(node_N, node_N).cuda()

    for i in range(node_N):
        idx_list = []
        for j in range(walk_num):
            idx = int(np.random.choice(np.arange(node_N), size=1, replace=False)[0])
            if idx == i:
                idx = idx+1 if (idx+1) < node_N else idx-1
            idx_list.append(idx)
            walk_mask[i][idx] = 1.0

    return walk_mask





def generate_class_mask(cls_num, instance_num, interact_class_num):
    """
    :param cls_num:
    :param instance_num:
    :param interact_class_num:
    :return:
    """

    # tmp_list = [0] * (cls_num-2)
    # tmp_list.append(1)   # [0, 0, ..., 1]  cls_num-1

    mask_list = []

    for i in range(cls_num):
        tmp_list = [0] * (cls_num - interact_class_num)
        for j in range(1, interact_class_num):
            idx = np.random.randint(0, cls_num-(interact_class_num-j))
            tmp_list.insert(idx, 1)
        # idx = np.random.randint(0, cls_num - 1)
        # tmp_list.insert(idx, 1)
        tmp_list.insert(i, 1)
        tmp_tensor = t.Tensor(tmp_list).reshape(1, -1)
        mask_list.append(tmp_tensor)

    mask_tensor = t.cat(mask_list, dim=0)

    # element_size = mask_tensor.size(0) * mask_tensor.size(1)
    # mask_tensor = mask_tensor.repeat_interleave(repeats=instance_num, dim=1).repeat_interleave(repeats=instance_num,dim=0)

    return mask_tensor


def generate_class_intra_mask(mask, instance_num):
    """

    :param mask:
    :param instance_num:
    :return:
    """
    cls_num = int(mask.size(0)/instance_num)
    class_intra_mask = t.zeros_like(mask)
    for i in range(1, cls_num+1):
        idx = (i-1) * instance_num
        class_intra_mask[idx:(idx+instance_num), idx:(idx+instance_num)] = 1

    mask_diag = t.eye(class_intra_mask.size(0))

    # print(class_intra_mask)
    # print(mask_diag)
    class_intra_mask = class_intra_mask - mask_diag

    return class_intra_mask


def generate_class_inter_mask(mask, class_intra_mask):
    """

    :param mask:
    :param instance_num:
    :return:
    """
    diag_mask = t.eye(class_intra_mask.size(0))

    class_inter_mask = mask - class_intra_mask - diag_mask

    return class_inter_mask


#=====================preprocess the node_sim===========================================================================
def preprocess_node(node_sim, show):

    shots, ways, ways = node_sim.shape

    diag_one_mask = t.eye(ways).cuda()  # way x ways
    diag_zero_mask = 1.0 - diag_one_mask
    # node_sim = node_sim * diag_zero_mask          #
    # ===================normalization [0, 1]===============================
    e_max = node_sim.max(dim=-1)[0].reshape(shots, ways, 1)  # shots x ways
    e_min = node_sim.min(dim=-1)[0].reshape(shots, ways, 1)  # shots x ways
    node_sim_norm = (node_sim - e_min) / (e_max - e_min)     # * self.temp 0.1

    tensor_eye = t.eye(ways).cuda().float()                  # ways x ways
    # mask = 1.0 - tensor_eye
    # print(node_sim_norm.is_cuda, tensor_eye.is_cuda)
    node_sim_norm = node_sim_norm + tensor_eye                   # shots x ways x ways

    return node_sim_norm



# ======================================================================================================================
# ======== All following helper functions have been borrowed from from https://github.com/Jia-Research-Lab/PFENet ======
# ======================================================================================================================

class CfgNode(dict):
    """
    CfgNode represents an internal node in the configuration tree. It's a simple
    dict-like container that allows for attribute-based access to keys.
    """

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        # Recursively convert nested dictionaries in init_dict into CfgNodes
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        for k, v in init_dict.items():
            if type(v) is dict:
                # Convert dict to CfgNode
                init_dict[k] = CfgNode(v, key_list=key_list + [k])
        super(CfgNode, self).__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super(CfgNode, self).__repr__())


def _decode_cfg_value(v):
    if not isinstance(v, str):
        return v
    try:
        v = literal_eval(v)
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
    original_type = type(original)
    replacement_type = type(replacement)

    # The types must match (with some exceptions)
    if replacement_type == original_type:
        return replacement

    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    casts = [(tuple, list), (list, tuple)]
    try:
        casts.append((str, unicode))  # noqa: F821
    except Exception:
        pass

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(
            original_type, replacement_type, original, replacement, full_key
        )
    )


def load_cfg_from_cfg_file(file: str):
    cfg = {}
    assert os.path.isfile(file) and file.endswith('.yaml'), \
        '{} is not a yaml file'.format(file)

    with open(file, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    for k, v in cfg_from_file.items():
        # for k, v in cfg_from_file[key].items():
        # print(k, type(v))
        cfg[k] = v

    cfg = CfgNode(cfg)
    return cfg


def merge_cfg_from_list(cfg,
                        cfg_list):
    new_cfg = copy.deepcopy(cfg)
    # assert len(cfg_list) % 2 == 0, cfg_list
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        subkey = full_key.split('.')[-1]
        assert subkey in cfg, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, cfg[subkey], subkey, full_key
        )
        # print(subkey, value)
        setattr(new_cfg, subkey, value)
        # print(new_cfg)
    return new_cfg


def parse_args(config_file):
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', type=str, default=config_file, help='config file')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    # print(cfg.coarse_lr)
    # breakpoint()
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg



def truncated_normal_(tensor, mean=0, std=0.09):
    with t.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size+(4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor


def entropy_calibration(dataset, saved_model):
    att_mat_path = r'/data/data/xlsa17/data/CUB/att_splits.mat'
    res_mat_path = r'/data/data/xlsa17/data/CUB/res101.mat'
    att_mat = sio.loadmat(att_mat_path)
    res_mat = sio.loadmat(res_mat_path)
    all_attr = att_mat['att'].T

    seen_loc = att_mat['trainval_loc'] - 1
    unseen_loc = att_mat['test_unseen_loc'] - 1
    seen_labels = np.unique(res_mat['labels'][seen_loc], return_inverse=True)[0] - 1
    unseen_labels = np.unique(res_mat['labels'][unseen_loc], return_inverse=True)[0] - 1

    seen_att = all_attr[seen_labels]
    unseen_att = all_attr[unseen_labels]

    # print('seen attributes: {}'.format(seen_att.shape))
    seen_attr = F.normalize(t.Tensor(seen_att), dim=-1)
    unseen_attr = F.normalize(t.Tensor(unseen_att), dim=-1)

    saved_v2s_model = '/data/code/C2F_ZSL/saved_model_cvpr/CUB/gmix_meta_entropy/coarse/9_lr_0.0005_decay_5e-07_way_8_shot_2_iter_300.pt'
    saved_pkl_data = '/data/data/coarse_pkl_data_cvpr/CUB/gmix_meta_entropy/448_9_lr_0.0005_decay_5e-07_data.pkl'
    model_dict = t.load(saved_v2s_model)
    v2s = model_dict['model']['V2S.weight']  # (312, 2048)

    pkl_dataset = pkl.load(open(saved_pkl_data, 'rb'), encoding='utf-8')
    train_X, train_Y = pkl_dataset['train_seen']['fea'], pkl_dataset['train_seen']['labels']
    test_seen_X, test_seen_Y = pkl_dataset['test_seen']['fea'], pkl_dataset['test_seen']['labels']
    test_unseen_X, test_unseen_Y = pkl_dataset['test_unseen']['fea'], pkl_dataset['test_unseen']['labels']

    # print(test_unseen_X.shape)               # (2967, 2048)
    # print(test_seen_X.shape)                 # (1764, 2048)

    test_seen_X, test_unseen_X = t.Tensor(test_seen_X).float(), t.Tensor(t.Tensor(test_unseen_X)).float()

    pre_seen_attr = t.mm(test_seen_X, v2s.t().cpu())  # [1764, 312]
    pre_unseen_attr = t.mm(test_unseen_X, v2s.t().cpu())  # [2967, 312]
    pre_seen_attr = F.normalize(pre_seen_attr, dim=-1)
    pre_unseen_attr = F.normalize(pre_unseen_attr, dim=-1)

    pre_seen_score = t.mm(pre_seen_attr, seen_attr.t()) * 20
    pre_unseen_score = t.mm(pre_unseen_attr, seen_attr.t()) * 20
    pre_seen_score = F.softmax(pre_seen_score, dim=-1)
    pre_unseen_score = F.softmax(pre_unseen_score, dim=-1)

    pre_seen_score_ = t.mm(pre_seen_attr, unseen_attr.t()) * 20
    pre_unseen_score_ = t.mm(pre_unseen_attr, unseen_attr.t()) * 20
    pre_seen_score_ = F.softmax(pre_seen_score_, dim=-1)
    pre_unseen_score_ = F.softmax(pre_unseen_score_, dim=-1)

    seen_entropy = ((-1) * t.log(pre_seen_score) * pre_seen_score).sum(dim=1)
    seen_entropy_ = ((-1) * t.log(pre_seen_score_) * pre_seen_score_).sum(dim=1)
    unseen_entropy = ((-1) * t.log(pre_unseen_score) * pre_unseen_score).sum(dim=1)
    unseen_entropy_ = ((-1) * t.log(pre_unseen_score_) * pre_unseen_score_).sum(dim=1)

    seen_entropy = seen_entropy - seen_entropy_
    unseen_entropy = unseen_entropy - unseen_entropy_


    # print(seen_entropy.shape)

    print('seen max entropy:{}, min entropy:{}, mean entropy:{}'.format(seen_entropy.max(), seen_entropy.min(),
                                                                        seen_entropy.mean()))
    # 4.17, 0.087, 1.51
    print('unseen max entropy:{}, min entropy:{}, mean entropy:{}'.format(unseen_entropy.max(), unseen_entropy.min(),
                                                                          unseen_entropy.mean()))
    # 4.12, 0.375, 2.52

    # assert 1==2

    threshold = -0.5
    for i in range(500):
        threshold = threshold + 0.003 * i
        split_1 = (seen_entropy < threshold).float().mean().item()
        split_2 = (unseen_entropy > threshold).float().mean().item()
        # print(split_1, split_2)
        print(split_1, split_2, 2 * (split_1 * split_2) / (split_1 + split_2))



def split_patch(img_batch, label_batch):
    b, c, w, h = img_batch.size()
    pathc_list = []
    # print(b)
    for i in range(b):
        # print(b)
        img = img_batch[i]
        # print(img.size())
        pathc_list.append(img[:, :224, :224].unsqueeze(dim=0))
        pathc_list.append(img[:, :224, 224:].unsqueeze(dim=0))
        pathc_list.append(img[:, 224:, :224].unsqueeze(dim=0))
        pathc_list.append(img[:, 224:, 224:].unsqueeze(dim=0))

    patch_tensor = t.cat(pathc_list, dim=0)

    label_batch = label_batch.reshape(-1, 1).repeat(1, 4).view(-1)

    return patch_tensor, label_batch

