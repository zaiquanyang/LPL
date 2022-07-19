import torch as t
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import time


class get_CAM(nn.Module):
    def __init__(self, pretrain_weight):
        super(get_CAM, self).__init__()

        self.weight = pretrain_weight # cls x 2048

    def forward(self, x_fea_map):

        b, c, h, w = x_fea_map.shape

        x_fea_map = self.weight[0] * x_fea_map.reshape(b, h*w, c)

        CAM_mask = x_fea_map.sum(dim=0)          # b x hw



def get_CAM(x_fea_map, pre_seen_scores, seen_att, W_weight):
    """

    :param x_fea_map: batch_size x C x w x h
    :param pre_seen_scores: batch_size x seen_cls_num
    :param seen_att: seen_cls_num x attr_num
    :return:
    """

    with t.no_grad():

        batch_size, C, w, h = x_fea_map.shape

        max_val, pred_idx = pre_seen_scores.max(dim=1)  # pre_idx 是预测的类别label

        # consider predict results
        pre_attr = seen_att[pred_idx]                       # batch_size x attr_num
        batch_weight = t.mm(pre_attr, W_weight)             # batch_size x 2048

        # do not consider predict results
        # batch_weight = W_weight.sum(dim=0).unsqueeze(dim=0) # 1 x 2048

        # calculate the CAM
        # (batch_size x 2048) x (batch_size x 2048 x w x h) = (batch_size x 2048 x w x h)
        weight_fea_map = batch_weight.view(batch_size, C, 1, 1) * x_fea_map

        # (batch_size x 2048 x w x h) --> (batch_size x w x h)
        fea_mask = weight_fea_map.sum(dim=1)

        # breakpoint()
        # normalize the CAM
        # print('CAM:{}'.format(fea_mask.reshape(batch_size, w*h).sum(-1).shape))
        fea_mask = fea_mask / (fea_mask.reshape(batch_size, w*h).sum(-1) + 1e-12).view(batch_size, 1, 1)


    return fea_mask



def Wasserstein_loss(x_fea_map, fea_mask, lam=5.0, epsilon=1e-8):
    """
    :param x_fea_map: batch_size x C x w x h
    :param fea_mask: batch_size x w x h or x 1 x w x h
    :param lam:
    :param epsilon:
    :return:
    """

    batch_size, c , w, h = x_fea_map.shape
    x_fea_map = x_fea_map.reshape(batch_size, w*h, c)
    x_fea_map = F.normalize(x_fea_map, dim=-1, p=2, eps=1e-12)

    split_index = t.arange(0, batch_size, 2).long().cuda() if t.cuda.is_available() else print('no cuda.')
    x_fea_map_s = x_fea_map[split_index]         # batch_size/2 x (w*h) x C
    x_fea_map_t = x_fea_map[split_index+1]       # batch_size/2 x (w*h) x C

    fea_mask_s = fea_mask[split_index]           # batch_size/2 x w x h
    fea_mask_t = fea_mask[split_index+1]         # batch_size/2 x w x h

    cost_matrix = t.matmul(x_fea_map_s, x_fea_map_t.transpose(2, 1))  # batch_size/2 x wh x wh

    # cost_matrix.mean().backward()
    #
    # assert 1==2
    # print('cost matrix shape: {}'.format(cost_matrix.shape))   # 8 x 49 x 49
    _, wh, wh = cost_matrix.shape
    cost_matrix = 1. - cost_matrix

    mu_s = (fea_mask_s/(fea_mask_s.view(int(batch_size/2), w*h).sum(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1) + 1e-12)).reshape(int(batch_size/2), w*h)       # batch_size/2 x wh
    mu_t = (fea_mask_t/(fea_mask_t.view(int(batch_size/2), w*h).sum(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1) + 1e-12)).reshape(int(batch_size/2), w*h)       # batch_size/2 x wh

    P = t.exp(-lam * cost_matrix)                       # batch_size/2 x wh x wh
    P = P / (P.sum()+1e-12)
    u = t.zeros(int(batch_size/2), wh).cuda()                # batch_size/2 x wh


    # normalize this matrix
    while t.max(t.abs(u - P.sum(2))) > epsilon:
        u = P.sum(2) + 1e-12
        P = P * (mu_s / u).reshape((int(batch_size/2), -1, 1))         # 行归r化，注意python中*号含义
        P = P * (mu_t / (P.sum(1) + 1e-12)).reshape((int(batch_size/2), 1, -1))  # 列归c化

        epsilon = epsilon * 2.0

    return P, t.sum(P * cost_matrix) * 2./batch_size

    # batch_size, c, w, h = x_fea_map.shape
    # x_fea_map = x_fea_map.reshape(batch_size, w * h, c)
    # x_fea_map = F.normalize(x_fea_map, dim=-1, p=2)
    #
    # x_fea_map_s = x_fea_map[0]  # (w*h) x C
    # x_fea_map_t = x_fea_map[1]  # (w*h) x C
    #
    # fea_mask_s = fea_mask[0]  # w x h
    # fea_mask_t = fea_mask[1]  # w x h
    #
    # cost_matrix = t.mm(x_fea_map_s, x_fea_map_t.transpose(1, 0))  # wh x wh
    # # 49 x 49
    # wh, wh = cost_matrix.shape
    # cost_matrix = 1. - cost_matrix
    #
    # mu_s = (fea_mask_s / (fea_mask_s.sum().unsqueeze(dim=-1).unsqueeze(dim=-1))).reshape(w * h)  #  wh
    # mu_t = (fea_mask_t / (fea_mask_t.sum().unsqueeze(dim=-1).unsqueeze(dim=-1))).reshape(w * h)  #  wh
    #
    # P = t.exp(-lam * cost_matrix)  #  wh x wh
    # P /= P.sum()
    # u = t.zeros(wh).cuda()  # wh
    #
    # # normalize this matrix
    # while t.max(t.abs(u - P.sum(1))) > epsilon:
    #     u = P.sum(1)
    #     P = P * (mu_s / u).reshape((-1, 1))  # 行归r化，注意python中*号含义
    #     P = P * (mu_t / P.sum(0)).reshape((1, -1))  # 列归c化
    #
    #     epsilon *= 2.0
    #
    # return P, t.sum(P * cost_matrix)




if __name__ == "__main__":

    x_fea_map, fea_mask = t.rand(16, 2048, 7, 7).cuda(), t.rand(16, 7, 7).cuda()
    start = time.time()
    p, w_loss = Wasserstein_loss(x_fea_map=x_fea_map, fea_mask=fea_mask)
    end = time.time()
    print('time: {}'.format(end-start))
    print(w_loss)