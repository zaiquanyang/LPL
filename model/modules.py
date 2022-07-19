import torch as t
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from utils.tool import *
from model.grl_layer import WarmStartGradientReverseLayer


t.set_printoptions(
    precision=4,    # 精度，保留小数点后几位，默认4
    threshold=1000,
    edgeitems=3,
    linewidth=1500,  # 每行最多显示的字符数，默认80，超过则换行显示
    profile=None,
    sci_mode=False  # 用科学技术法显示数据，默认True
)


np.set_printoptions(suppress=True)  # 取消科学计数法输出
np.set_printoptions(precision=4)    # 设置输出精度


class Class_Standardization(nn.Module):
    """
    Class Standardization from paper ICLR2021  CLASS NORMALIZATION FOR (CONTINUAL) GENERALIZED ZERO-SHOT LEARNING
    Conceptually, it is equivalent to nn.BatchNorm1d with affine=False,
    but for some reason nn.BatchNorm1d performs slightly worse.
    """
    def __init__(self, feat_dim: int):
        super().__init__()

        self.running_mean = nn.Parameter(t.zeros(feat_dim), requires_grad=False)
        self.running_var = nn.Parameter(t.ones(feat_dim), requires_grad=False)

    def forward(self, class_fea):
        """
        Input: class_feats of shape [num_classes, feat_dim]
        Output: class_feats (standardized) of shape [num_classes, feat_dim]
        """
        if self.training:
            batch_mean = class_fea.mean(dim=0)
            batch_var = class_fea.var(dim=0)
            # Normalizing the batch
            result = (class_fea - batch_mean.unsqueeze(0)) / (batch_var.unsqueeze(0) + 1e-5)
            # Updating the running mean/std
            self.running_mean.data = 0.9 * self.running_mean.data + 0.1 * batch_mean.detach()
            self.running_var.data = 0.9 * self.running_var.data + 0.1 * batch_var.detach()
        else:
            print(class_fea.size(), self.running_mean.size())
            result = (class_fea - self.running_mean.unsqueeze(0)) / (self.running_var.unsqueeze(0) + 1e-5)

        return result


class Attr_Standardization(nn.Module):

    def __init__(self, feat_dim=312):
        super().__init__()

        self.running_mean = nn.Parameter(t.zeros(feat_dim), requires_grad=False)
        self.running_var = nn.Parameter(t.ones(feat_dim), requires_grad=False)

    def forward(self, pre_attr):

        if self.training:
            batch_mean = pre_attr.mean(dim=0)
            batch_var = pre_attr.var(dim=0)
            # Normalizing the batch
            result = (pre_attr - batch_mean.unsqueeze(0)) / (batch_var.unsqueeze(0) + 1e-5)
            # Updating the running mean/std
            self.running_mean.data = 0.9 * self.running_mean.data + 0.1 * batch_mean.detach()
            self.running_var.data = 0.9 * self.running_var.data + 0.1 * batch_var.detach()
        else:
            # print(pre_attr.size(), self.running_mean.size())
            result = (pre_attr - self.running_mean.unsqueeze(0)) / (self.running_var.unsqueeze(0) + 1e-5)

        return result



class Attr_Decouple_Layer(nn.Module):

    def __init__(self, attr_dim, static=False, scale=1.0):
        super(Attr_Decouple_Layer, self).__init__()

        self.attr_dim = attr_dim
        self.static = static
        self.scale = scale

        if not self.static:
            weight = t.FloatTensor(1, self.attr_dim).fill_(1.0)
            self.attr_var_scale = nn.Parameter(weight, requires_grad=True)
        else:
            weight = t.FloatTensor(1, self.attr_dim).fill_(self.scale)
            self.attr_var_scale = nn.Parameter(weight, requires_grad=False)

        self.running_mean = nn.Parameter(t.zeros(self.attr_dim), requires_grad=False)
        # self.running_var = nn.Parameter(t.ones(self.attr_dim), requires_grad=False)

    def forward(self, pre_attr):

        if self.training:
            batch_mean = pre_attr.mean(dim=0)
            # decouping the mean and var
            result = (pre_attr - batch_mean.unsqueeze(0)) * self.attr_var_scale + batch_mean.unsqueeze(0)

            self.running_mean.data = 0.9 * self.running_mean.data + 0.1 * batch_mean.detach()

        else:
            result = pre_attr

        return result


class CrossEntropyLoss_label_smooth(nn.Module):
    """
    A placeholder identity operator that is argument-insensitive.
    """
    def __init__(self, num_classes=10, epsilon=0.1):
        super(CrossEntropyLoss_label_smooth, self).__init__()

        self.num_cls = num_classes
        self.epi = epsilon

    def forward(self, outputs, targets):

        N = targets.size(0)
        # torch.Size([8, 10])
        smoothed_labels = t.full(size=(N, self.num_cls), fill_value=self.epi / (self.num_cls - 1)).cuda()
        targets = targets.data

        # 为矩阵中的每一行的某个index的位置赋值为1 - epsilon
        smoothed_labels.scatter_(dim=1, index=t.unsqueeze(targets, dim=1).cuda(), value=1 - self.epi)
        # print(smoothed_labels.size())
        # print(outputs.size())
        log_prob = nn.functional.log_softmax(outputs, dim=1)
        # print(log_prob.size())
        # 用之前得到的smoothed_labels来调整log_prob中每个值
        loss = - t.sum(log_prob * smoothed_labels) / N

        return loss



class Identity(nn.Module):
    """
    A placeholder identity operator that is argument-insensitive.
    """
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):

        return input




class Contrastive_Embedding_Loss(nn.Module):
    def __init__(self, cfg, temperature=0.1):
        super(Contrastive_Embedding_Loss, self).__init__()

        self.temperature = temperature
        self.cfg = cfg

    def forward(self, x_in, labels):
        # breakpoint()
        x_in_a, x_in_b = x_in
        labels_a, labels_b = labels
        
        batch_size = x_in[0].shape[0]
        
        labels_a = labels_a.contiguous().view(-1, 1)
        labels_b = labels_b.contiguous().view(-1, 1)
        # mask = t.eq(labels, labels.T).float().to(self.cfg.device)
        mask = t.eq(labels_a, labels_b.t()).float().cuda()

        # compute logits
        anchor_dot_contrast = t.div(t.matmul(x_in_a, x_in_b.t()), self.temperature)

        # reprocess
        logit_max, _ = t.max(anchor_dot_contrast, dim=1, keepdim=True)
        logit = anchor_dot_contrast - logit_max

        # logit_max = t.scatter(t.ones_like(mask), 1, t.arange(batch_size).view(-1, 1).to(self.cfg.device), 0)
        logit_mask = t.scatter(t.ones_like(mask), 1, t.arange(batch_size).view(-1, 1).cuda(), 0)

        mask = mask * logit_mask


        # compute log probability
        exp_logits = t.exp(logit) * logit_mask
        logit_prob = logit - t.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over posotive
        mean_log_prob_pos = (mask * logit_prob).sum(1) / mask.sum(1)

        # compute the loss
        loss = -mean_log_prob_pos
        loss = loss.mean()

        return loss


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    图注意力层
    """

    def __init__(self, ways, shots, in_features, out_features, dropout, alpha, concat=False):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features    # 节点表示向量的输入特征维度
        # self.out_features = out_features  # 节点表示向量的输出特征维度
        self.dropout = dropout            # dropout参数
        self.ways = ways
        self.shots = shots
        self.concat = concat              # 如果为true, 再进行elu激活

        self.temp = alpha

        # self.W1 = nn.Parameter(t.zeros(size=(in_features, out_features)).float())
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        # self.W2 = nn.Parameter(t.ones(size=(in_features, out_features)).float())
        # nn.init.xavier_normal_(self.W2.data, gain=1.414)  # xavier初始化
        # self.W2.data.normal_(0, 0.02)
        self.sparse_softmax = Sparsemax(dim=1)

        self.relu = nn.Tanh()

    def forward(self, node_x):
        """
        inp: input_fea [shots, ways, in_features]  in_features表示节点的输入特征向量元素个数
        adj: 图的邻接矩阵  [ways, ways] 非零即一，数据结构基本知识
        """

        # node_x = node_x.reshape(self.ways, self.shots, -1).transpose(1, 0)
        shots, ways, n_dim = node_x.shape
        # node_x = node_x.reshape(shots*ways, -1)        # (shots x ways) x n_dim
        node_x = F.normalize(node_x, dim=-1)

        node_x_F = node_x

        node_sim = []
        for shot in range(shots):
            x_shot = node_x_F[shot]                                         # ways x n_dim
            x_shot_T = x_shot.transpose(1, 0)                               # n_dim x ways
            shot_sim = t.mm(x_shot, x_shot_T)                               # ways x ways
            node_sim.append(shot_sim.unsqueeze(dim=0))                      # 1 x ways x ways
        node_sim = t.cat(node_sim, dim=0)                                   # shots x ways x ways


        return node_sim



    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class GMIX(nn.Module):
    def __init__(self, args, ways, shots):
        super(GMIX, self).__init__()

        self.sparse_softmax = Sparsemax(dim=1)
        self.args = args
        self.mix_scale_cls = nn.Parameter(t.FloatTensor(1).fill_(20.0).to(args.device), requires_grad=True)
        self.ways = ways
        self.shots = shots

    def forward(self, node_v_, node_s_, x_label, show, epoch):

        node_v_ = node_v_.reshape(self.ways, self.shots, -1).transpose(1, 0)  # shots x ways x vision_dim
        node_s_ = node_s_.reshape(self.ways, self.shots, -1).transpose(1, 0)  # shots x ways x attr_dim
        x_label_ = x_label.reshape(self.ways, self.shots)[:, 0].squeeze()

        v_shots, v_ways, v_dim = node_v_.shape
        s_shots, s_ways, s_dim = node_s_.shape

        # assert (v_shots == s_shots, s_ways == v_ways), print('error....')
        shots, ways = v_shots, v_ways

        diag_one_mask = t.eye(ways).to(self.args.device)  # way x ways
        diag_zero_mask = 1.0 - diag_one_mask

        if self.args.random_distribution == 'B':
            if epoch < self.args.warm_epoch:
                beta = t.from_numpy(np.random.beta(self.args.beta1, self.args.beta2, ways).reshape(1, -1).repeat(shots, axis=0).reshape(shots, ways, 1)).to(self.args.device).float()  # shots x ways
            else:
                beta = t.from_numpy(np.random.beta(1., 0.0001, ways).reshape(1, -1).repeat(shots, axis=0)
                                    .reshape(shots, ways, 1)).to(self.args.device).float()  # shots x ways

            # beta = t.from_numpy(np.random.beta(self.args.beta1, self.args.beta2, ways).reshape(1, -1).repeat(shots, axis=0).reshape(shots, ways, 1)).to(self.args.device).float()  # shots x ways


        if self.args.random_distribution == 'U':
            beta = t.from_numpy(np.random.uniform(0, self.args.beta, ways).reshape(1, -1).repeat(shots, axis=0)
                                .reshape(shots, ways, 1)).to(self.args.device).float()  # shots x ways

        node_v = F.normalize(node_v_, dim=-1)
        node_s = F.normalize(node_s_, dim=-1)

        v_sim = []
        for shot in range(shots):
            tmp = node_v[shot]                  # ways x n_dim
            tmp_T = tmp.transpose(1, 0)         # n_dim x ways
            sim = t.mm(tmp, tmp_T)              # ways x ways
            v_sim.append(sim.unsqueeze(dim=0))  # 1 x ways x ways
        v_sim = t.cat(v_sim, dim=0)             # shots x ways x ways

        s_sim = []
        for shot in range(shots):
            tmp = node_s[shot]                  # ways x n_dim
            tmp_T = tmp.transpose(1, 0)         # n_dim x ways
            sim = t.mm(tmp, tmp_T)              # ways x ways
            s_sim.append(sim.unsqueeze(dim=0))  # 1 x ways x ways
        s_sim = t.cat(s_sim, dim=0)             # shots x ways x ways

        new_v_sim_ = preprocess_node(v_sim, show=False)
        new_s_sim_ = preprocess_node(s_sim, show=False)


        # new_sim = (1-gama) * (1-sim) + (gama) * (sim)
        new_v_sim = ((1 - self.args.gama) * (1 - new_v_sim_ + diag_one_mask * -1e4)+
                     self.args.gama * (new_v_sim_ + diag_one_mask * -1e4)).reshape(shots * ways, -1)
        new_s_sim = ((1 - self.args.gama) * (1 - new_s_sim_ + diag_one_mask * -1e4) +
                     self.args.gama * (new_s_sim_ + diag_one_mask * -1e4)).reshape(shots * ways, -1)

        v_softmax_sim = self.sparse_softmax(new_v_sim * self.args.temp).reshape(shots, ways, -1)
        s_softmax_sim = self.sparse_softmax(new_s_sim * self.args.temp).reshape(shots, ways, -1)
        v_softmax_sim_mean = v_softmax_sim.mean(dim=0)  # ways x ways
        s_softmax_sim_mean = s_softmax_sim.mean(dim=0)  # ways x ways



        mask = t.ones(ways, ways).to(self.args.device)
        mask_ = biased_random_walk(graph_edge_bool=mask, walk_num=self.args.walk_num)

        if self.args.GMIX == 'GMIX':
            shared_sim = (0.5 * (v_softmax_sim_mean + s_softmax_sim_mean)) * diag_zero_mask * mask_

            x_inter_fea = t.matmul(shared_sim, node_v_)
            x_inter_attr = t.matmul(shared_sim, node_s_)

            x_fea_sim = node_v_ * beta + 20.0 * x_inter_fea * (1 - beta)
            attr_sim = node_s_ * beta + 20.0 * x_inter_attr * (1 - beta)
            
        else:
            shared_sim =  diag_zero_mask * mask_
            x_inter_fea = t.matmul(shared_sim, node_v_)
            x_inter_attr = t.matmul(shared_sim, node_s_)
            x_fea_sim = node_v_ * beta + x_inter_fea * (1 - beta)
            attr_sim =  node_s_ * beta + x_inter_attr * (1 - beta)

        gmix_feat = x_fea_sim.transpose(1, 0).reshape(ways*shots, -1)  # (ways x shots) x v_dim
        gmix_attr = attr_sim.transpose(1, 0).reshape(ways*shots, -1)


        return gmix_feat, gmix_attr

def Alpha_Transform(attr_a, attr_b, alpha, tmp=0.1):
    """

    :param attr_a: input attributes for mixup 4 x 20 x 312
    :param attr_b:  input attributes for mixup 4 x 20 x 312
    :param alpha:  linear mixup coff  4 x 20 x 1
    :param tmp: temperature
    :return: alpha_t
    """
    # Identity_MAT = t.ones_like(attr_a).cuda()
    # temp = 1.0
    # attr_a = F.normalize(attr_a, dim=-1, p=2)
    # attr_b = F.normalize(attr_b, dim=-1, p=2)
    # c1a = t.exp(tmp * attr_a)
    # c1b = t.exp(tmp * attr_b)

    c2a = t.exp(-tmp*attr_a)
    c2b = t.exp(-tmp*attr_b)

    # c_alpha_1a = t.exp(tmp*alpha*attr_a)
    c_alpha_1b = t.exp(tmp*alpha*attr_b)

    c_alpha_2a = t.exp(-tmp * alpha * attr_a)
    # c_alpha_2b = t.exp(-tmp * alpha * attr_b)
    # print((c2a + c2b + c2a * c2b - c_alpha_2a * c_alpha_1b * c2b).mean())
    # print((1 + c_alpha_2a * c_alpha_1b * c2b).mean())
    # print((c2b-c2a).mean())
    alpha_t = ((c2a + c2b + c2a * c2b - c_alpha_2a * c_alpha_1b * c2b)/(1 + c_alpha_2a * c_alpha_1b * c2b) - c2a)/(c2b-c2a+1e-12)

    # print("alpha_t size: {}".format(alpha_t))

    return alpha_t



class AdversarialLoss(nn.Module):
    def __init__(self, cfg, classifier, seen_att):
        super(AdversarialLoss, self).__init__()

        self.GRL_Layer = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=300, auto_step=True)
        self.classifier = nn.Sequential(
            classifier,
            # nn.BatchNorm1d(seen_att.size(-1)),
            # nn.ReLU()
        )
        self.seen_att = seen_att
        self.cfg = cfg

    def forward(self, x, gt_s_labels):
        """

        :param x:
        :param gt_s_labels:
        :return:
        """
        x_grl = self.GRL_Layer(x)
        y_att = self.classifier(x_grl)                  # 注意这里输出的不是普通的分类结果 batch_size x cls_num, 而是 batch_size x attr_num

        y_att = F.normalize(y_att, p=2, dim=-1)
        seen_att = F.normalize(self.seen_att, p=2, dim=-1)
        # seen_att = self.seen_att

        y_output = 20. * t.mm(y_att, seen_att.t())      # 注意这里输出的才是分类结果 batch_size x cls_num

        loss_adv = get_intra_pair_loss(labels=gt_s_labels, outputs=y_output, temp_factor=self.cfg.JS_Softmax_Temp)
        # print('raw loss_pdd:{}'.format(loss_adv))
        return loss_adv * (self.cfg.JS_Softmax_Temp ** 2)

def get_intra_pair_loss(labels, outputs, temp_factor):
    loss = 0.0
    count = 0
    batch_size = labels.size(0)
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            if labels[i] == labels[j]:
                count += 1
                loss += JS_Divergence_With_Temperature(outputs[i], outputs[j], temp_factor)
                # print('JS_loss:{}'.format(loss))

    if count == 0:
        return loss
    else:
        return loss / count


def JS_Divergence_With_Temperature(p, q, temp_factor, get_softmax=True):
    KLDivLoss = nn.KLDivLoss(reduction='sum')
    if get_softmax:
        p_softmax_output = F.softmax(p / temp_factor)
        q_softmax_output = F.softmax(q / temp_factor)
    log_mean_softmax_output = ((p_softmax_output + q_softmax_output) / 2).log()
    return (KLDivLoss(log_mean_softmax_output, p_softmax_output) + KLDivLoss(log_mean_softmax_output, q_softmax_output)) / 2



def Patch_KL_loss(patch_pre_attr, label, temp_factor):

    # patch_pre_attr
    loss = 0.0
    count = 0
    batch_size = label.size(0)

    for i in range(batch_size):
        for j in range(i+1, batch_size):
            if label[i] == label[j]:
                count += 1
                loss = JS_Divergence_With_Temperature(patch_pre_attr[i], patch_pre_attr[j], temp_factor)

    if count == 0:
        return loss
    else:
        return loss / count





class Coarse_GMIX(nn.Module):
    def __init__(self, args, ways, shots):
        super(Coarse_GMIX, self).__init__()

        self.sparse_softmax = Sparsemax(dim=1)
        self.args = args
        self.ways = ways
        self.shots = shots

    def forward(self, node_v_, node_s_, show):

        node_v_ = node_v_.reshape(self.ways, self.shots, -1).transpose(1, 0)  # shots x ways x vision_dim
        node_s_ = node_s_.reshape(self.ways, self.shots, -1).transpose(1, 0)  # shots x ways x attr_dim
        # x_label_ = x_label.reshape(self.ways, self.shots)[:, 0].squeeze()

        v_shots, v_ways, v_dim = node_v_.shape
        s_shots, s_ways, s_dim = node_s_.shape

        shots, ways = v_shots, v_ways

        diag_one_mask = t.eye(ways).to(self.args.device)  # way x ways
        diag_zero_mask = 1.0 - diag_one_mask

        if self.args.random_distribution == 'B':
            beta = t.from_numpy(np.random.beta(5, 1, ways).reshape(1, -1).repeat(shots, axis=0)
                                .reshape(shots, ways, 1)).to(self.args.device).float()  # shots x ways
        if self.args.random_distribution == 'U':
            beta = t.from_numpy(np.random.uniform(0, self.args.beta, ways).reshape(1, -1).repeat(shots, axis=0)
                                .reshape(shots, ways, 1)).to(self.args.device).float()  # shots x ways

        node_v = F.normalize(node_v_, dim=-1)
        node_s = F.normalize(node_s_, dim=-1)

        v_sim = []
        for shot in range(shots):
            tmp = node_v[shot]  # ways x n_dim
            tmp_T = tmp.transpose(1, 0)  # n_dim x ways
            sim = t.mm(tmp, tmp_T)  # ways x ways
            v_sim.append(sim.unsqueeze(dim=0))  # 1 x ways x ways
        v_sim = t.cat(v_sim, dim=0)  # shots x ways x ways

        s_sim = []
        for shot in range(shots):
            tmp = node_s[shot]  # ways x n_dim
            tmp_T = tmp.transpose(1, 0)  # n_dim x ways
            sim = t.mm(tmp, tmp_T)  # ways x ways
            s_sim.append(sim.unsqueeze(dim=0))  # 1 x ways x ways
        s_sim = t.cat(s_sim, dim=0)  # shots x ways x ways

        new_v_sim = preprocess_node(v_sim, show=False)
        new_s_sim = preprocess_node(s_sim, show=False)

        new_v_sim, new_s_sim = new_v_sim.reshape(shots*ways, -1), new_s_sim.reshape(shots*ways, -1)
        v_softmax_sim = self.sparse_softmax(new_v_sim * self.args.temp).reshape(shots, ways, -1)
        s_softmax_sim = self.sparse_softmax(new_s_sim * self.args.temp).reshape(shots, ways, -1)

        v_softmax_sim_mean = v_softmax_sim.mean(dim=0)  # ways x ways
        s_softmax_sim_mean = s_softmax_sim.mean(dim=0)  # ways x ways

        # mask = (v_softmax_sim_mean > 1e-3) * (s_softmax_sim_mean > 1e-3)  # ways x ways
        # mask_ = biased_random_walk(graph_edge_bool=mask, walk_num=self.args.walk_num)
        # mask = mask_ * mask

        # shared_sim = (0.5 * (v_softmax_sim_mean + s_softmax_sim_mean)) * diag_zero_mask * mask
        # shared_sim =  diag_zero_mask * mask
        one_mask = t.ones(ways, ways).to(self.args.device)
        mask_ = random_sample(graph_edge_bool=one_mask, walk_num=1)
        mask = diag_zero_mask * mask_
        # print(mask[:4])
        x_inter_fea = t.matmul(mask, node_v_)
        x_inter_attr = t.matmul(mask, node_s_)

        # x_fea_sim = 0.05 * node_v_ * beta + x_inter_fea * (1 - beta)
        # attr_sim = 0.05 * node_s_ * beta + x_inter_attr * (1 - beta)
        # print(node_v_.shape)    # 2 x 8 x 2048
        # print(beta.shape)       # 2 x 8 x 1
        # print(node_v_)
        x_fea_sim = node_v_ * beta + x_inter_fea * (1 - beta)
        attr_sim =  node_s_ * beta + x_inter_attr * (1 - beta)
        gmix_feat = x_fea_sim.transpose(1, 0).reshape(ways*shots, -1)  # (ways x shots) x v_dim
        gmix_attr = attr_sim.transpose(1, 0).reshape(ways*shots, -1)


        return gmix_feat, gmix_attr, beta


class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation

        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - t.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = t.sort(input=input, dim=dim, descending=True)[0]
        range = t.arange(start=1, end=number_of_logits + 1, step=1, dtype=input.dtype).view(1, -1).cuda()
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = t.cumsum(zs, dim)
        is_gt = t.gt(bound, cumulative_sum_zs).type(input.type())
        k = t.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (t.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = t.max(t.zeros_like(input), input - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = t.ne(self.output, 0)
        sum = t.sum(grad_output * nonzeros, dim=dim) / t.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input


class Adaptative_Dropout(nn.Module):
    def __init__(self, p=0.5):
        super(Adaptative_Dropout, self).__init__()
        self.p = p

    def forward(self, f, V2S_W):

        with t.no_grad():
            pre_attr = t.mm(f, V2S_W.t())
            pre_attr_norm = F.normalize(pre_attr, p=2, dim=-1, eps=1e-12)  # batch_size x attr_num
            #             max_, min_ = pre_attr_norm.max(dim=-1)[0].unsqueeze(dim=-1), pre_attr_norm.min(dim=-1)[0].unsqueeze(
            #                 dim=-1)  #
            #             pre_attr_mix_max = (pre_attr_norm - min_) / (max_ - min_)  # batch_size x attr_num

            #             max_, min_ = V2S_W.max(dim=-1)[0].unsqueeze(dim=-1), V2S_W.min(dim=-1)[0].unsqueeze(dim=-1)  #
            #             V2S_weight_mix_max = (V2S_W - min_) / (max_ - min_)  # attr_num x v_dim

            Compositive_p = t.mm(pre_attr_norm, V2S_W)  # batch_size x v_dim
            max_, min_ = Compositive_p.max(dim=-1)[0].unsqueeze(dim=-1), Compositive_p.min(dim=-1)[0].unsqueeze(dim=-1)
            Compositive_p = (Compositive_p - min_) / (max_ - min_)

            # stair split
            p_list = []
            p_new_list = []
            stair_list = [0.2, 0.4, 0.6, 0.8, 1.0 + 1e-6]

            for stair in stair_list:
                Compositive_p_copy = copy.deepcopy(Compositive_p)
                Compositive_p_copy = Compositive_p_copy - stair
                tmp = t.heaviside(Compositive_p_copy, t.Tensor([0.]))
                p_list.append(tmp)

            for it in range(len(p_list) - 1):
                p_new_list.append(((p_list[it] - p_list[it + 1]) * stair_list[it]).unsqueeze(dim=0))

            p_new_list = t.cat(p_new_list, dim=0).sum(dim=0)  # batch_size x v_dim

            q = t.rand(p_new_list.size(0), p_new_list.size(1)).cuda()  #
            # print(q)
            # print(p_new_list)
            mask = (q < p_new_list)

        return mask, q, p_new_list

class StochasticPooling(nn.Module):
    """
    """
    def __init__(self, *args, **kwargs):
        super(StochasticPooling, self).__init__()
        self.grid_size = 7
        self.h_choice = np.arange(self.grid_size+1)
        self.w_choice = np.arange(self.grid_size+1)
        self.num = 32
    def forward(self, feat_map):
        b, c, h, w = feat_map.size()
        assert feat_map.size(-1) == 14, print('StochasticPooling ERROR!')
        h_list = np.random.choice(self.grid_choice, self.num)
        w_list = np.random.choice(self.grid_choice, self.num)
        sample_feature_vec_list = []
        for it in range(32):
            sample_feature_map = feat_map[:, :, h_list[it]:h_list[it]+7, w_list[it]:w_list[it]+7]
            sample_feature_vec = F.avg_pool2d(sample_feature_map, kernel_size=(7, 7)).squeeze()   # b x c
            sample_feature_vec_list.append(sample_feature_vec)
        sample_feature_vec = t.cat(sample_feature_vec_list, dim=-1).reshape(b*self.num, c)  # (b*32, c)

        return sample_feature_vec



class relative_loss(nn.Module):
    def __init__(self, cfg, classifier, seen_att):
        super(relative_loss, self).__init__()
        self.cfg = cfg
        self.classifier = classifier
        self.seen_att = seen_att

        self.temp = 1.

    def forward(self, x_f, x_labels):
        pre_attr = self.classifier(x_f)

        gt_attr = self.seen_att[x_labels]
        try:
            pre_attr_softmax = F.softmax(pre_attr / self.temp, dim=-1)
            gt_attr_softmax = F.softmax(gt_attr / self.temp, dim=-1)
        except:
            print(pre_attr)
        batch_size = x_labels.size(0)
        loss = 0.0
        count = 0
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                if x_labels[i] == x_labels[j]:
                    count = count + 1

                    loss += F.kl_div((pre_attr_softmax[i]/pre_attr_softmax[j]).log(), (gt_attr_softmax[i]/gt_attr_softmax[j]), reduction='sum')

        if count > 0:
            return loss / count
        else:
            return loss




if __name__ == "__main__":
    # Contra_Embed_Loss = Contrastive_Embedding_Loss(cfg=None, temperature=0.1)
    #
    # x_labels = t.Tensor([1, 1, 2, 2, 3, 3, 4, 4]).long().cuda()
    # x_feat = t.rand(8, 5).cuda()
    #
    # loss = Contra_Embed_Loss(x_feat, x_labels)
    attr_a = t.rand(4, 20, 312)
    attr_b = t.rand(4, 20, 312)
    alpha = 1.0
    alpha_t = Alpha_Transform(attr_a_=attr_a, attr_b_=attr_b, alpha=alpha, tmp=1.0)
    print(alpha_t)
    # print(x_feat, x_labels)
    # print(loss)

