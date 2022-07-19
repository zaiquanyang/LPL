import torch as t
import torch.nn.functional as F
from sklearn.manifold import TSNE
import numpy as np

import matplotlib.pyplot as plt
import pickle
import random
from mpl_toolkits.mplot3d import Axes3D


# 可视化 V---2---S 后的空间分布

coarse_model_path = '/data/code/C2F_ZSL/saved_ck_model_coarse_mix/CUB/mix_train_backbone/CUB_coarse_backbone_10_lr_0.0005_decay_5e-07_way_8_shot_2_iter_300.pt'
# pkl_data_path = '/data/data/coarse_pkl_data_cvpr/CUB/CLS_loss/448_14_lr_0.0005_decay_5e-07_data.pkl'
pkl_data_path = '/data/data/coarse_pkl_data_cvpr/CUB/s2v_loss/448_18_lr_0.0005_decay_5e-07_data.pkl'

pkl_dataset = pickle.load(open(pkl_data_path, 'rb'), encoding='utf-8')
train_X, train_Y = pkl_dataset['train_seen']['fea'], pkl_dataset['train_seen']['labels']
test_seen_X, test_seen_Y = pkl_dataset['test_seen']['fea'], pkl_dataset['test_seen']['labels']
test_unseen_X, test_unseen_Y = pkl_dataset['test_unseen']['fea'], pkl_dataset['test_unseen']['labels']

coarse_model = t.load(coarse_model_path)['model']

V2S_Weight = coarse_model['V2S.weight'].cpu()
print('V2S weight shape:{}'.format(V2S_Weight.shape))

# train_V2S = t.mm(t.Tensor(train_X).float(), V2S_Weight.t())

test_data= F.normalize(t.Tensor(test_unseen_X), dim=-1, p=2).numpy()
test_Y = test_unseen_Y

# TSNE 可视化

def get_color_list(num):
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    color_list = []

    for k in range(num):
        color = ''
        for i in range(6):
            color += colorArr[random.randint(0,14)]
        color_list.append("#"+color)

    return color_list

def TSNE_API(data, label):
    """

    :param data: numpy array N x dim
    :param label: N
    :return:
    """
    # plt.rcParams['savefig.dpi'] = 1000  # 图片像素
    # plt.rcParams['figure.dpi'] = 800  # 分辨率
    print('data shape: {}, label shape: {}'.format(data.shape, label.shape))

    tsne = TSNE(n_components=2, init='random', random_state=100, perplexity=40, n_iter=1000)
    # data
    # normalize data
    # data_norm = F.normalize(data, p=2, dim=-1)
    data = tsne.fit_transform(data)

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data_norm = (data - x_min) / (x_max - x_min)

    # 获取要可视化的类别列表
    N = 20
    unique_cls = np.unique(label)[:N].tolist()     # 只绘制前十个类别
    plot_class_list = unique_cls
    plot_class_list = [ 79, 86, 87, 91]  # 71， 90
    N = len(plot_class_list)

    cls_num = len(plot_class_list)
    color_list = ["#000000", "#B22222", "#9400D3", "#FFA500"] # "#000000", "#8B0000",         # get_color_list(num=cls_num)

    fig = plt.figure(dpi=600)
    # ax = plt.subplot()
    ax = Axes3D(fig)

    flag = [0] * N

    for i in range(data_norm.shape[0]):
        ss = 10
        marker = 'o'
        alpha = 0.6
        if label[i] in plot_class_list:
            index = plot_class_list.index(label[i])
            ax.scatter(data_norm[i, 0], data_norm[i, 1], 0, s=10, c=color_list[index], alpha=alpha, marker=marker, label=label[i])
            if flag[index] == 0:
                ax.text(data_norm[i, 0], data_norm[i, 1], 0, str(label[i]), fontdict={'weight': 'bold', 'size': 10}, color="#000000")
                # plt.text(data_norm[i, 0], data_norm[i, 1], data_norm[i, 2], 60, str(label[i]), size=10, alpha=0.8, weight="bold", color="#000000")
                flag[index] = 1

    # plt.legend()
    ax.set_xlabel('X label')
    ax.set_ylabel('Y label')
    # ax.set_zlabel('Z label')
    # plt.axis('off')

    # bbox_inches='tight',  pad_inches=0 去除图像空白
    # plt.savefig('/data/code/C2F_ZSL/visualize/awa2_tsne_10_class_mix.png', dpi=1000, bbox_inches='tight', pad_inches=0)
    plt.show()

    return fig


if __name__ == "__main__":
    # TSNE_API(data=test_data, label=test_Y)
    model_dict = t.load('/data/code/ZSL/C2F_ZSL_CVPR/saved_model/AWA2/v2s_loss_+LeakyReLU_0.8/coarse/10_lr_0.0005_decay_0.001_way_8_shot_2_iter_100.pt')
    print(model_dict['model']['V2S.0.weight'].shape)

