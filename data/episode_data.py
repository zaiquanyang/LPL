import os
import platform
import argparse
import sys
import pickle
from scipy import io as io
from PIL import Image
import random

import numpy as np
import torch as t
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms


from .class_sample_tool import CategoriesSampler


# init data path

linux_data_root = '/data/code/ZSL_Data'                            # 存放全部数据集的目录

windows_data_root = 'D:/File/Dataset'


dataset_mat_dir = linux_data_root if platform.system() == 'Linux' else windows_data_root  # 提取好的特征数据集路径
dataset_img_dir = linux_data_root if platform.system() == 'Linux' else windows_data_root  # 数据集图片数据集路径


class Train_Dataset(Dataset):

    def __init__(self, cfg, img_path, atts, small_labels, big_labels):
        # print('train data size:', len(img_path))
        self.img_path = img_path
        self.atts = t.from_numpy(atts).float()
        self.s_labels = t.from_numpy(small_labels).long()

        self.b_labels = t.from_numpy(big_labels.astype(float)).long()
        self.img_size = cfg.image_size
        # self.classes = np.unique(small_labels)

        self.transforms = transforms.Compose([transforms.Resize(int(self.img_size * 8. / 7.)),
                                              # transforms.RandomResizedCrop(448, (0.2, 1), (0.75, 4.0 / 3)),
                                              # transforms.RandomRotation((0, 90)),
                                              transforms.RandomCrop(self.img_size),
                                              transforms.RandomHorizontalFlip(0.5),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    def __getitem__(self, index):
        img_path = self.img_path[index]
        img = Image.open(img_path).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)

        s_label = self.s_labels[index]
        b_label = self.b_labels[index]
        att = self.atts[index]

        return img, att, s_label, b_label, img_path

    def __len__(self):
        return self.s_labels.size(0)


class TestDataset(Dataset):

    def __init__(self, cfg, img_path, att, small_labels, big_labels):
        self.img_path = img_path
        # self.atts = t.Tensor(att).float()
        self.atts = att.astype(float)

        # self.s_labels = t.Tensor(small_labels).long()
        # self.b_labels = t.Tensor(big_labels.astype(float)).long()
        self.s_labels = small_labels.astype(int)
        self.b_labels = big_labels.astype(int)

        # self.classes = np.unique(self.s_labels)
        self.img_size = cfg.image_size
        self.transforms = transforms.Compose([
            transforms.Resize(int(self.img_size * 8. / 7.)),
            transforms.CenterCrop(self.img_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        img_path = self.img_path[index]
        img = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)

        s_label = self.s_labels[index]
        b_label = self.b_labels[index]
        att = self.atts[index]
        # print(type(img))
        # print(img.is_cuda)
        # return img, att, b_label, img_path

        return img, s_label, b_label   #,   img_path

    def __len__(self):
        return self.s_labels.shape[0]


# class Patch_Img_Dataset(Dataset):
#     def __init__(self, img_path, att, small_labels, big_labels):
#         self.img_path = img_path
#         # self.atts = t.Tensor(att).float()
#         self.s_labels = t.Tensor(small_labels).long()
#         self.b_labels = t.Tensor(big_labels).long()
#         # self.classes = np.unique(self.s_labels)
#
#         rand_scale_size = int(448 * random.uniform(1.0, 1.0))
#         rand_crop_size = rand_scale_size * random.uniform(0.33, 0.66)
#
#         self.transforms = transforms.Compose([
#             transforms.Resize(rand_scale_size),
#             transforms.RandomCrop(rand_crop_size),                               # patch img size
#             # transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
#
#     def __getitem__(self, index):
#         img_path = self.img_path[index]
#         img = Image.open(img_path).convert('RGB')
#         if self.transforms is not None:
#             img = self.transforms(img)
#
#         s_label = self.s_labels[index]
#         b_label = self.b_labels[index]
#         # att = self.atts[index]
#
#         return img, s_label, b_label
#
#     def __len__(self):
#         return self.s_labels.size(0)



class  ZSL_Episode_Img_Dataset(object):

    def __init__(self, cfg):
        self.cfg = cfg
        mat_dir = dataset_mat_dir

        self.mat_embeddings = io.loadmat(os.path.join(mat_dir, 'xlsa17/data', self.cfg.dataset_name, 'res101.mat'))
        self.mat_attr = io.loadmat(os.path.join(mat_dir, 'xlsa17/data', self.cfg.dataset_name, 'att_splits.mat'))
        self.img_files = np.squeeze(self.mat_embeddings['image_files'])


        self.raw_attr = self.mat_attr['att'].T
        # if cfg.USE_1024_ATTR:
        #     self.raw_attr = io.loadmat(os.path.join(mat_dir, 'xlsa17/data', self.cfg.dataset_name, 'sent_splits.mat'))['att'].T  # 1024 x attr_dim
        # self.raw_attr = F.normalize(self.raw_attr, dim=-1, p=2)
        self.label = self.mat_embeddings['labels'].squeeze() - 1            # !!!!!!!!!!

        self.img_root = dataset_img_dir
        local_img_files = []

        for img_file in self.img_files:
            img_path = img_file[0]
            if cfg.dataset_name == 'CUB':
                img_path = os.path.join(self.img_root, cfg.dataset_name, '/'.join(img_path.split('/')[7:]))
            elif cfg.dataset_name == 'AWA2':
                eff_path = img_path.split('/')[8:]
                img_path = os.path.join(self.img_root, cfg.dataset_name, 'images', '/'.join(eff_path))
            elif cfg.dataset_name == 'SUN':
                img_path = os.path.join(self.img_root, cfg.dataset_name, '/'.join(img_path.split('/')[7:]))
            elif cfg.dataset_name == 'FLO':
                img_path = os.path.join(self.img_root, cfg.dataset_name, '/'.join(img_path.split('/')[7:]))
            elif cfg.dataset_name == 'APY':
                img_path = os.path.join(self.img_root, cfg.dataset_name, 'images', '/'.join(img_path.split('/')[-3:]))
            local_img_files.append(img_path)

        local_img_files = np.array(local_img_files)
        # train_idx = self.mat_attr['trainval_loc'].squeeze() - 1             # !!!!!!!

        train_idx = (self.mat_attr['trainval_loc'].squeeze() - 1)[:]             # !!!!!!!
        test_seen_idx = (self.mat_attr['test_seen_loc'].squeeze() - 1)[:]        # !!!!!!!
        test_unseen_idx = (self.mat_attr['test_unseen_loc'].squeeze() - 1)[:]    # !!!!!!!
        print('train data {} / test seen data {} / test unseen data {}'.format(len(train_idx), len(test_seen_idx), len(test_unseen_idx)))

        train_img = local_img_files[train_idx]
        test_seen_img = local_img_files[test_seen_idx]
        test_unseen_img = local_img_files[test_unseen_idx]

        train_big_label = self.label[train_idx]
        test_seen_big_label = self.label[test_seen_idx]
        test_unseen_big_label = self.label[test_unseen_idx]

        train_all_att = self.raw_attr[train_big_label]
        test_seen_all_att = self.raw_attr[test_seen_big_label]
        test_unseen_all_att = self.raw_attr[test_unseen_big_label]

        seen_class_big_idx, train_small_label = np.unique(train_big_label, return_inverse=True)

        seen_class_big_idx, test_seen_small_label = np.unique(test_seen_big_label, return_inverse=True)

        unseen_class_big_idx, test_unseen_small_label = np.unique(test_unseen_big_label, return_inverse=True)
        self.seen_attr = self.raw_attr[seen_class_big_idx]
        self.unseen_attr = self.raw_attr[unseen_class_big_idx]
        self.seen_class_big_idx = seen_class_big_idx
        self.unseen_class_big_idx = unseen_class_big_idx

        # print(self.mat_attr['allclasses_names'][unseen_class_big_idx])
        #
        # breakpoint()

        if cfg.DATALOADER_MODE == 'RANDOM':
            dataset = Train_Dataset(cfg, train_img, train_all_att, train_small_label, train_big_label)
            sampler = t.utils.data.sampler.RandomSampler(dataset)
            batch = cfg.ways * cfg.shots
            batch_sampler = t.utils.data.sampler.BatchSampler(sampler, batch_size=batch, drop_last=True)
            tr_dataloader = t.utils.data.DataLoader(dataset=dataset, num_workers=0, batch_sampler=batch_sampler, pin_memory=False)

        elif cfg.DATALOADER_MODE == 'EPISODE':
            dataset = Train_Dataset(cfg, train_img, train_all_att, train_small_label, train_big_label)
            batch_sampler = CategoriesSampler(
                train_small_label,
                cfg.coarse_iter_num,
                cfg.ways,
                cfg.shots,
                1,
            )
            tr_dataloader = DataLoader(dataset=dataset, batch_sampler=batch_sampler, num_workers=0, pin_memory=False)

        else:
            tr_dataloader = None
            print('not surpport sample mode')

        tr_data = TestDataset(cfg, train_img, train_all_att, train_small_label, train_big_label)
        if platform.system() == 'Linux':
            tr_loader = t.utils.data.DataLoader(tr_data, batch_size=32, shuffle=False, num_workers=2, pin_memory=False)
        else:
            tr_loader = t.utils.data.DataLoader(tr_data, batch_size=32, shuffle=False, num_workers=0,  pin_memory=False)

        # # test unseen dataloader
        tu_data = TestDataset(cfg, test_unseen_img, test_unseen_all_att, test_unseen_small_label, test_unseen_big_label)
        if platform.system() == 'Linux':
            tu_loader = t.utils.data.DataLoader(tu_data, batch_size=32, shuffle=False, num_workers=2, pin_memory=False)
        else:
            tu_loader = t.utils.data.DataLoader(tu_data, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)

        # test seen dataloader
        ts_data = TestDataset(cfg, test_seen_img, test_seen_all_att, test_seen_small_label, test_seen_big_label)
        if platform.system() == 'Linux':
            ts_loader = t.utils.data.DataLoader(ts_data, batch_size=32, shuffle=False, num_workers=2, pin_memory=False)
        else:
            ts_loader = t.utils.data.DataLoader(ts_data, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)

        self.train_dataloader = tr_dataloader
        self.train_tmp_dataloader = tr_loader
        self.test_seen_dataloader = ts_loader
        self.test_unseen_dataloader = tu_loader

        self.seen_class_indices_inside_test = \
            {c: [i for i in range(test_seen_big_label.shape[0]) if test_seen_big_label[i] == c] for c in seen_class_big_idx}
        self.unseen_class_indices_inside_test = \
            {c: [i for i in range(test_unseen_big_label.shape[0]) if test_unseen_big_label[i] == c] for c in unseen_class_big_idx}

    # def forward(self):





