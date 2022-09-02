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



class CategoriesSampler(object):

    def __init__(self, label_for_imgs, n_batch, n_cls, n_per, ep_per_batch=1):
        # super(CategoriesSampler, self).__init__(data_source=None)
        self.n_batch = n_batch # batchs for each epoch
        self.n_cls = n_cls # ways
        self.n_per = n_per # shots
        self.ep_per_batch = ep_per_batch # episodes for each batch, defult set 1

        self.cat =  list(np.unique(label_for_imgs))
        self.catlocs = {}

        for c in self.cat:
            self.catlocs[c] = np.argwhere(label_for_imgs == c).reshape(-1)


    def __len__(self):
        return  self.n_batch

    def __iter__(self):
        
        for i_batch in range(self.n_batch):
            batch = []
            for i_ep in range(self.ep_per_batch):
                episode = []
                selected_classes = np.random.choice(self.cat, self.n_cls, replace=False)

                for c in selected_classes:
                    l = np.random.choice(self.catlocs[c], self.n_per, replace=True)
                    episode.append(t.from_numpy(l))
                episode = t.stack(episode)
                batch.append(episode)

            batch = t.stack(batch)  # bs * n_cls * n_per
            breakpoint()

            yield batch.view(-1)