import sys
sys.path.extend(['../'])

import torch
import pickle
import cv2
import numpy as np
from torch.utils.data import Dataset

from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path, length_path,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True,
                 robust_add=False, robust_drop=False, add_rate=0.0, drop_rate=0.0):
        """
        :param data_path:
        :param label_path:
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.length_path = length_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.robust_add = robust_add
        self.robust_drop = robust_drop
        self.add_rate = add_rate
        self.drop_rate = drop_rate
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
                #self.label = self.label[0:10000]
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(
                    f, encoding='latin1')
                #self.label = self.label[0:10000]
        try:
            self.length = np.load(self.length_path)
        except:
            self.length = np.zeros((self.label.shape[0],))

        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.length = self.length[0:100]
            self.sample_name = self.sample_name[0:100]

        if self.robust_add == True and self.robust_drop == True:
            raise ValueError('Test either add or drop!')
        elif self.robust_add == True:
            print('start adding')
            N, C, T, V, M = self.data.shape
            add_data = np.zeros((N, C, int(T* (1 + self.add_rate)), V, M))
            for i in range(len(self.data)):
                insert_len = int((1 + self.add_rate) * int(self.length[i].item()))
                data_numpy = self.data[i][:,:int(self.length[i].item())].transpose(3, 1, 2, 0)
                
                data_rescaled = np.zeros(
                    (data_numpy.shape[0], insert_len, data_numpy.shape[2], data_numpy.shape[3]))
                for person_id in range(data_numpy.shape[0]):
                    data_rescaled[person_id] = cv2.resize(data_numpy[person_id],
                                                          (data_numpy.shape[2],
                                                           insert_len),
                                                          cv2.INTER_LINEAR)
                
                tmp_add = data_rescaled.transpose(3, 1, 2, 0)
                rest = int(T* (1 + self.add_rate)) - tmp_add.shape[1]
                num = int(np.ceil(rest / tmp_add.shape[1]))
                
                pad = np.concatenate([tmp_add
                                      for _ in range(num + 1)], 1)[:, :int(T* (1 + self.add_rate))]
                add_data[i] = pad
                self.length[i] = int(self.length[i].item() * (1 + self.add_rate))
            self.data = add_data
            
        elif self.robust_drop == True:
            print('start dropping')
            for i in range(len(self.label)):
                drop_index = np.random.choice(range(1, int(self.length[i].item())), size=int(
                    self.drop_rate * int(self.length[i].item())), replace=False)
                # self.data[i, :, drop_index] = self.data[i, :, drop_index-1]
                tmp_deleted = np.delete(self.data[i], drop_index, 1)[
                    :, :int(self.length[i].item() * (1 - self.drop_rate))]
                try:
                    rest = self.data.shape[2] - tmp_deleted.shape[1]
                    num = int(np.ceil(rest / tmp_deleted.shape[1]))
                    pad = np.concatenate([tmp_deleted
                                          for _ in range(num + 1)], 1)[:, :self.data.shape[2]]
                    self.data[i] = pad
                except:
                    continue

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(
            axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape(
            (N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        length = self.length[index]
        data_numpy = np.array(data_numpy)

        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, label, length, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


