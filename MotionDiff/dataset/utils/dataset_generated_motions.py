from torch.utils.data import Dataset
import numpy as np
import torch
import os
from h5py import File
import scipy.io as sio
from matplotlib import pyplot as plt


def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


class CustomH36M(Dataset):

    def __init__(self, config, path_to_data, input_n=25, output_n=100, dct_used=None):
        """
        :param path_to_data:
        :param input_n:
        :param output_n:
        """
        self.path_to_data = path_to_data
        self.input_n = input_n
        self.output_n = output_n

        if dct_used is None:
            dct_used = input_n + output_n

        # load generated motions
        data_diff = os.path.join(path_to_data, 'generate_diversity_diff.pth')
        data_gt = os.path.join(path_to_data, 'generate_diversity_gt.pth')
        data = torch.load(data_diff)
        data_gt = torch.load(data_gt)
        num_coordinate = data_gt.shape[2]

        self.all_seqs = data
        data = data.reshape(-1, input_n + output_n, data.shape[-1])
        data = data.permute(0, 2, 1)
        data = data.reshape(-1, input_n + output_n)
        all_seqs = data.transpose(0, 1)



        dct_m_in, _ = get_dct_matrix(input_n + output_n)
        input_dct_seq = np.matmul(dct_m_in[0:dct_used, :], all_seqs)
        input_dct_seq = input_dct_seq.transpose(0, 1).reshape([-1, config.nk, num_coordinate, dct_used])

        output_seq = data_gt

        self.input_dct_seq = input_dct_seq.permute(0, 1, 3, 2).contiguous()
        self.output_seq = output_seq



    def __len__(self):
        return np.shape(self.input_dct_seq)[0]

    def __getitem__(self, item):
        return self.input_dct_seq[item], self.output_seq[item], self.all_seqs[item]
