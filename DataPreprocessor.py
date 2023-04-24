import os
import platform
import numpy as np
import torch
from torch.utils.data import Dataset

import pickle


class DataPreprocessor:
    def __init__(self, dataset, input_time_window, output_time_window, induce_time_window=0,
                 train_ratio=0.6, valid_ratio=0.2, normalize=None, stride=1, positional_encoding='zero'):
        var_num = []
        dataset = dataset.transpose()
        for i in dataset:
            var_num.append(np.unique(i).shape[0])
        constant_var = np.where(np.array(var_num) == 1)
        self.non_constant_var = np.setdiff1d(np.arange(len(var_num)), constant_var)
        non_constant_data = dataset[self.non_constant_var]
        self.std = None
        self.mean = None
        self.min = None
        self.max = None
        if normalize == 'std':
            std = np.std(non_constant_data, axis=1).reshape([-1, 1])
            mean = np.mean(non_constant_data, axis=1).reshape([-1, 1])
            self.std = std
            self.mean = mean
            non_constant_data = (non_constant_data - mean) / std
        elif normalize == 'minmax':
            minn = np.min(non_constant_data, axis=1).reshape([-1, 1])
            maxx = np.max(non_constant_data, axis=1).reshape([-1, 1])
            self.min = minn
            self.max = maxx
            non_constant_data = (non_constant_data - minn) / (maxx - minn)
        elif normalize == 'zeromean':
            std = np.std(non_constant_data, axis=1).reshape([-1, 1])
            mean = np.mean(non_constant_data, axis=1).reshape([-1, 1])
            self.mean = mean
            self.std = std
            non_constant_data = non_constant_data - mean
        # print('data mean', np.mean(non_constant_data,axis=1))
        # exit()
        non_constant_data = non_constant_data.transpose()
        self.num_sensors = non_constant_data.shape[1]
        total_time_window = input_time_window + output_time_window
        sample_indices = np.arange(0, non_constant_data.shape[0] - total_time_window + 1, stride)
        num_samples = sample_indices.shape[0]

        # positional encoding
        encoding_dimension = self.num_sensors
        self.tgt = None
        if positional_encoding == 'zero':
            self.tgt = np.zeros((output_time_window, encoding_dimension))
        elif positional_encoding == 'sin':
            self.tgt = np.zeros((output_time_window, encoding_dimension))
            for i in range(output_time_window):
                self.tgt[i, :] = np.sin((i + 1) / (output_time_window + 1) * np.pi)
        elif positional_encoding == 'sinusodial':
            sinusodial_encoding = np.zeros((output_time_window, encoding_dimension))
            for i in range(output_time_window):
                for j in range(encoding_dimension):
                    if j % 2 == 0:
                        sinusodial_encoding[i, j] = np.sin(i / 10000 ** (j / encoding_dimension))
                    else:
                        sinusodial_encoding[i, j] = np.cos(i / 10000 ** (j / encoding_dimension))
            self.tgt = sinusodial_encoding

        sample_index_mask = np.repeat(sample_indices, total_time_window).reshape(num_samples, -1)
        window_mask = np.arange(total_time_window).reshape(1, -1)
        sample_index_mask = sample_index_mask + window_mask
        total_sample = non_constant_data[sample_index_mask]
        input_samle = total_sample[:, :input_time_window, :]
        predict_sample = total_sample[:, input_time_window:, :]
        induce_sample = None
        if induce_time_window > 0:
            induce_sample = total_sample[:, :, input_time_window - induce_time_window:input_samle, :]
        self.train_set_input = input_samle[:int(num_samples * train_ratio)]
        self.train_set_predict = predict_sample[:int(num_samples * train_ratio)]
        self.valid_set_input = input_samle[
                               int(num_samples * train_ratio):int(num_samples * (train_ratio + valid_ratio))]
        self.valid_set_predict = predict_sample[
                                 int(num_samples * train_ratio):int(num_samples * (train_ratio + valid_ratio))]
        self.test_set_input = input_samle[int(num_samples * (train_ratio + valid_ratio)):]
        self.test_set_predict = predict_sample[int(num_samples * (train_ratio + valid_ratio)):]

        if induce_time_window > 0:
            self.train_set_induce = induce_sample[:int(num_samples * train_ratio)]
            self.valid_set_induce = induce_sample[
                                    int(num_samples * train_ratio):int(num_samples * (train_ratio + valid_ratio))]
            self.test_set_induce = induce_sample[int(num_samples * (train_ratio + valid_ratio)):]

    def load_train_data(self):
        return torch.Tensor(self.train_set_input), torch.Tensor(self.tgt), torch.Tensor(self.train_set_predict)

    def load_validate_data(self):
        return torch.Tensor(self.valid_set_input), torch.Tensor(self.tgt), torch.Tensor(self.valid_set_predict)

    def load_test_data(self):
        return torch.Tensor(self.test_set_input), torch.Tensor(self.tgt), torch.Tensor(self.test_set_predict)

    def load_num_sensors(self):
        return self.num_sensors

    def load_std_and_mean(self):
        return self.std, self.mean

    def load_min_max(self):
        return self.min, self.max


class TSDataset(Dataset):
    def __init__(self, enc_input, dec_input, dec_output):
        super(TSDataset, self).__init__()
        self.enc_input = enc_input
        self.dec_input = dec_input
        self.dec_output = dec_output

    def __getitem__(self, item):
        # n_sensors = self.enc_input.shape[1]
        # time_window = self.dec_input.shape[0]
        # hidden_size = self.dec_input.shape[1]
        enc_input = self.enc_input[item]
        dec_input = self.dec_input
        dec_output = self.dec_output[item]
        return enc_input, dec_input, dec_output

    def __len__(self):
        return self.enc_input.shape[0]
