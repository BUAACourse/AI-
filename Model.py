import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from collections import OrderedDict
from Model_Base import *







#### Best for RUL
class FC_STGNN_RUL(nn.Cell):
    def __init__(self, indim_fea, Conv_out, lstmhidden_dim, lstmout_dim, conv_kernel,hidden_dim, time_length, num_node, num_windows, moving_window,stride,decay, pooling_choice, n_class):
        super(FC_STGNN_RUL, self).__init__()
        # graph_construction_type = args.graph_construction_type
        self.nonlin_map = Feature_extractor_1DCNN_RUL(1, lstmhidden_dim, lstmout_dim,kernel_size=conv_kernel)
        self.nonlin_map2 = nn.SequentialCell(
            nn.Dense(lstmout_dim*Conv_out, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim)
        )

        self.positional_encoding = PositionalEncoding(2*hidden_dim,0.1,max_len=5000)

        self.MPNN1 = GraphConvpoolMPNN_block_v6(2*hidden_dim, hidden_dim, num_node, time_length, time_window_size=moving_window[0], stride=stride[0], decay = decay, pool_choice=pooling_choice)
        self.MPNN2 = GraphConvpoolMPNN_block_v6(2*hidden_dim, hidden_dim, num_node, time_length, time_window_size=moving_window[1], stride=stride[1], decay = decay, pool_choice=pooling_choice)

        self.fc = nn.SequentialCell(OrderedDict([
            ('fc1', nn.Dense(hidden_dim * num_windows * num_node, 2*hidden_dim)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Dense(2*hidden_dim, 2*hidden_dim)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Dense(2*hidden_dim, hidden_dim)),
            ('relu3', nn.ReLU()),
            ('fc4', nn.Dense(hidden_dim, n_class)),
        ]))

        self.reshape = ops.Reshape()
        self.concat = ops.Concat(axis=-1)
        self.transpose = ops.Transpose()

    def construct(self, X):
        bs, tlen, num_node, dimension = X.shape ### tlen = 1

        ### Graph Generation
        A_input = self.reshape(X, (bs*tlen*num_node, dimension, 1))
        A_input_ = self.nonlin_map(A_input)
        A_input_ = self.reshape(A_input_, (bs*tlen*num_node,-1))
        A_input_ = self.nonlin_map2(A_input_)
        A_input_ = self.reshape(A_input_, (bs, tlen, num_node, -1))

        ## positional encoding before mapping starting
        X_ = self.reshape(A_input_, (bs, tlen, num_node, -1))
        X_ = self.transpose(X_, (0, 2, 1, 3))
        X_ = self.reshape(X_, (bs*num_node, tlen, -1))
        X_ = self.positional_encoding(X_)
        X_ = self.reshape(X_, (bs, num_node, tlen, -1))
        X_ = self.transpose(X_, (0, 2, 1, 3))
        A_input_ = X_

        ## positional encoding before mapping ending

        MPNN_output1 = self.MPNN1(A_input_)
        MPNN_output2 = self.MPNN2(A_input_)

        features1 = self.reshape(MPNN_output1, (bs, -1))
        features2 = self.reshape(MPNN_output2, (bs, -1))

        features = self.concat((features1, features2))

        features = self.fc(features)

        return features


#### Best for HAR
class FC_STGNN_HAR(nn.Cell):
    def __init__(self, patch_size, conv_out, lstmhidden_dim, lstmout_dim, conv_kernel, hidden_dim,
                 time_denpen_len, num_sensor, num_windows, moving_window, stride, decay, pool_choice, n_class):
        super(FC_STGNN_HAR, self).__init__()

        self.patch_size = patch_size
        self.conv_out = conv_out
        self.lstmhidden_dim = lstmhidden_dim
        self.lstmout_dim = lstmout_dim
        self.conv_kernel = conv_kernel
        self.hidden_dim = hidden_dim
        self.time_denpen_len = time_denpen_len
        self.num_sensor = num_sensor
        self.num_windows = num_windows
        self.moving_window = moving_window
        self.stride = stride
        self.decay = decay
        self.pool_choice = pool_choice
        self.n_class = n_class

        # Define layers
        self.conv1d = nn.Conv1d(1, conv_out, kernel_size=conv_kernel, pad_mode='valid')
        self.lstm = nn.LSTM(input_size=conv_out, hidden_size=lstmhidden_dim, num_layers=1, batch_first=True)
        self.linear1 = nn.Dense(lstmhidden_dim, lstmout_dim)
        self.linear2 = nn.Dense(lstmout_dim * num_sensor, n_class)

        # Define operations
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.concat = ops.Concat(axis=1)
        self.mean = ops.ReduceMean()
        self.max = ops.ReduceMax()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def construct(self, x):
        batch_size = x.shape[0]

        # Reshape input
        x = self.reshape(x, (batch_size * self.time_denpen_len * self.num_sensor, 1, self.patch_size))

        # Apply Conv1D
        x = self.conv1d(x)
        x = self.relu(x)

        # Reshape for LSTM
        x = self.reshape(x, (batch_size, self.time_denpen_len * self.num_sensor, -1))

        # Apply LSTM
        x, _ = self.lstm(x)
        x = self.tanh(x)

        # Apply linear layer
        x = self.linear1(x)

        # Pool features
        if self.pool_choice == 0:  # mean pooling
            x = self.mean(x, 1)
        else:  # max pooling
            x = self.max(x, 1)

        # Final classification
        x = self.linear2(x)

        return x



#### Best for SSC
class FC_STGNN_SSC(nn.Cell):
    def __init__(self, indim_fea, Conv_out, lstmhidden_dim, lstmout_dim, conv_kernel,hidden_dim, time_length, num_node, num_windows, moving_window,stride,decay, pooling_choice, n_class,dropout):
        super(FC_STGNN_SSC, self).__init__()
        self.nonlin_map = Feature_extractor_1DCNN_HAR_SSC(1, lstmhidden_dim, lstmout_dim,kernel_size=conv_kernel,dropout=dropout)
        self.nonlin_map2 = nn.SequentialCell(
            nn.Dense(lstmout_dim*Conv_out, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim)
        )

        self.positional_encoding = PositionalEncoding(2*hidden_dim,0.1,max_len=5000)

        self.MPNN1 = GraphConvpoolMPNN_block_v6(2*hidden_dim, hidden_dim, num_node, time_length, time_window_size=moving_window[0], stride=stride[0], decay = decay, pool_choice=pooling_choice)
        self.MPNN2 = GraphConvpoolMPNN_block_v6(2*hidden_dim, hidden_dim, num_node, time_length, time_window_size=moving_window[1], stride=stride[1], decay = decay, pool_choice=pooling_choice)

        self.fc = nn.SequentialCell(OrderedDict([
            ('fc1', nn.Dense(hidden_dim * num_windows * num_node, 2*hidden_dim)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Dense(2*hidden_dim, 2*hidden_dim)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Dense(2*hidden_dim, hidden_dim)),
            ('relu3', nn.ReLU()),
            ('fc4', nn.Dense(hidden_dim, n_class)),
        ]))

        self.reshape = ops.Reshape()
        self.concat = ops.Concat(axis=-1)
        self.transpose = ops.Transpose()

    def construct(self, X):
        bs, tlen, num_node, dimension = X.shape

        ### Graph Generation
        A_input = self.reshape(X, (bs*tlen*num_node, dimension, 1))
        A_input_ = self.nonlin_map(A_input)
        A_input_ = self.reshape(A_input_, (bs*tlen*num_node,-1))
        A_input_ = self.nonlin_map2(A_input_)
        A_input_ = self.reshape(A_input_, (bs, tlen, num_node,-1))

        ## positional encoding before mapping starting
        X_ = self.reshape(A_input_, (bs, tlen, num_node, -1))
        X_ = self.transpose(X_, (0, 2, 1, 3))
        X_ = self.reshape(X_, (bs*num_node, tlen, -1))
        X_ = self.positional_encoding(X_)
        X_ = self.reshape(X_, (bs, num_node, tlen, -1))
        X_ = self.transpose(X_, (0, 2, 1, 3))
        A_input_ = X_

        ## positional encoding before mapping ending

        MPNN_output1 = self.MPNN1(A_input_)
        MPNN_output2 = self.MPNN2(A_input_)

        features1 = self.reshape(MPNN_output1, (bs, -1))
        features2 = self.reshape(MPNN_output2, (bs, -1))

        features = self.concat((features1, features2))

        features = self.fc(features)

        return features
