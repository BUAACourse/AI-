import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from collections import OrderedDict



class Feature_extractor_1DCNN_RUL(nn.Cell):
    def __init__(self, input_channels, num_hidden, out_dim, kernel_size = 8, stride = 1, dropout = .0):
        super(Feature_extractor_1DCNN_RUL, self).__init__()

        self.conv_block1 = nn.SequentialCell(
            nn.Conv1d(input_channels, num_hidden, kernel_size=kernel_size,
                      stride=stride, has_bias=False, padding=(kernel_size//2), pad_mode='pad'),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(1 - dropout)
        )

        self.conv_block2 = nn.SequentialCell(
            nn.Conv1d(num_hidden, out_dim, kernel_size=kernel_size, stride=1, has_bias=False, padding=1, pad_mode='pad'),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.transpose = ops.Transpose()

    def construct(self, x_in):
        # print('input size is {}'.format(x_in.size()))
        ### input dim is (bs, tlen, feature_dim)
        x = self.transpose(x_in, (0, 2, 1))

        x = self.conv_block1(x)
        x = self.conv_block2(x)

        return x



class Feature_extractor_1DCNN_HAR_SSC(nn.Cell):
    def __init__(self, input_channels, num_hidden,embedding_dimension, kernel_size = 3, stride = 1, dropout = .0):
        super(Feature_extractor_1DCNN_HAR_SSC, self).__init__()

        self.conv_block1 = nn.SequentialCell(
            nn.Conv1d(input_channels, num_hidden, kernel_size=kernel_size,
                      stride=stride, has_bias=False, padding=(kernel_size//2), pad_mode='pad'),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1, pad_mode='pad'),
            nn.Dropout(1 - dropout)
        )

        self.conv_block2 = nn.SequentialCell(
            nn.Conv1d(num_hidden, num_hidden*2, kernel_size=kernel_size, stride=1, has_bias=False, padding=2, pad_mode='pad'),
            nn.BatchNorm1d(num_hidden*2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1, pad_mode='pad')
        )

        self.conv_block3 = nn.SequentialCell(
            nn.Conv1d(num_hidden*2, embedding_dimension, kernel_size=kernel_size, stride=1, has_bias=False, padding=3, pad_mode='pad'),
            nn.BatchNorm1d(embedding_dimension),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1, pad_mode='pad'),
        )

        self.transpose = ops.Transpose()

    def construct(self, x_in):
        # print('input size is {}'.format(x_in.size()))
        x = self.transpose(x_in, (0, 2, 1))

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        # print(x.size())
        return x


def Dot_Graph_Construction(node_features):
    ## node features size is (bs, N, dimension)
    ## output size is (bs, N, N)
    bs, N, dimen = node_features.shape

    transpose = ops.Transpose()
    bmm = ops.BatchMatMul()
    softmax = ops.Softmax(axis=-1)
    leaky_relu = nn.LeakyReLU()
    
    node_features_1 = transpose(node_features, (0, 2, 1))

    Adj = bmm(node_features, node_features_1)

    eyes_like = ops.Eye()(N, N, ms.float32)
    eyes_like = ops.Tile()(eyes_like, (bs, 1, 1))
    eyes_like_inf = eyes_like * 1e8
    Adj = leaky_relu(Adj - eyes_like_inf)
    Adj = softmax(Adj)
    # print(Adj[0])
    Adj = Adj + eyes_like
    # print(Adj[0])
    # if prior:


    return Adj

class Dot_Graph_Construction_weights(nn.Cell):
    def __init__(self, input_dim):
        super().__init__()
        self.mapping = nn.Dense(input_dim, input_dim)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = ops.Softmax(axis=-1)
        self.transpose = ops.Transpose()
        self.bmm = ops.BatchMatMul()

    def construct(self, node_features):
        node_features = self.mapping(node_features)
        # node_features = F.leaky_relu(node_features)
        bs, N, dimen = node_features.shape

        node_features_1 = self.transpose(node_features, (0, 2, 1))

        Adj = self.bmm(node_features, node_features_1)

        eyes_like = ops.Eye()(N, N, ms.float32)
        eyes_like = ops.Tile()(eyes_like, (bs, 1, 1))
        eyes_like_inf = eyes_like * 1e8
        Adj = self.leaky_relu(Adj - eyes_like_inf)
        Adj = self.softmax(Adj)
        # print(Adj[0])
        Adj = Adj + eyes_like
        # print(Adj[0])
        # if prior:

        return Adj

class Dot_Graph_Construction_weights_v2(nn.Cell):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.mapping = nn.Dense(input_dim, hidden_dim)
        self.leaky_relu = ops.LeakyReLU()
        self.softmax = ops.Softmax(axis=-1)
        self.transpose = ops.Transpose()
        self.bmm = ops.BatchMatMul()

    def construct(self, node_features):
        node_features = self.mapping(node_features)
        # node_features = F.leaky_relu(node_features)
        bs, N, dimen = node_features.shape

        node_features_1 = self.transpose(node_features, (0, 2, 1))

        Adj = self.bmm(node_features, node_features_1)

        eyes_like = ops.Eye()(N, N, ms.float32)
        eyes_like = ops.Tile()(eyes_like, (bs, 1, 1))
        eyes_like_inf = eyes_like * 1e8
        Adj = self.leaky_relu(Adj - eyes_like_inf)
        Adj = self.softmax(Adj)
        # print(Adj[0])
        Adj = Adj + eyes_like
        # print(Adj[0])
        # if prior:

        return Adj



def iDot_Graph_Construction(node_features):
    ## node features size is (bs, N, dimension)
    ## output size is (bs, N, N)
    bs, N, dimen = node_features.shape

    ##

    node_features_1 = ops.Transpose()(node_features, (0, 2, 1))

    Adj = ops.BatchMatMul()(node_features, node_features_1)

    eyes_like = ops.Eye()(N, N, ms.float32)
    eyes_like = ops.Tile()(eyes_like, (bs, 1, 1))
    eyes_like_inf = eyes_like*1e8
    Adj = nn.LeakyReLU()(Adj-eyes_like_inf)
    Adj = ops.Softmax(axis=-1)(Adj)
    Adj = Adj+eyes_like

    return Adj



class MPNN_mk(nn.Cell):
    def __init__(self, input_dimension, outpuut_dinmension, k):A_
        ### In GCN, k means the size of receptive field. Different receptive fields can be concatnated or summed
        ### k=1 means the traditional GCN
        super(MPNN_mk, self).__init__()
        self.way_multi_field = 'sum' ## two choices 'cat' (concatnate) or 'sum' (sum up)
        self.k = k
        theta = []
        for kk in range(self.k):
            theta.append(nn.Dense(input_dimension, outpuut_dinmension))
        self.theta = nn.CellList(theta)
        self.bmm = ops.BatchMatMul()
        self.concat = ops.Concat(axis=-1)

    def construct(self, X, A):
        ## size of X is (bs, N, A)
        ## size of A is (bs, N, N)
        GCN_output_ = []
        for kk in range(self.k):
            if kk == 0:
                A_ = A
            else:
                A_ = self.bmm(A_,A)
            out_k = self.theta[kk](self.bmm(A_,X))
            GCN_output_.append(out_k)

        if self.way_multi_field == 'cat':
            return self.concat(GCN_output_)

        elif self.way_multi_field == 'sum':
            out_ = GCN_output_[0]
            for kk in range(1, self.k):
                out_ += GCN_output_[kk]
            return out_


class MPNN_mk_v2(nn.Cell):
    def __init__(self, input_dimension, outpuut_dinmension, k):
        ### In GCN, k means the size of receptive field. Different receptive fields can be concatnated or summed
        ### k=1 means the traditional GCN
        super(MPNN_mk_v2, self).__init__()
        self.way_multi_field = 'sum' ## two choices 'cat' (concatnate) or 'sum' (sum up)
        self.k = k
        theta = []
        for kk in range(self.k):
            theta.append(nn.Dense(input_dimension, outpuut_dinmension))
        self.theta = nn.CellList(theta)
        self.bmm = ops.BatchMatMul()
        self.concat = ops.Concat(axis=-1)

    def construct(self, X, A):
        ## size of X is (bs, N, A)
        ## size of A is (bs, N, N)
        GCN_output_ = []
        for kk in range(self.k):
            if kk == 0:
                A_ = A
            else:
                A_ = self.bmm(A_,A)
            out_k = self.theta[kk](self.bmm(A_,X))
            GCN_output_.append(out_k)

        if self.way_multi_field == 'cat':
            return self.concat(GCN_output_)

        elif self.way_multi_field == 'sum':
            out_ = GCN_output_[0]
            for kk in range(1, self.k):
                out_ += GCN_output_[kk]
            return out_

def Graph_regularization_loss(X, Adj, gamma):
    ### X size is (bs, N, dimension)
    ### Adj size is (bs, N, N)
    
    bmm = ops.BatchMatMul()
    transpose = ops.Transpose()
    sum_op = ops.ReduceSum()
    matmul = ops.MatMul()
    mean = ops.ReduceMean()
    
    # X_distance = []
    bs, N, dim = X.shape
    X_i = X.reshape(bs*N, dim)
    X_i = ops.Tile()(X_i.reshape(bs*N, 1, dim), (1, N, 1))
    X_j = ops.Tile()(X.reshape(1, bs*N, dim), (N, 1, 1))
    X_distance = X_i - X_j
    X_distance = X_distance.reshape(bs, N, N, dim)
    return gamma * mean(sum_op(X_distance**2, -1) * Adj)



class PositionalEncoding(nn.Cell):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(float(1-dropout))

        # Compute the positional encodings once in log space.
        pe = np.zeros((max_len, d_model))
        position = np.expand_dims(np.arange(0, max_len), 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = np.expand_dims(pe, 0)
        self.pe = ms.Tensor(pe, ms.float32)

    def construct(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)

def Conv_GraphST(input, time_window_size, stride):
    ## input size is (bs, time_length, num_sensors, feature_dim)
    ## output size is (bs, num_windows, num_sensors, time_window_size, feature_dim)
    bs, time_length, num_sensor, feature_dim = input.shape
    
    padding_zeros = ops.Zeros()((bs, time_window_size, num_sensor, feature_dim), ms.float32)
    input = ops.Concat(1)((padding_zeros, input))
    
    fold_op = nn.Unfold(ksizes=[1, time_window_size, 1, 1], strides=[1, stride, 1, 1], rates=[1, 1, 1, 1], padding="VALID")
    output_unfolded = fold_op(input.transpose(0, 3, 2, 1))
    
    output = output_unfolded.reshape(bs, feature_dim, num_sensor, -1, time_window_size)
    output = output.transpose(0, 3, 2, 4, 1)
    
    return output  # (bs, num_windows, num_sensors, time_window_size, feature_dim)

def Conv_GraphST_pad(input, time_window_size, stride, padding):
    ## input size is (bs, time_length, num_sensors, feature_dim)
    ## output size is (bs, num_windows, num_sensors, time_window_size, feature_dim)
    bs, time_length, num_sensor, feature_dim = input.shape
    
    padding_zeros = ops.Zeros()((bs, padding, num_sensor, feature_dim), ms.float32)
    input = ops.Concat(1)((padding_zeros, input))
    
    fold_op = nn.Unfold(ksizes=[1, time_window_size, 1, 1], strides=[1, stride, 1, 1], rates=[1, 1, 1, 1], padding="VALID")
    output_unfolded = fold_op(input.transpose(0, 3, 2, 1))
    
    output = output_unfolded.reshape(bs, feature_dim, num_sensor, -1, time_window_size)
    output = output.transpose(0, 3, 2, 4, 1)
    
    return output  # (bs, num_windows, num_sensors, time_window_size, feature_dim)
def Mask_Matrix(num_node, time_length, decay_rate):
    mask = np.zeros((time_length, time_length))
    for i in range(time_length):
        for j in range(time_length):
            mask[i, j] = np.exp(-1 * decay_rate * abs(i - j))
    mask = np.expand_dims(mask, axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    mask = np.expand_dims(mask, axis=0)
    mask = np.tile(mask, (1, 1, 1, num_node, num_node))
    mask = np.swapaxes(mask, 1, 3)
    mask = np.swapaxes(mask, 2, 4)
    # print('mask size is ',mask.shape)

    spatial_mask = np.ones((num_node, num_node))
    spatial_mask = np.expand_dims(spatial_mask, axis=-1)
    spatial_mask = np.expand_dims(spatial_mask, axis=-1)
    spatial_mask = np.expand_dims(spatial_mask, axis=0)
    spatial_mask = np.tile(spatial_mask, (1, 1, 1, time_length, time_length))
    # print('spatial_mask size is ', spatial_mask.shape)
    mask = mask * spatial_mask
    mask = ms.Tensor(mask, ms.float32)
    return mask




class GraphConvpoolMPNN_block_v6(nn.Cell):
    def __init__(self, input_dim, output_dim, num_sensors, time_length, time_window_size, stride, decay, pool_choice):
        super(GraphConvpoolMPNN_block_v6, self).__init__()
        self.time_window_size = time_window_size
        self.stride = stride
        self.output_dim = output_dim

        self.graph_construction = Dot_Graph_Construction_weights(input_dim)
        self.BN = nn.BatchNorm1d(input_dim)

        self.MPNN = MPNN_mk_v2(input_dim, output_dim, k=1)

        self.pre_relation = Mask_Matrix(num_sensors,time_window_size,decay)

        self.pool_choice = pool_choice
    def construct(self, input):
        ## input size (bs, time_length, num_nodes, input_dim)
        ## output size (bs, output_node_t, output_node_s, output_dim)

        input_con = Conv_GraphST(input, self.time_window_size, self.stride)
        ## input_con size (bs, num_windows, num_sensors, time_window_size, feature_dim)
        bs, num_windows, num_sensors, time_window_size, feature_dim = input_con.shape
        input_con_ = ops.Transpose()(input_con, (0, 2, 3))
        input_con_ = ops.Reshape()(input_con_, (bs*num_windows, time_window_size*num_sensors, feature_dim))

        A_input = self.graph_construction(input_con_)
        # print(A_input.size())
        # print(self.pre_relation.size())
        A_input = A_input*self.pre_relation


        input_con_ = ops.Transpose()(input_con_, (0, 2, 1))
        input_con_ = self.BN(input_con_)
        input_con_ = ops.Transpose()(input_con_, (0, 2, 1))
        X_output = self.MPNN(input_con_, A_input)


        X_output = ops.Reshape()(X_output, (bs, num_windows, time_window_size,num_sensors, self.output_dim))
        # print(X_output.size())

        if self.pool_choice == 0:
            X_output = ops.ReduceMean()(X_output, 2)
        elif self.pool_choice == 1:

            X_output, ind = ops.ReduceMax()(X_output, 2)
        else:
            print('input choice for pooling cannot be read')
        # X_output = tr.reshape(X_output, [bs, num_windows*time_window_size,num_sensors, self.output_dim])
        # print(X_output.size())

        return X_output


class MPNN_block_seperate(nn.Cell):
    def __init__(self, input_dim, output_dim, num_sensors, time_conv, time_window_size, stride, decay, pool_choice):
        super(MPNN_block_seperate, self).__init__()
        self.time_window_size = time_window_size
        self.stride = stride
        self.output_dim = output_dim

        self.graph_construction = Dot_Graph_Construction_weights(input_dim*2)
        self.BN = nn.BatchNorm1d(input_dim)

        self.Temporal = Feature_extractor_1DCNN_RUL(input_dim, input_dim*2, input_dim*2,kernel_size=3)
        self.time_conv = time_conv
        self.Spatial = MPNN_mk_v2(2*input_dim, output_dim, k=1)

        self.pre_relation = Mask_Matrix(num_sensors,time_window_size,decay)

        self.pool_choice = pool_choice
    def construct(self, input):
        ## input size (bs, time_length, num_nodes, input_dim)
        bs, time_length, num_nodes, input_dim = input.shape

        # input_con = Conv_GraphST(input, self.time_window_size, self.stride)
        # ## input_con size (bs, num_windows, num_sensors, time_window_size, feature_dim)
        # bs, num_windows, num_sensors, time_window_size, feature_dim = input_con.shape
        # input_con_ = tr.transpose(input_con, (0, 2, 3))
        # input_con_ = tr.reshape(input_con_, (bs*num_windows, time_window_size*num_sensors, feature_dim))

        tem_input = ops.Transpose()(input, (0, 2, 1))
        tem_input = ops.Reshape()(tem_input, (bs*num_nodes, time_length, input_dim))

        tem_output = self.Temporal(tem_input)
        # print(tem_output.size())
        tem_output = ops.Reshape()(tem_output, (bs, num_nodes, self.time_conv, 2*input_dim))
        spa_input = ops.Transpose()(tem_output, (0, 2, 1))

        spa_input = ops.Reshape()(spa_input, (bs*self.time_conv, num_nodes, 2*input_dim))
        A_input = self.graph_construction(spa_input)

        spa_output = self.Spatial(spa_input, A_input)



        return spa_output


class GraphMPNNConv_block(nn.Cell):
    def __init__(self, input_dim, output_dim, num_sensors, time_window_size, stride, decay):
        super(GraphMPNNConv_block, self).__init__()
        self.time_window_size = time_window_size
        self.stride = stride
        self.output_dim = output_dim

        self.graph_construction = Dot_Graph_Construction_weights(input_dim)
        self.MPNN = MPNN_mk(input_dim, output_dim, k=1)
        self.pre_relation = Mask_Matrix(num_sensors, time_window_size, decay)


    def construct(self, input):
        ## input size (bs, time_length, num_nodes, input_dim)
        ## output size (bs, output_node_t, output_node_s, output_dim)

        input_con = Conv_GraphST(input, self.time_window_size, self.stride)
        ## input_con size (bs, num_windows, num_sensors, time_window_size, feature_dim)
        bs, num_windows, num_sensors, time_window_size, feature_dim = input_con.shape
        input_con_ = ops.Transpose()(input_con, (0, 2, 3))
        input_con_ = ops.Reshape()(input_con_, (bs * num_windows, time_window_size * num_sensors, feature_dim))

        A_input = self.graph_construction(input_con_)

        A_input = A_input * self.pre_relation

        X_output = self.MPNN(input_con_, A_input)

        X_output = ops.Reshape()(X_output, (bs, num_windows, time_window_size, num_sensors, self.output_dim))

        X_output = ops.Reshape()(X_output, (bs, num_windows*time_window_size, num_sensors, self.output_dim))

        return X_output


class GraphMPNN_block(nn.Cell):
    def __init__(self, input_dim, output_dim, num_sensors, time_length, decay):
        super(GraphMPNN_block, self).__init__()

        self.graph_construction = Dot_Graph_Construction_weights(input_dim)
        self.MPNN = MPNN_mk(input_dim, output_dim, k=1)
        self.pre_relation = Mask_Matrix(num_sensors,time_length,decay)

    def construct(self, input):
        bs, tlen, num_sensors, feature_dim = input.shape
        input_con_ = ops.Reshape()(input, (bs, tlen*num_sensors, feature_dim))

        A_input = self.graph_construction(input_con_)

        A_input = A_input*self.pre_relation

        X_output = self.MPNN(input_con_, A_input)

        X_output = ops.Reshape()(X_output, (bs, tlen, num_sensors, -1))

        return X_output

