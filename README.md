# 1. 论文介绍

## 1.1 背景介绍

多变量时间序列（MTS）数据在众多实际应用领域中具有重要价值，如预测性维护、医疗健康等。由于其固有的时序性和多源特性（即来自多个传感器的数据），MTS数据展现出显著的空间-时间（ST）依赖性。这种依赖性不仅包括不同时间戳之间的时序相关性，还包括同一时间戳内不同传感器之间的空间相关性。为了有效利用这些信息，图神经网络（GNNs）方法被广泛应用。然而，现有方法通常分别捕捉空间依赖性和时间依赖性，而忽略了不同传感器在不同时间戳之间的相关性（DEDT）。这种忽略限制了现有GNNs在全面建模ST依赖性方面的能力，从而影响了它们在MTS数据上学习有效表示的性能。

传统方法主要关注捕捉时序相关性，通常采用时间编码器（如CNNs、LSTM和Transformer）来处理MTS数据。这些方法虽然在一定程度上取得了成功，但由于忽略了空间依赖性，其性能受到限制。近年来，GNNs因其能够有效捕捉空间依赖性而成为研究热点。GNNs通常与时间编码器结合使用，分别捕捉空间依赖性和时间依赖性。然而，现有方法在图构建和图卷积过程中存在局限性，无法显式考虑DEDT之间的相关性，从而限制了它们在全面建模ST依赖性方面的能力。


## 1.2 论文方法

《Fully-Connected Spatial-Temporal Graph for Multivariate Time Series Data》

这篇论文提出了提出了一种新颖的方法，称为全连接空间-时间图神经网络（FC-STGNN）。该方法通过全连接图构建和移动池化GNN层，显式建模传感器之间的DEDT相关性，从而全面捕捉MTS数据中的ST依赖性。FC-STGNN的设计旨在提高MTS数据的表示学习能力，进而提升其在各种下游任务中的性能。

本文提出的方法（FC-STGNN）具有以下优势：

- 通过显式建模传感器之间的相关性，并引入衰减矩阵，充分考虑了时间距离对传感器依赖关系的影响，使模型能够更好地捕捉全局的时间依赖特性。
- 通过移动窗口的图卷积操作，模型能够捕捉局部的时空依赖性，从而有效提取局部的动态特征。
- 通过时间池化和多层并行的特征拼接，模型能够聚合多尺度的时空信息，增强了对复杂传感器数据的建模能力。

## 1.3 数据集介绍

本文使用的数据集是UCI-HAR数据集与ISRUC-S3 数据集。

UCI Human Activity Recognition (HAR) 数据集是一个广泛使用的多维时间序列（MTS）数据集，旨在支持可穿戴设备的人体活动识别任务，该数据集由 30 名志愿者在佩戴智能手机（Samsung Galaxy S II）进行日常活动时采集，包括 6 种活动（步行、上下楼梯、坐、站和躺下）。每条数据包含 3 轴加速度、3 轴陀螺仪等多维传感器信号，采样频率为 50Hz。ISRUC-S3（ISRUC Sleep Dataset, Subject Group 3）是一个多通道生理信号数据集，用于睡眠分期和睡眠分析任务。数据集由葡萄牙科英布拉大学 ISRUC 团队采集，包含 10 名健康受试者的多夜多通道 PSG（多导睡眠图）信号，包括脑电（EEG）、眼动（EOG）、肌电（EMG）、心电（ECG）等信号，采样频率为 200 Hz。

## 1.4 pipeline

本作业将基于论文[官方代码仓库](https://github.com/Frank-Wang-oss/FCSTGNN)实现，将pytorch版本的网络模型转换成mindspore版本的模型。





# 2. pytorch实现版本

## 2.1 准备工作

创建环境：

```
conda create -n mamo python=3.7
```

安装依赖包：

```
# Name                    Version                   Build  Channel                                                         
numpy                     1.21.6                   pypi_0    pypi
pandas                    1.3.5                    pypi_0    pypi
pillow                    9.5.0                    pypi_0    pypi
pip                       23.1.1             pyhd8ed1ab_0    http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
python                    3.7.12          h900ac77_100_cpython    http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
torch                     1.12.1+cu113             pypi_0    pypi
torchaudio                0.12.1+cu113             pypi_0    pypi
torchvision               0.13.1+cu113             pypi_0    pypi
tqdm                      4.65.0                   pypi_0    pypi
urllib3                   1.26.15                  pypi_0    pypi
```

数据集下载：

UCI-HAR数据集：数据下载地址：https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones

ISRUC-S3数据集：数据下载地址：https://sleeptight.isr.uc.pt/





该项目的文件目录树如下：


```
code
├─ 📁Data_preprocessing
│  ├─ 📁HAR
│  │  ├─ 📁test
│  │  └─ 📁train
│  ├─ 📁ISRUC_S3
│  │  ├─ 📁ExtractedChannels
│  │  └─ 📁RawData
│  ├─ 📄bash.sh
│  ├─ 📄preprocess_ISRUC.py
│  └─ 📄preprocess_UCI_HAR.py
├─ 📁HAR
│  ├─ 📄test.ckpt
│  ├─ 📄train.ckpt
│  └─ 📄val.ckpt
├─ 📁ISRUC
│  ├─ 📄ISRUC_S1.npz
│  ├─ 📄ISRUC_S10.npz
│  ├─ 📄ISRUC_S2.npz
│  ├─ 📄ISRUC_S3.npz
│  ├─ 📄ISRUC_S4.npz
│  ├─ 📄ISRUC_S5.npz
│  ├─ 📄ISRUC_S6.npz
│  ├─ 📄ISRUC_S7.npz
│  ├─ 📄ISRUC_S8.npz
│  └─ 📄ISRUC_S9.npz
├─ 📄args.py
├─ 📄data_loader_HAR.py
├─ 📄data_loader_ISRUC.py
├─ 📄main_HAR.py
├─ 📄main_ISRUC.py
├─ 📄Model.py
├─ 📄Model_Base.py
└─ 📄README.md
```


## 2.2 运行代码
下载与预处理步骤：
1. 下载S3数据集并放入`ISRUC`目录，下载HAC数据集放入`HAR`目录
2. 首先运行`preprocess_ISRUC.py`与`preprocess_UCI_HAR.py`进行数据预处理


执行脚本`python main_ISRUC.py`和`python main_HAR.py`输出结果如下：

```
start train
train finished
...
In the 0th epoch, accuracy is 70.60%
In the 1th epoch, accuracy is 71.55%
...
...
In the 37th epoch, accuracy is 93.96%
In the 38th epoch, accuracy is 94.10%
...
...
Final accuracy: 95.20%
Final MF1 score: 94.94%
...
...
Process finished with exit code 0
```





# 3. mindspore实现版本

代码仓库：https://github.com/BUAACourse/AI-/tree/main

## 3.1 mindspore框架介绍

MindSpore是华为推出的一款人工智能计算框架，主要用于开发AI应用和模型。它的特点如下:

- 框架设计：MindSpore采用静态计算图设计，PyTorch采用动态计算图设计。静态计算图在模型编译时确定计算过程，动态计算图在运行时确定计算过程。静态计算图通常更高效，动态计算图更灵活；
- 设备支持：MindSpore在云端和边缘端都有较好的支持，可以在Ascend、CPU、GPU等硬件上运行；
- 自动微分：MindSpore提供自动微分功能，可以自动求导数，简化模型训练过程；
- 运算符和层：MindSpore提供丰富的神经网络层和运算符，覆盖CNN、RNN、GAN等多种模型；
- 训练和部署：MindSpore提供方便的模型训练和部署功能，支持ONNX、CANN和MindSpore格式的模型导出，可以部署到Ascend、GPU、CPU等硬件；



## 3.2 环境准备

使用华为ModelArts，操作系统Ubuntu 22.04。

安装anaconda环境：

```
wget https://mirrors.bfsu.edu.cn/anaconda/archive/Anaconda3-2022.10-Linux-x86_64.sh --no-check-certificate
bash Anaconda3-2022.10-Linux-x86_64.sh
```

创建虚拟环境并且切换到环境：

```
conda create -n mamo python=3.7
conda activate mamo
```


下载依赖包：

```
pip install numpy==1.21.6                   
pip install pandas==1.3.5 
...
...
```





## 3.3 模型迁移

将Pytorch的API替换成mindspore的API，官方给出了[文档说明](https://www.mindspore.cn/docs/zh-CN/r1.7/note/api_mapping/pytorch_api_mapping.html)。

另外mindspore还提供了[MindConverter](https://www.mindspore.cn/mindinsight/docs/zh-CN/r1.7/migrate_3rd_scripts_mindconverter.html)工具，方便从pytorch迁移模型。

下面是我在模型迁移过程替换的API以及Class：

| pytorch API / Class       | mindspore API/Class                | 说明                          | 两者差异                                                     |
| ------------------------- | ---------------------------------- | ----------------------------- | ------------------------------------------------------------ |
| torch.from_numpy          | mindspore.tensor.from_numpy        | 从numpy得到tensor             | 无                                                           |
| torch.tensor.to           | mindspore.tensor.to_device         | 将tensor传入指定的设备        | 无                                                           |
| torch.utils.data.Dataset  | mindspore.dataset.GeneratorDataset | 数据集类                      | PyTorch：自定义数据集的抽象类，自定义数据子类可以通过调用`__len__()`和`__getitem__()`这两个方法继承这个抽象类。<br />MindSpore：通过每次调用Python层自定义的Dataset以生成数据集。 |
| torch.zeros_like          | mindspore.ops.ZerosLike            | 获得指定shape的全零元素tensor | 无                                                           |
| torch.nn.Sigmoid          | mindspore.nn.Sigmoid               | 激活函数                      | 无                                                           |
| torch.nn.Tanh             | mindspore.nn.Tanh                  | 激活函数                      | 无                                                           |
| torch.nn.ReLU             | mindspore.nn.ReLU                  | 激活函数                      | 无                                                           |
| torch.nn.Softmax          | mindspore.nn.Softmax               | 归一化                        | 无                                                           |
| torch.nn.LeakyReLU        | mindspore.nn.LeakyReLU             | 激活函数                      | 无                                                           |
| torch.nn.Sequential       | mindspore.nn.SequentialCell        | 整合多个网络模块              | 无                                                           |
| torch.argmax              | mindspore.ops.argmax               | 返回最大值下标                | PyTorch：沿着给定的维度返回Tensor最大值所在的下标，返回值类型为torch.int64。<br />MindSpore：MindSpore此API实现功能与PyTorch基本一致，返回值类型为int32. |
| torch.abs                 | mindspore.ops.abs                  | 计算tensor绝对值              | PyTorch：计算输入的绝对值。<br />MindSpore：MindSpore此API实现功能与PyTorch一致，仅参数名不同。 |
| torch.mean                | mindspore.ops.ReduceMean           | 计算均值                      | 无                                                           |
| torch.optim.Adam          | mindspore.nn.Adam                  | 优化器                        | 无                                                           |
| torch.nn.CrossEntropyLoss | mindspore.nn.CrossEntropyLoss      | 损失函数                      | PyTorch：计算预测值和目标值之间的交叉熵损失。<br />MindSpore：MindSpore此API实现功能与PyTorch基本一致，而且目标值支持两种不同的数据形式：标量和概率。 |
| torch.nn.Module           | mindspore.nn.Cell                  | 神经网络的基本构成单位        |                                                              |
| torch.nn.Linear           | mindspore.nn.Dense                 | 全连接层                      | PyTorch：全连接层，实现矩阵相乘的运算。<br />MindSpore：MindSpore此API实现功能与PyTorch基本一致，而且可以在全连接层后添加激活函数。 |
| torch.cat                 | mindspore.ops.concat               | tensor按照指定维度拼接        | 无                                                           |
| torch.randn               | mindspore.ops.StandardNormal       | 获得正态分布数据的tensor      | 无                                                           |
| torch.mm                  | mindspore.ops.MatMul               | 矩阵乘法                      | 无                                                           |
| torch.sqrt                | mindspore.ops.Sqrt                 | 开根号                        | 无                                                           |
| torch.sum                 | mindspore.ops.ReduceSum            | 求和                          | 无                                                           |
| torch.Tensor.mul          | mindspore.ops.Mul                  | 相乘                          | 无                                                           |
| torch.div                 | mindspore.ops.div                  | 除法                          | 无                                                           |
| torch.nn.Embedding        | mindspore.nn.Embedding             |                               | PyTorch：支持使用`_weight`属性初始化embedding，并且可以通过`weight`变量获取当前embedding的权重。<br />MindSpore：支持使用`embedding_table`属性初始化embedding，并且可以通过`embedding_table`属性获取当前embedding的权重。除此之外，当`use_one_hot`为True时，你可以得到具有one-hot特征的embedding。 |
| torch.tensor.repeat       | mindspore.ops.tile                 | 对tensor进行重复叠加          | 无                                                           |
| torch.tensor.view         | mindspore.ops.Reshape              | 重新排列tensor的维度          | 无                                                           |
| Adam.zero_grad            | Adam.clear_grad                    | 清除梯度                      | 无                                                           |
|                           |                                    |                               |                                                              |



## 3.4 详细迁移代码

### 数据集实现

```python
import mindspore as ms
import mindspore.dataset as ds


class Load_Dataset:
    def __init__(self, dataset, args):
        samples = dataset["samples"]
        labels = dataset["labels"]

        if len(samples.shape) < 3:
            samples = np.expand_dims(samples, axis=2)

        if samples.shape.index(min(samples.shape)) != 1:  # make sure the Channels in second dim
            samples = np.transpose(samples, (0, 2, 1))

        # Convert to NumPy arrays first
        if isinstance(samples, ms.Tensor):
            samples = samples.asnumpy()
        if isinstance(labels, ms.Tensor):
            labels = labels.asnumpy()

        shape = samples.shape
        self.x_data = samples.reshape((shape[0], shape[1], args.time_denpen_len, args.patch_size))
        self.x_data = np.transpose(self.x_data, (0, 2, 1, 3))
        self.y_data = labels

    def __getitem__(self, index):
        # Convert to MindSpore Tensor only when accessing the data
        return (ms.Tensor(self.x_data[index], dtype=ms.float32), 
                ms.Tensor(self.y_data[index], dtype=ms.int32))

    def __len__(self):
        return self.x_data.shape[0]
```



### 网络实现

```python
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

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
```





## 3.5 训练结果

下面是将pytorch模型转为mindspore模型后的训练测试结果：


```
start train
train finished
...
In the 0th epoch, accuracy is 70.60%
In the 1th epoch, accuracy is 71.55%
...
...
In the 37th epoch, accuracy is 93.96%
In the 38th epoch, accuracy is 94.10%
...
...
Final accuracy: 95.20%
Final MF1 score: 94.94%
...
...
Process finished with exit code 0
```









