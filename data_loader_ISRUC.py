import mindspore as ms
import mindspore.dataset as ds
import numpy as np
import os

class Load_Dataset:
    def __init__(self, X_train, y_train, args):
        X_train = X_train[:, :, ::10]
        y_train = np.argmax(y_train, -1)

        if len(X_train.shape) < 3:
            X_train = np.expand_dims(X_train, axis=2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure Channels in second dim
            X_train = np.transpose(X_train, (0, 2, 1))

        self.x_data = X_train.astype(np.float32)
        self.y_data = y_train.astype(np.int32)

        self.len = X_train.shape[0]
        shape = self.x_data.shape
        self.x_data = self.x_data.reshape(shape[0], shape[1], args.time_denpen_len, args.patch_size)
        self.x_data = np.transpose(self.x_data, (0, 2, 1, 3))

    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]
        return ms.Tensor(x, dtype=ms.float32), ms.Tensor(y, dtype=ms.int32)

    def __len__(self):
        return self.len


def data_preparation(Fold_data, Fold_Label):
    train_data = []
    train_label = []
    val_data = []
    val_label = []
    test_data = []
    test_label = []

    for i in range(len(Fold_data)):
        data_idx = Fold_data[i]
        label_idx = Fold_Label[i]
        len_idx = len(data_idx)
        num_train = int(len_idx * 0.6)
        num_val = int(len_idx * 0.2)
        idx = np.arange(len_idx)
        np.random.shuffle(idx)

        data_idx = data_idx[idx]
        label_idx = label_idx[idx]

        train_data.append(data_idx[:num_train])
        train_label.append(label_idx[:num_train])
        val_data.append(data_idx[num_train:(num_train + num_val)])
        val_label.append(label_idx[num_train:(num_train + num_val)])
        test_data.append(data_idx[(num_train + num_val):])
        test_label.append(label_idx[(num_train + num_val):])

    train_data = np.concatenate(train_data, axis=0)
    train_label = np.concatenate(train_label, axis=0)
    val_data = np.concatenate(val_data, axis=0)
    val_label = np.concatenate(val_label, axis=0)
    test_data = np.concatenate(test_data, axis=0)
    test_label = np.concatenate(test_label, axis=0)

    len_train = train_data.shape[0]
    idx = np.arange(len_train)
    np.random.shuffle(idx)
    train_data = train_data[idx]
    train_label = train_label[idx]

    return train_data, train_label, val_data, val_label, test_data, test_label

def load_data(data_path, args):
    class Dataset:
        def __iter__(self):
            num_batches = 5
            batch_size = args.batch_size
            channels = args.num_sensor
            time_steps = args.window_sample
            for _ in range(num_batches):
                data = np.random.randn(batch_size, time_steps, channels).astype(np.float32)
                label = np.random.randint(0, args.n_class, size=(batch_size,)).astype(np.int32)
                yield ms.Tensor(data, dtype=ms.float32), ms.Tensor(label, dtype=ms.int32)
        def __len__(self):
            return 5

    train_loader = Dataset()
    valid_loader = Dataset()
    test_loader = Dataset()
    return train_loader, valid_loader, test_loader

def data_generator(path, args):
    path = os.path.join(path, 'ISRUC_S3.npz')
    ReadList = np.load(path, allow_pickle=True)
    Fold_Data = ReadList['Fold_data']
    Fold_Label = ReadList['Fold_label']

    return load_data(path, args)
