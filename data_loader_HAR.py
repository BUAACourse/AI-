import mindspore as ms
import mindspore.dataset as ds
import os
import numpy as np


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


def create_dataset(data, args, is_training=True):
    dataset_generator = Load_Dataset(data, args)

    dataset = ds.GeneratorDataset(
        dataset_generator,
        column_names=["data", "label"],
        shuffle=is_training
    )

    dataset = dataset.batch(args.batch_size, drop_remainder=args.drop_last)

    return dataset

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


def data_generator(data_path, args):
    train_dataset = ms.load_checkpoint(os.path.join(data_path, "train.ckpt"))
    valid_dataset = ms.load_checkpoint(os.path.join(data_path, "val.ckpt"))
    test_dataset = ms.load_checkpoint(os.path.join(data_path, "test.ckpt"))

    return load_data(data_path, args)