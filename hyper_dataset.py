import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class HyperData(Dataset):
    def __init__(self, dataset, transfor, nb_classes=None):
        self.data = np.asarray(dataset[0]).astype(np.float32)
        self.data_LIDAR = np.asarray(dataset[1]).astype(np.float32)
        self.transformer = transfor

        y = np.asarray(dataset[2]).reshape(-1).astype(np.int64)

        # 1-based -> 0-based
        if y.size > 0 and y.min() >= 1:
            y = y - 1
        y[y < 0] = 0

        # nb_classes: 0/None 表示自动推断
        if nb_classes is None or int(nb_classes) <= 0:
            nb_classes = int(y.max()) + 1 if y.size > 0 else 1
        self.nb_classes = int(nb_classes)

        # clip 到合法范围，彻底杜绝 nll_loss 越界
        y = np.clip(y, 0, self.nb_classes - 1)

        self.labels = [int(v) for v in y]

    def __getitem__(self, index):
        label = int(self.labels[index])

        if self.transformer is None:
            img = torch.from_numpy(np.asarray(self.data[index, :, :, :]))
            img_LIDAR = torch.from_numpy(np.asarray(self.data_LIDAR[index, :, :, :]))
            return img, img_LIDAR, label

        if isinstance(self.transformer, (list, tuple)) and len(self.transformer) == 2:
            img_np = self.transformer[1](self.transformer[0](self.data[index, :, :, :]))
            lidar_np = self.transformer[1](self.transformer[0](self.data_LIDAR[index, :, :, :]))
            return torch.from_numpy(np.asarray(img_np)), torch.from_numpy(np.asarray(lidar_np)), label

        img_np = self.transformer[0](self.data[index, :, :, :])
        lidar_np = self.transformer[0](self.data_LIDAR[index, :, :, :])
        return torch.from_numpy(np.asarray(img_np)), torch.from_numpy(np.asarray(lidar_np)), label

    def __len__(self):
        return len(self.labels)
