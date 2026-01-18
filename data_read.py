import os
import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA
from build_EMP import build_emp


def _load_mat_key(file_path, keys):
    data = sio.loadmat(file_path)
    for k in keys:
        if k in data:
            return data[k]
    for k, v in data.items():
        if not k.startswith("__"):
            return v
    raise KeyError(f"No valid keys found in {file_path}. keys={list(data.keys())}")


def _merge_train_test(train_data, test_data):
    # allow different N but require sample shape match
    if train_data.ndim != test_data.ndim:
        raise ValueError(f"ndim mismatch: {train_data.ndim} vs {test_data.ndim}")
    if train_data.shape[1:] != test_data.shape[1:]:
        raise ValueError(f"sample shape mismatch: {train_data.shape} vs {test_data.shape}")
    return np.concatenate([train_data, test_data], axis=0)


def _merge_labels(y_tr, y_te):
    y_tr = np.asarray(y_tr).reshape(-1)
    y_te = np.asarray(y_te).reshape(-1)
    return np.concatenate([y_tr, y_te], axis=0)


def _normalize01(x):
    x = x.astype(np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def pca_whitening_patches(patches, number_of_pc):
    # patches: (N, ws, ws, C) -> (N, ws, ws, number_of_pc)
    n, h, w, c = patches.shape
    x = patches.reshape(-1, c)
    pca = PCA(n_components=number_of_pc)
    z = pca.fit_transform(x)
    return z.reshape(n, h, w, number_of_pc).astype(np.float32)


def load_data(dataset):
    """
    Trento patch-only loader

    Returns:
      patch_HSI   : (N, ws, ws, B) float32
      patch_LiDAR : (N, ws, ws, 1) float32
      y           : (N,) int64 (0-based class index)
    """
    if dataset != "Trento":
        raise ValueError("This data_read.py supports only Trento patch dataset.")

    base = "/root/autodl-tmp/Trento遥感图像数据集【HSI+LiDAR】】"

    HSI_tr = _load_mat_key(os.path.join(base, "HSI_Tr.mat"), ["HSI", "HSI_Tr", "hsi"])
    HSI_te = _load_mat_key(os.path.join(base, "HSI_Te.mat"), ["HSI", "HSI_Te", "hsi"])
    LiDAR_tr = _load_mat_key(os.path.join(base, "LIDAR_Tr.mat"), ["LiDAR", "LIDAR_Tr", "lidar"])
    LiDAR_te = _load_mat_key(os.path.join(base, "LIDAR_Te.mat"), ["LiDAR", "LIDAR_Te", "lidar"])

    y_tr = _load_mat_key(os.path.join(base, "TrLabel.mat"), ["TRLabel", "TrLabel", "trainlabels"])
    y_te = _load_mat_key(os.path.join(base, "TeLabel.mat"), ["TSLabel", "TeLabel", "testlabels"])

    patch_HSI = _merge_train_test(HSI_tr, HSI_te).astype(np.float32)
    patch_LiDAR = _merge_train_test(LiDAR_tr, LiDAR_te).astype(np.float32)

    if patch_LiDAR.ndim == 3:
        patch_LiDAR = np.expand_dims(patch_LiDAR, -1)

    y = _merge_labels(y_tr, y_te).astype(np.int64)
    if y.min() == 1:
        y = y - 1  # to 0-based

    if patch_HSI.shape[0] != y.shape[0]:
        raise ValueError(f"HSI N != label N: {patch_HSI.shape[0]} vs {y.shape[0]}")
    if patch_LiDAR.shape[0] != y.shape[0]:
        raise ValueError(f"LiDAR N != label N: {patch_LiDAR.shape[0]} vs {y.shape[0]}")

    return patch_HSI, patch_LiDAR, y


def readdata(type, dataset, windowsize, train_num, val_num, num):
    """
    Patch-only readdata.
    Keep SAME 15 return items as original code so train.py unpacking won't break.
    """
    patch_HSI, patch_LiDAR, y = load_data(dataset)

    if type == "PCA":
        patch_HSI = pca_whitening_patches(patch_HSI, number_of_pc=30)
    elif type == "EMP":
        patch_HSI = pca_whitening_patches(patch_HSI, number_of_pc=4)
    elif type == "none":
        pass
    else:
        raise ValueError("type does not find")

    patch_HSI = _normalize01(patch_HSI)
    patch_LiDAR = _normalize01(patch_LiDAR)

    num_classes = int(y.max()) + 1
    total_num = int(y.shape[0])

    # shuffle split
    rng = np.random.RandomState(num)
    idx = np.arange(total_num)
    rng.shuffle(idx)

    # if user passes very large val_num, cap it
    val_num = int(min(val_num, max(1, total_num // 5)))
    val_idx = idx[:val_num]
    train_idx = idx[val_num:]

    train_image = patch_HSI[train_idx]
    train_image_LIDAR = patch_LiDAR[train_idx]
    train_label = y[train_idx]

    validation_image = patch_HSI[val_idx]
    validation_image_LIDAR = patch_LiDAR[val_idx]
    validation_label = y[val_idx]

    # placeholders to match original signature
    nTrain_perClass = np.zeros(num_classes, dtype=np.int64)
    nvalid_perClass = np.zeros(num_classes, dtype=np.int64)
    train_index = train_idx.reshape(-1, 1).astype(np.int32)
    val_index = val_idx.reshape(-1, 1).astype(np.int32)
    index = []

    image = None
    image_LiDAR = None
    label = y

    return (
        train_image, train_image_LIDAR, train_label,
        validation_image, validation_image_LIDAR, validation_label,
        nTrain_perClass, nvalid_perClass,
        train_index, val_index, index,
        image, image_LiDAR, label, total_num
    )
