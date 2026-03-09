import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class HyperData(Dataset):
    """
    A unified dataset class that supports:

    1) Old classification mode:
        sample = (img_A, img_B, label)

    2) New segmentation mode:
        sample = {
            "img_A": Tensor[C,H,W],
            "img_B": Tensor[C2,H,W],
            "masks": Tensor[N,H,W],
            "labels": Tensor[N]
        }

    Args
    ----
    dataset:
        For cls mode:
            (data_A, data_B, labels)

        For seg mode:
            Can be:
            - list of dicts, each dict contains:
                {
                    "img_A": ...,
                    "img_B": ...,
                    "masks": ...,
                    "labels": ...
                }
            - or tuple/list:
                (data_A, data_B, masks_list, labels_list)

    transfor:
        same as your original code, can be None / list / tuple

    nb_classes:
        number of classes

    task:
        "cls" or "seg"
    """

    def __init__(self, dataset, transfor=None, nb_classes=None, task="cls"):
        super().__init__()
        self.transformer = transfor
        self.task = task

        if self.task == "cls":
            self._init_cls(dataset, nb_classes)
        elif self.task == "seg":
            self._init_seg(dataset, nb_classes)
        else:
            raise ValueError(f"Unsupported task: {task}")

    # =====================================================
    # ================= Classification Init ===============
    # =====================================================

    def _init_cls(self, dataset, nb_classes=None):
        self.data = np.asarray(dataset[0]).astype(np.float32)
        self.data_B = np.asarray(dataset[1]).astype(np.float32)

        y = np.asarray(dataset[2]).reshape(-1).astype(np.int64)

        # 1-based -> 0-based
        if y.size > 0 and y.min() >= 1:
            y = y - 1
        y[y < 0] = 0

        if nb_classes is None or int(nb_classes) <= 0:
            nb_classes = int(y.max()) + 1 if y.size > 0 else 1
        self.nb_classes = int(nb_classes)

        y = np.clip(y, 0, self.nb_classes - 1)
        self.labels = [int(v) for v in y]

    # =====================================================
    # ================= Segmentation Init =================
    # =====================================================

    def _init_seg(self, dataset, nb_classes=None):
        self.samples = []

        # -------- Case 1: list of dict --------
        if isinstance(dataset, (list, tuple)) and len(dataset) > 0 and isinstance(dataset[0], dict):
            for sample in dataset:
                img_A = sample["img_A"]
                img_B = sample["img_B"]
                masks = sample["masks"]
                labels = sample["labels"]

                labels = np.asarray(labels).reshape(-1).astype(np.int64)
                if labels.size > 0 and labels.min() >= 1:
                    labels = labels - 1
                labels[labels < 0] = 0

                self.samples.append({
                    "img_A": np.asarray(img_A).astype(np.float32),
                    "img_B": np.asarray(img_B).astype(np.float32),
                    "masks": np.asarray(masks).astype(np.float32),
                    "labels": labels
                })

        # -------- Case 2: tuple/list of arrays --------
        elif isinstance(dataset, (list, tuple)) and len(dataset) == 4:
            data_A, data_B, masks_list, labels_list = dataset

            for img_A, img_B, masks, labels in zip(data_A, data_B, masks_list, labels_list):
                labels = np.asarray(labels).reshape(-1).astype(np.int64)
                if labels.size > 0 and labels.min() >= 1:
                    labels = labels - 1
                labels[labels < 0] = 0

                self.samples.append({
                    "img_A": np.asarray(img_A).astype(np.float32),
                    "img_B": np.asarray(img_B).astype(np.float32),
                    "masks": np.asarray(masks).astype(np.float32),
                    "labels": labels
                })

        else:
            raise ValueError(
                "For task='seg', dataset must be either:\n"
                "1) list of dict samples\n"
                "2) tuple/list: (data_A, data_B, masks_list, labels_list)"
            )

        # infer nb_classes
        all_labels = []
        for s in self.samples:
            if len(s["labels"]) > 0:
                all_labels.append(s["labels"])

        if len(all_labels) > 0:
            all_labels = np.concatenate(all_labels, axis=0)
        else:
            all_labels = np.array([0], dtype=np.int64)

        if nb_classes is None or int(nb_classes) <= 0:
            nb_classes = int(all_labels.max()) + 1 if all_labels.size > 0 else 1
        self.nb_classes = int(nb_classes)

        # clip labels
        for s in self.samples:
            s["labels"] = np.clip(s["labels"], 0, self.nb_classes - 1)

    # =====================================================
    # ==================== Transform ======================
    # =====================================================

    def _apply_transform_single(self, arr):
        """
        arr: numpy array
        """
        if self.transformer is None:
            return arr

        if isinstance(self.transformer, (list, tuple)) and len(self.transformer) == 2:
            arr = self.transformer[1](self.transformer[0](arr))
            return arr

        if isinstance(self.transformer, (list, tuple)) and len(self.transformer) == 1:
            arr = self.transformer[0](arr)
            return arr

        return arr

    # =====================================================
    # ==================== Get Item =======================
    # =====================================================

    def __getitem__(self, index):
        if self.task == "cls":
            label = int(self.labels[index])

            img = self._apply_transform_single(self.data[index])
            img_B = self._apply_transform_single(self.data_B[index])

            img = torch.from_numpy(np.asarray(img)).float()
            img_B = torch.from_numpy(np.asarray(img_B)).float()

            return img, img_B, label

        elif self.task == "seg":
            sample = self.samples[index]

            img_A = self._apply_transform_single(sample["img_A"])
            img_B = self._apply_transform_single(sample["img_B"])

            # masks 一般不要做和 RGB 一样的增强，除非你保证几何一致
            masks = np.asarray(sample["masks"]).astype(np.float32)
            labels = np.asarray(sample["labels"]).astype(np.int64)

            img_A = torch.from_numpy(np.asarray(img_A)).float()
            img_B = torch.from_numpy(np.asarray(img_B)).float()
            masks = torch.from_numpy(masks).float()
            labels = torch.from_numpy(labels).long()

            # 保证 masks 维度是 [N,H,W]
            if masks.ndim == 2:
                masks = masks.unsqueeze(0)

            return {
                "img_A": img_A,
                "img_B": img_B,
                "masks": masks,
                "labels": labels
            }

        else:
            raise ValueError(f"Unsupported task: {self.task}")

    # =====================================================
    # ===================== Length ========================
    # =====================================================

    def __len__(self):
        if self.task == "cls":
            return len(self.labels)
        elif self.task == "seg":
            return len(self.samples)
        else:
            raise ValueError(f"Unsupported task: {self.task}")
