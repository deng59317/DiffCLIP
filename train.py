import os
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import load_args
from model import mae_vit_underwater_ovinstseg_patch16


# =========================================================
# ===================== Basic Utils =======================
# =========================================================

def move_to_device(batch, device):
    img_A = batch["img_A"].to(device, non_blocking=True)
    img_B = batch["img_B"].to(device, non_blocking=True)

    targets = []
    for t in batch["targets"]:
        new_t = {}
        for k, v in t.items():
            if torch.is_tensor(v):
                new_t[k] = v.to(device, non_blocking=True)
            else:
                new_t[k] = v
        targets.append(new_t)

    return img_A, img_B, targets


def underwater_collate_fn(batch):
    """
    每个样本格式:
    {
        "img_A": Tensor[3,H,W],
        "img_B": Tensor[1,H,W],
        "masks": Tensor[N,H,W],
        "labels": Tensor[N]
    }
    """
    img_A = torch.stack([b["img_A"] for b in batch], dim=0)
    img_B = torch.stack([b["img_B"] for b in batch], dim=0)

    targets = []
    for b in batch:
        targets.append({
            "masks": b["masks"],   # [N,H,W]
            "labels": b["labels"], # [N]
        })

    return {
        "img_A": img_A,
        "img_B": img_B,
        "targets": targets
    }


def build_text_features_from_labels(num_classes, text_dim, device):
    """
    这是一个占位函数。
    正式版应改成：
        1. 用类别名生成 underwater prompts
        2. 用 CLIP text encoder 编码
        3. 聚合成 [num_classes, text_dim]
    当前先随机初始化占位，确保 train.py 结构可通。
    """
    text_features = torch.randn(num_classes, text_dim, device=device)
    text_features = torch.nn.functional.normalize(text_features, dim=-1)
    return text_features


# =========================================================
# ==================== Matching Utils =====================
# =========================================================

def simple_match_queries_to_targets(pred_masks, pred_logits, targets):
    """
    一个很简化的占位匹配器，不是最终论文版 Hungarian matching。
    用来先把训练流程串起来。

    pred_masks:  [B, Q, H, W]
    pred_logits: [B, Q, C] or None
    targets: list of dict

    返回:
        matched_masks:  [B, Q, H, W]
        matched_labels: [B, Q]
        matched_obj:    [B, Q]
    """
    device = pred_masks.device
    B, Q, H, W = pred_masks.shape

    matched_masks = torch.zeros((B, Q, H, W), device=device)
    matched_labels = torch.zeros((B, Q), dtype=torch.long, device=device)
    matched_obj = torch.zeros((B, Q), dtype=torch.float32, device=device)

    for b in range(B):
        gt_masks = targets[b]["masks"]   # [N,H,W]
        gt_labels = targets[b]["labels"] # [N]

        num_gt = gt_masks.shape[0]
        num_use = min(Q, num_gt)

        if num_use > 0:
            matched_masks[b, :num_use] = gt_masks[:num_use]
            matched_labels[b, :num_use] = gt_labels[:num_use]
            matched_obj[b, :num_use] = 1.0

    return matched_masks, matched_labels, matched_obj


# =========================================================
# ===================== Train Loops =======================
# =========================================================

def pretrain_one_epoch(model, loader, optimizer, device, epoch, mask_ratio=0.75):
    model.train()
    total_loss = 0.0

    for idx, batch in enumerate(loader):
        img_A, img_B, _ = move_to_device(batch, device)

        out = model(
            img_A,
            img_B,
            mode="pretrain",
            mask_ratio=mask_ratio
        )

        loss = out["loss_rec"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu())

    total_loss /= (idx + 1)
    print(f"[Pretrain] Epoch {epoch:03d} | loss_rec: {total_loss:.6f}")
    return total_loss


def seg_train_one_epoch(model, loader, optimizer, device, epoch, num_classes, text_dim=512):
    model.train()
    total_loss = 0.0
    total_loss_mask = 0.0
    total_loss_cls = 0.0
    total_loss_obj = 0.0

    # 正式版你应该提前缓存 text_features，不要每个epoch随机生成
    text_features = build_text_features_from_labels(num_classes, text_dim, device)

    for idx, batch in enumerate(loader):
        img_A, img_B, targets = move_to_device(batch, device)

        out = model(
            img_A,
            img_B,
            text_features=text_features,
            mode="seg"
        )

        pred_masks = out["pred_masks"]                 # [B,Q,H,W]
        pred_logits = out["pred_logits"]               # [B,Q,C]
        pred_objectness = out["pred_objectness"].squeeze(-1)  # [B,Q]

        matched_masks, matched_labels, matched_obj = simple_match_queries_to_targets(
            pred_masks, pred_logits, targets
        )

        # mask loss
        loss_mask_bce = model.sigmoid_ce_loss(pred_masks, matched_masks)
        loss_mask_dice = model.dice_loss(pred_masks, matched_masks)
        loss_mask = loss_mask_bce + loss_mask_dice

        # objectness loss
        loss_obj = nn.functional.binary_cross_entropy_with_logits(
            pred_objectness, matched_obj
        )

        # classification loss
        B, Q, C = pred_logits.shape
        loss_cls = nn.functional.cross_entropy(
            pred_logits.reshape(B * Q, C),
            matched_labels.reshape(B * Q)
        )

        loss = loss_mask + 0.5 * loss_cls + 0.5 * loss_obj

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_loss_mask += float(loss_mask.detach().cpu())
        total_loss_cls += float(loss_cls.detach().cpu())
        total_loss_obj += float(loss_obj.detach().cpu())

    total_loss /= (idx + 1)
    total_loss_mask /= (idx + 1)
    total_loss_cls /= (idx + 1)
    total_loss_obj /= (idx + 1)

    print(
        f"[SegTrain] Epoch {epoch:03d} | "
        f"loss: {total_loss:.6f} | "
        f"mask: {total_loss_mask:.6f} | "
        f"cls: {total_loss_cls:.6f} | "
        f"obj: {total_loss_obj:.6f}"
    )
    return total_loss


@torch.no_grad()
def seg_validate_one_epoch(model, loader, device, epoch, num_classes, text_dim=512):
    model.eval()
    total_loss = 0.0

    text_features = build_text_features_from_labels(num_classes, text_dim, device)

    for idx, batch in enumerate(loader):
        img_A, img_B, targets = move_to_device(batch, device)

        out = model(
            img_A,
            img_B,
            text_features=text_features,
            mode="seg"
        )

        pred_masks = out["pred_masks"]
        pred_logits = out["pred_logits"]
        pred_objectness = out["pred_objectness"].squeeze(-1)

        matched_masks, matched_labels, matched_obj = simple_match_queries_to_targets(
            pred_masks, pred_logits, targets
        )

        loss_mask_bce = model.sigmoid_ce_loss(pred_masks, matched_masks)
        loss_mask_dice = model.dice_loss(pred_masks, matched_masks)
        loss_mask = loss_mask_bce + loss_mask_dice

        B, Q, C = pred_logits.shape
        loss_cls = nn.functional.cross_entropy(
            pred_logits.reshape(B * Q, C),
            matched_labels.reshape(B * Q)
        )

        loss_obj = nn.functional.binary_cross_entropy_with_logits(
            pred_objectness, matched_obj
        )

        loss = loss_mask + 0.5 * loss_cls + 0.5 * loss_obj
        total_loss += float(loss.detach().cpu())

    total_loss /= (idx + 1)
    print(f"[Val] Epoch {epoch:03d} | loss: {total_loss:.6f}")
    return total_loss


# =========================================================
# ======================= Main ============================
# =========================================================

def main():
    args = load_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epoch = args.epochs
    lr = args.lr
    batch_size = args.batch_size
    windowsize = args.windowsize

    # 你这里要自己在 config.py 里补
    # args.nb_classes
    # args.text_dim
    # args.num_queries
    # args.mask_dim
    # args.pretrain_epochs
    # args.save_path
    num_classes = args.nb_classes
    text_dim = getattr(args, "text_dim", 512)
    num_queries = getattr(args, "num_queries", 50)
    mask_dim = getattr(args, "mask_dim", 256)
    pretrain_epochs = getattr(args, "pretrain_epochs", 0)
    save_path = getattr(args, "save_path", "./net_seg.pt")

    net_name = "DiffCLIP_Underwater_OVIS"
    day = datetime.datetime.now().strftime("%m_%d_%H_%M")
    print(f"=> creating model '{net_name}' at {day}")

    model = mae_vit_underwater_ovinstseg_patch16(
        img_size=windowsize,
        in_chans=3,
        in_chans_B=1,
        proj_dim=512,
        text_dim=text_dim,
        num_queries=num_queries,
        mask_dim=mask_dim,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2
    )

    # =====================================================
    # 这里你必须换成你自己的新数据集
    # 不能再直接用老的 HyperData(x, x_LIDAR, y)
    # =====================================================
    #
    # 例如:
    # from hyper_dataset import MARISSegDataset
    #
    # train_dataset = MARISSegDataset(...)
    # val_dataset = MARISSegDataset(...)
    #
    # 这里只给结构，不强行写死你的数据路径
    #
    # =====================================================

    from hyper_dataset import HyperData  # 占位，后面你要替换成新分割dataset

    train_dataset = HyperData(split="train")  # 这里必须改成你自己的分割dataset接口
    val_dataset = HyperData(split="val")      # 这里必须改

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=underwater_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=underwater_collate_fn
    )

    best_val = 1e9
    tic = time.time()

    # ---------------- Pretrain ----------------
    for epoch in range(pretrain_epochs):
        pretrain_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            mask_ratio=getattr(args, "mask_ratio", 0.75),
        )
        scheduler.step()

    # ---------------- Seg Train ----------------
    for epoch in range(pretrain_epochs, num_epoch):
        seg_train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            num_classes=num_classes,
            text_dim=text_dim,
        )

        val_loss = seg_validate_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            epoch=epoch,
            num_classes=num_classes,
            text_dim=text_dim,
        )

        scheduler.step()

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
        }

        torch.save(ckpt, save_path)

        if val_loss < best_val:
            best_val = val_loss
            best_path = save_path.replace(".pt", "_best.pt")
            torch.save(ckpt, best_path)
            print(f"Best model saved to {best_path}")

    toc = time.time()
    print("Training time(s):", toc - tic)
    print("-------- Training Finished -----------")


if __name__ == "__main__":
    main()
