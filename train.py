import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import datetime
import numpy as np

from config import load_args
from data_read import readdata
from diffusion import create_diffusion
from hyper_dataset import HyperData
from augment import CenterResizeCrop
from model import mae_vit_HSIandLIDAR_patch3, DDPM_LOSS


args = load_args()

mask_ratio = args.mask_ratio
windowsize = args.windowsize
dataset = args.dataset
type = args.type
num_epoch = args.epochs
lr = args.lr
train_num_per = args.train_num_perclass
batch_size = args.batch_size

net_name = 'LDS2AE'
day = datetime.datetime.now()
day_str = day.strftime('%m_%d_%H_%M')

val_num = 1000
_, _, _, _, _, _, _, _, _, _, _, _, _, gt0, s0 = readdata(type, dataset, windowsize, train_num_per, val_num, 0)
nclass_auto = int(np.max(gt0)) + 1  # 因为 data_read 已经 0-based
nclass = args.nb_classes if args.nb_classes > 0 else nclass_auto
print("nclass =", nclass)

if args.dataset == 'Muufl':
    in_chans_LIDAR = 2
else:
    in_chans_LIDAR = 1

size = 1
criterion = nn.CrossEntropyLoss()

print("=> creating model '{}'".format(net_name))

diffusion = create_diffusion(timestep_respacing="1000")

for num in range(0, 1):  # 先跑一次实验，避免你10次循环太慢
    print('num:', num)

    train_image, train_image_LIDAR, train_label, val_image, val_image_LIDAR, val_label, *_ = readdata(
        type, dataset, windowsize, train_num_per, int(s0 * 0.2), num
    )

    nband = train_image.shape[3]

    # NHWC -> NCHW
    train_image = np.transpose(train_image, (0, 3, 1, 2))
    train_image_LIDAR = np.transpose(train_image_LIDAR, (0, 3, 1, 2))

    if args.augment:
        transform_train = [CenterResizeCrop(scale_begin=args.scale, windowsize=windowsize)]
    else:
        transform_train = None

    untrain_dataset = HyperData((train_image, train_image_LIDAR, train_label), transform_train, nb_classes=nclass)
    untrain_loader = DataLoader(dataset=untrain_dataset, batch_size=batch_size, shuffle=True)

    net = mae_vit_HSIandLIDAR_patch3(
        img_size=(windowsize, windowsize),
        in_chans=nband,
        in_chans_LIDAR=in_chans_LIDAR,
        hid_chans=args.hid_chans,
        hid_chans_LIDAR=args.hid_chans_LIDAR,
        embed_dim=args.encoder_dim,
        depth=args.encoder_depth,
        num_heads=args.encoder_num_heads,
        mlp_ratio=args.mlp_ratio,
        decoder_embed_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads,
        nb_classes=nclass,
        global_pool=False
    ).cuda()

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    tic = time.time()
    for epoch in range(num_epoch):
        net.train()
        total_loss = 0.0

        for idx, (x, x_LIDAR, y) in enumerate(untrain_loader):
            x = x.cuda()
            x_LIDAR = x_LIDAR.cuda()
            y = y.cuda()

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],)).cuda()
            model_kwargs = dict(y=y)
            model_kwargs['mask_ratio'] = mask_ratio

            noise = torch.randn_like(x)
            x_t = diffusion.q_sample(x, t, noise)
            noise_L = torch.randn_like(x_LIDAR)
            x_L_t = diffusion.q_sample(x_LIDAR, t, noise_L)

            pred_imgs, pred_imgs_L, logits, mask, mask_L = net(x_t, x_L_t, t, **model_kwargs)

            # 分类 loss（你原来虽然算了但没加，这里照旧不加）
            _ = criterion(logits / args.temperature, y)

            loss_mse_m, loss_mse_L_m, loss_mse_v, loss_mse_L_v = DDPM_LOSS(
                pred_imgs, pred_imgs_L, x, x_LIDAR, mask, mask_L, size
            )

            loss = 0.1 * (loss_mse_m + loss_mse_L_m) + loss_mse_v + loss_mse_L_v

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach().cpu())

        scheduler.step()
        total_loss = total_loss / (idx + 1)
        print(f"epoch: {epoch}  loss: {total_loss:.6f}")

    toc = time.time()
    print("Training time(s):", toc - tic)

    torch.save({'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': num_epoch}, './net.pt')
    print("Saved to ./net.pt")

print("--------" + net_name + " Training Finished-----------")
