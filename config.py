import argparse
import torch.nn


def load_args():
    parser = argparse.ArgumentParser()

    # Pre training
    parser.add_argument('--train_num_perclass', type=int, default=40)
    parser.add_argument('--windowsize', type=int, default=11)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--fine_tuned_epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--fine_tuned_lr', type=float, default=1e-4)

    parser.add_argument('--type', type=str, default='none')
    parser.add_argument('--dataset', type=str, default='Trento')  # ✅ 默认改成 Trento（你现在就是 Trento）

    # Network
    parser.add_argument('--mask_ratio', default=0.7, type=float)
    parser.add_argument('--mlp_ratio', default=2.0, type=float)
    parser.add_argument('--hid_chans', default=128, type=int)
    parser.add_argument('--hid_chans_LIDAR', default=128, type=int)

    # Augmentation
    parser.add_argument('--augment', default=True, type=bool)
    parser.add_argument('--scale', default=9, type=int)

    # MAE encoder specifics
    parser.add_argument('--encoder_dim', default=128, type=int)
    parser.add_argument('--encoder_depth', default=4, type=int)
    parser.add_argument('--encoder_num_heads', default=8, type=int)

    # MAE decoder specifics
    parser.add_argument('--decoder_dim', default=128, type=int)
    parser.add_argument('--decoder_depth', default=3, type=int)
    parser.add_argument('--decoder_num_heads', default=8, type=int)

    # options for supervised MAE
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--cls_loss_ratio', default=0.005, type=float)

    # ✅ 新增：类别数。0 表示自动从数据推断
    parser.add_argument('--nb_classes', default=0, type=int)

    # ✅ 新增：是否跑 finetune/整图评测（你现在是 patch-only，默认关）
    parser.add_argument('--do_finetune', default=False, type=bool)

    args = parser.parse_args()
    return args
