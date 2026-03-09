import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    elif v.lower() in ("no", "false", "f", "0", "n"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def load_args():
    parser = argparse.ArgumentParser()

    # =====================================================
    # Basic
    # =====================================================
    parser.add_argument('--dataset', type=str, default='MARIS')
    parser.add_argument('--type', type=str, default='none')
    parser.add_argument('--save_path', type=str, default='./net_seg.pt')
    parser.add_argument('--device', type=str, default='cuda')

    # =====================================================
    # Input / Image
    # =====================================================
    parser.add_argument('--windowsize', type=int, default=224)   # 输入图像大小
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)

    # =====================================================
    # Training
    # =====================================================
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--pretrain_epochs', type=int, default=20)
    parser.add_argument('--fine_tuned_epochs', type=int, default=50)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--fine_tuned_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # =====================================================
    # Dataset / Class
    # =====================================================
    parser.add_argument('--nb_classes', type=int, default=158)   # MARIS 细粒度类别数
    parser.add_argument('--train_num_perclass', type=int, default=0)  # 分割任务一般不用这个，保留兼容
    parser.add_argument('--val_ratio', type=float, default=0.2)

    # =====================================================
    # Open-vocabulary / Text
    # =====================================================
    parser.add_argument('--use_text', type=str2bool, default=True)
    parser.add_argument('--text_dim', type=int, default=512)
    parser.add_argument('--proj_dim', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=1.0)

    # =====================================================
    # Geometry branch
    # =====================================================
    parser.add_argument('--use_geometry', type=str2bool, default=True)
    parser.add_argument('--geometry_type', type=str, default='canny',
                        choices=['canny', 'sobel', 'laplacian', 'depth'])
    parser.add_argument('--in_chans', type=int, default=3)       # RGB
    parser.add_argument('--in_chans_B', type=int, default=1)     # geometry
    parser.add_argument('--hid_chans', type=int, default=32)
    parser.add_argument('--hid_chans_B', type=int, default=32)

    # =====================================================
    # MAE / Encoder
    # =====================================================
    parser.add_argument('--mask_ratio', type=float, default=0.7)
    parser.add_argument('--mlp_ratio', type=float, default=4.0)

    parser.add_argument('--encoder_dim', type=int, default=768)
    parser.add_argument('--encoder_depth', type=int, default=12)
    parser.add_argument('--encoder_num_heads', type=int, default=12)

    # =====================================================
    # Decoder
    # =====================================================
    parser.add_argument('--decoder_dim', type=int, default=384)
    parser.add_argument('--decoder_depth', type=int, default=4)
    parser.add_argument('--decoder_num_heads', type=int, default=8)

    # =====================================================
    # Segmentation head
    # =====================================================
    parser.add_argument('--num_queries', type=int, default=50)
    parser.add_argument('--mask_dim', type=int, default=256)
    parser.add_argument('--mask_threshold', type=float, default=0.5)

    # =====================================================
    # Loss weights
    # =====================================================
    parser.add_argument('--loss_mask_weight', type=float, default=1.0)
    parser.add_argument('--loss_dice_weight', type=float, default=1.0)
    parser.add_argument('--loss_cls_weight', type=float, default=0.5)
    parser.add_argument('--loss_obj_weight', type=float, default=0.5)
    parser.add_argument('--loss_rec_weight', type=float, default=0.1)
    parser.add_argument('--cls_loss_ratio', type=float, default=0.005)  # 保留兼容

    # =====================================================
    # Augmentation
    # =====================================================
    parser.add_argument('--augment', type=str2bool, default=True)
    parser.add_argument('--scale', type=int, default=9)
    parser.add_argument('--center_crop', type=str2bool, default=False)

    # =====================================================
    # Run mode
    # =====================================================
    parser.add_argument('--mode', type=str, default='seg', choices=['pretrain', 'seg'])
    parser.add_argument('--do_finetune', type=str2bool, default=True)
    parser.add_argument('--eval_only', type=str2bool, default=False)

    args = parser.parse_args()
    return args
