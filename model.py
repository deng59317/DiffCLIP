import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import _assert
from timm.models.vision_transformer import Block
from timm.models.layers import to_2tuple

# =========================================================
# =============== Lightweight Residual UNet ===============
# =========================================================

class ConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chans, out_chans, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SimpleUNet(nn.Module):
    """
    Lightweight UNet used as *residual feature enhancer*
    """
    def __init__(self, in_chans, base_chans=32):
        super().__init__()
        self.enc1 = ConvBlock(in_chans, base_chans)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_chans, base_chans * 2)
        self.up = nn.ConvTranspose2d(base_chans * 2, base_chans, 2, stride=2)
        self.dec1 = ConvBlock(base_chans * 2, base_chans)
        self.out_conv = nn.Conv2d(base_chans, in_chans, 1)

    def forward(self, x):
        identity = x
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d1 = self.up(e2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return identity + self.out_conv(d1)


# =========================================================
# ===================== Patch Embed =======================
# =========================================================

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0] and W == self.img_size[1], "Image size mismatch")
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# =========================================================
# ================= Masked Autoencoder ====================
# =========================================================

class MaskedAutoencoderViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        in_chans_B=1,
        hid_chans=32,
        hid_chans_B=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=384,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        nb_classes=10,
    ):
        super().__init__()

        # ---------- UNet Preprocess ----------
        self.unet_A = SimpleUNet(in_chans, base_chans=32)
        self.unet_B = SimpleUNet(in_chans_B, base_chans=32)

        # ---------- Dim reduction ----------
        self.dimen_A = nn.Conv2d(in_chans, hid_chans, 1)
        self.dimen_B = nn.Conv2d(in_chans_B, hid_chans_B, 1)

        # ---------- Patch embed ----------
        self.patch_embed_A = PatchEmbed(img_size, patch_size, hid_chans, embed_dim)
        self.patch_embed_B = PatchEmbed(img_size, patch_size, hid_chans_B, embed_dim)

        num_patches = self.patch_embed_A.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # ---------- Decoder ----------
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred_A = nn.Linear(decoder_embed_dim, patch_size**2 * hid_chans)
        self.decoder_pred_B = nn.Linear(decoder_embed_dim, patch_size**2 * hid_chans_B)

        # ---------- Fusion Decoder ----------
        self.fusion_decoder = nn.Sequential(
            nn.Linear(embed_dim * 2, decoder_embed_dim),
            nn.GELU(),
            nn.Linear(decoder_embed_dim, decoder_embed_dim),
        )
        self.fusion_pred_A = nn.Linear(decoder_embed_dim, patch_size**2 * hid_chans)
        self.fusion_pred_B = nn.Linear(decoder_embed_dim, patch_size**2 * hid_chans_B)

        # ---------- Classification ----------
        self.cls_head = nn.Linear(embed_dim * 2, nb_classes)

        self.patch_size = patch_size
        self.img_size = img_size

    # ---------------- Masking ----------------
    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_visible, mask, ids_restore

    # ---------------- Forward ----------------
    def forward(self, img_A, img_B, mask_ratio=0.75, return_fusion=False):
        # UNet enhance
        img_A = self.unet_A(img_A)
        img_B = self.unet_B(img_B)

        # Dim reduce
        img_A = self.dimen_A(img_A)
        img_B = self.dimen_B(img_B)

        # Patch embed
        xA = self.patch_embed_A(img_A)
        xB = self.patch_embed_B(img_B)

        xA_vis, maskA, idsA = self.random_masking(xA, mask_ratio)
        xB_vis, maskB, idsB = self.random_masking(xB, mask_ratio)

        cls = self.cls_token.expand(xA_vis.size(0), -1, -1)
        xA = torch.cat([cls, xA_vis], dim=1)
        xB = torch.cat([cls, xB_vis], dim=1)

        xA = xA + self.pos_embed[:, :xA.size(1)]
        xB = xB + self.pos_embed[:, :xB.size(1)]

        for blk in self.blocks:
            xA = blk(xA)
            xB = blk(xB)

        xA = self.norm(xA)
        xB = self.norm(xB)

        logits = self.cls_head(torch.cat([xA[:, 0], xB[:, 0]], dim=-1))

        if not return_fusion:
            return logits, maskA, maskB

        # -------- Fusion reconstruction --------
        fused = torch.cat([xA, xB], dim=-1)
        fused = self.fusion_decoder(fused)
        predA = self.fusion_pred_A(fused[:, 1:])
        predB = self.fusion_pred_B(fused[:, 1:])

        return logits, maskA, maskB, predA, predB
