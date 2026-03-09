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
    Lightweight UNet used as residual feature enhancer
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
        x = x.flatten(2).transpose(1, 2)  # [B, L, D]
        return x


# =========================================================
# =================== Prompt Utilities ====================
# =========================================================

def build_underwater_prompts(class_name):
    prompts = [
        f"a photo of a {class_name} underwater",
        f"an underwater image of a {class_name}",
        f"a blurry underwater photo of a {class_name}",
        f"a low-visibility underwater image of a {class_name}",
        f"a blue-green underwater photo of a {class_name}",
        f"a marine organism called {class_name}",
        f"an underwater creature of type {class_name}",
        f"a close-up underwater shot of a {class_name}",
        f"a {class_name} in the ocean underwater",
        f"a {class_name} near coral reef underwater",
    ]
    return prompts


def aggregate_text_features(text_features, method="mean"):
    """
    text_features:
        [num_prompts, dim] or [num_classes, num_prompts, dim]
    """
    if text_features.dim() == 2:
        if method == "mean":
            feat = text_features.mean(dim=0)
        elif method == "max":
            feat = text_features.max(dim=0)[0]
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")
        return F.normalize(feat, dim=-1)

    elif text_features.dim() == 3:
        if method == "mean":
            feat = text_features.mean(dim=1)
        elif method == "max":
            feat = text_features.max(dim=1)[0]
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")
        return F.normalize(feat, dim=-1)

    else:
        raise ValueError("text_features must be 2D or 3D")


# =========================================================
# ===================== Helper MLP ========================
# =========================================================

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3, dropout=0.0):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =========================================================
# =============== Query-based Mask Decoder ================
# =========================================================

class QueryMaskDecoder(nn.Module):
    """
    A light query-based instance segmentation head.

    Inputs:
        memory_tokens: [B, L, D]
        feat_map:       [B, D, H, W]
        text_features:  [num_classes, text_dim] or None

    Outputs:
        pred_masks:       [B, Q, H_img, W_img]
        pred_logits:      [B, Q, num_classes] or None
        pred_objectness:  [B, Q, 1]
        query_features:   [B, Q, D]
    """
    def __init__(
        self,
        embed_dim=768,
        num_queries=50,
        num_heads=8,
        num_decoder_layers=3,
        mask_dim=256,
        proj_dim=512,
        text_dim=512,
        upsample_scale=16,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.mask_dim = mask_dim
        self.upsample_scale = upsample_scale

        self.query_embed = nn.Embedding(num_queries, embed_dim)

        self.query_layers = nn.ModuleList()
        self.query_norms = nn.ModuleList()
        for _ in range(num_decoder_layers):
            layer = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
            self.query_layers.append(layer)
            self.query_norms.append(nn.LayerNorm(embed_dim))

        self.memory_proj = nn.Linear(embed_dim, embed_dim)
        self.query_ffn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Linear(embed_dim * 4, embed_dim),
            )
            for _ in range(num_decoder_layers)
        ])

        self.feat_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, mask_dim, kernel_size=1),
        )

        self.mask_embed_head = MLP(embed_dim, embed_dim, mask_dim, num_layers=3)
        self.objectness_head = MLP(embed_dim, embed_dim, 1, num_layers=3)

        self.query_proj = MLP(embed_dim, embed_dim, proj_dim, num_layers=2)
        self.text_proj = nn.Identity() if proj_dim == text_dim else nn.Linear(text_dim, proj_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, memory_tokens, feat_map, text_features=None, output_size=None):
        """
        memory_tokens: [B, L, D]
        feat_map: [B, D, H, W]
        """
        B, L, D = memory_tokens.shape

        memory = self.memory_proj(memory_tokens)  # [B, L, D]
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # [B, Q, D]

        for attn, norm, ffn in zip(self.query_layers, self.query_norms, self.query_ffn):
            attn_out, _ = attn(query=queries, key=memory, value=memory)
            queries = norm(queries + attn_out)
            queries = norm(queries + ffn(queries))

        query_features = queries  # [B, Q, D]

        feat_map = self.feat_proj(feat_map)  # [B, mask_dim, H, W]
        mask_embed = self.mask_embed_head(query_features)  # [B, Q, mask_dim]

        pred_masks = torch.einsum("bqc,bchw->bqhw", mask_embed, feat_map)  # [B, Q, H, W]

        if output_size is not None:
            pred_masks = F.interpolate(
                pred_masks,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

        pred_objectness = self.objectness_head(query_features)  # [B, Q, 1]

        pred_logits = None
        if text_features is not None:
            text_features = self.text_proj(text_features)
            text_features = F.normalize(text_features, dim=-1)

            query_proj = self.query_proj(query_features)
            query_proj = F.normalize(query_proj, dim=-1)

            pred_logits = self.logit_scale.exp() * torch.einsum(
                "bqd,cd->bqc", query_proj, text_features
            )

        return {
            "pred_masks": pred_masks,
            "pred_logits": pred_logits,
            "pred_objectness": pred_objectness,
            "query_features": query_features,
        }


# =========================================================
# ================= DiffCLIP Seg Model ====================
# =========================================================

class MaskedAutoencoderViT(nn.Module):
    """
    DiffCLIP-style dual-branch encoder + reconstruction + open-vocab instance segmentation

    img_A: RGB image               [B, 3, H, W]
    img_B: geometric prior image   [B, 1, H, W]   e.g. Canny / depth

    Modes:
        - "pretrain": masked reconstruction
        - "seg": open-vocab instance segmentation
    """

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
        proj_dim=512,
        text_dim=512,
        num_queries=50,
        mask_dim=256,
    ):
        super().__init__()

        # ---------- UNet preprocess ----------
        self.unet_A = SimpleUNet(in_chans, base_chans=32)
        self.unet_B = SimpleUNet(in_chans_B, base_chans=32)

        # ---------- Dim reduction ----------
        self.dimen_A = nn.Conv2d(in_chans, hid_chans, 1)
        self.dimen_B = nn.Conv2d(in_chans_B, hid_chans_B, 1)

        # ---------- Patch embedding ----------
        self.patch_embed_A = PatchEmbed(img_size, patch_size, hid_chans, embed_dim)
        self.patch_embed_B = PatchEmbed(img_size, patch_size, hid_chans_B, embed_dim)

        num_patches = self.patch_embed_A.num_patches
        self.num_patches = num_patches
        self.patch_grid = int(math.sqrt(num_patches))

        # ---------- Encoder ----------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # ---------- Reconstruction decoder ----------
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred_A = nn.Linear(decoder_embed_dim, patch_size ** 2 * hid_chans)
        self.decoder_pred_B = nn.Linear(decoder_embed_dim, patch_size ** 2 * hid_chans_B)

        # ---------- Token fusion ----------
        self.token_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.cls_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.feature_fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # ---------- Segmentation decoder ----------
        self.seg_decoder = QueryMaskDecoder(
            embed_dim=embed_dim,
            num_queries=num_queries,
            num_heads=8,
            num_decoder_layers=3,
            mask_dim=mask_dim,
            proj_dim=proj_dim,
            text_dim=text_dim,
            upsample_scale=patch_size,
        )

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.num_queries = num_queries

        self.initialize_weights()

    # =====================================================
    # ================= Initialization ====================
    # =====================================================

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.pos_embed, std=0.02)
        torch.nn.init.normal_(self.decoder_pos_embed, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    # =====================================================
    # ==================== Patch Utils =====================
    # =====================================================

    def patchify(self, imgs, in_chans):
        """
        imgs: (N, C, H, W)
        x:    (N, L, patch_size**2 * C)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], in_chans, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * in_chans))
        return x

    def tokens_to_map(self, tokens):
        """
        tokens: [B, L, D]
        return: [B, D, H, W]
        """
        B, L, D = tokens.shape
        H = W = int(math.sqrt(L))
        assert H * W == L, f"L={L} is not a square number"
        feat = tokens.transpose(1, 2).reshape(B, D, H, W)
        return feat

    # =====================================================
    # ===================== Masking ========================
    # =====================================================

    def random_masking(self, x, mask_ratio):
        """
        x: [N, L, D]
        """
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

    # =====================================================
    # ================= Encoder Forward ====================
    # =====================================================

    def forward_encoder_single(self, x, mask_ratio):
        """
        x: [B, L, D]
        """
        x_vis, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token.expand(x_vis.shape[0], -1, -1)
        x_vis = torch.cat([cls_token, x_vis], dim=1)

        x_vis = x_vis + self.pos_embed[:, :x_vis.shape[1], :]

        for blk in self.blocks:
            x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis, mask, ids_restore

    def encode_images(self, img_A, img_B, mask_ratio=0.75):
        # UNet enhance
        img_A = self.unet_A(img_A)
        img_B = self.unet_B(img_B)

        # Dim reduce
        img_A_reduced = self.dimen_A(img_A)
        img_B_reduced = self.dimen_B(img_B)

        # Patch embed
        xA = self.patch_embed_A(img_A_reduced)
        xB = self.patch_embed_B(img_B_reduced)

        # Encode
        xA_enc, maskA, idsA = self.forward_encoder_single(xA, mask_ratio)
        xB_enc, maskB, idsB = self.forward_encoder_single(xB, mask_ratio)

        return {
            "img_A_enhanced": img_A,
            "img_B_enhanced": img_B,
            "img_A_reduced": img_A_reduced,
            "img_B_reduced": img_B_reduced,
            "xA_tokens": xA,
            "xB_tokens": xB,
            "xA_enc": xA_enc,
            "xB_enc": xB_enc,
            "maskA": maskA,
            "maskB": maskB,
            "idsA": idsA,
            "idsB": idsB,
        }

    # =====================================================
    # ================= Decoder Forward ====================
    # =====================================================

    def forward_decoder_single(self, x_enc, ids_restore, decoder_pred):
        """
        x_enc: [B, 1+L_visible, D]
        """
        x = self.decoder_embed(x_enc)

        B, L_vis_plus_cls, D = x.shape
        L = ids_restore.shape[1]

        mask_tokens = self.mask_token.repeat(B, L + 1 - L_vis_plus_cls, 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).repeat(1, 1, D))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.decoder_pos_embed[:, :x.shape[1], :]

        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)
        pred = decoder_pred(x[:, 1:, :])
        return pred

    # =====================================================
    # ================= Segmentation Path ==================
    # =====================================================

    def forward_segmentation_features(self, img_A, img_B):
        """
        For segmentation, usually no random masking.
        """
        outs = self.encode_images(img_A, img_B, mask_ratio=0.0)

        cls_A = outs["xA_enc"][:, 0]         # [B, D]
        cls_B = outs["xB_enc"][:, 0]         # [B, D]
        tokens_A = outs["xA_enc"][:, 1:]     # [B, L, D]
        tokens_B = outs["xB_enc"][:, 1:]     # [B, L, D]

        fused_cls = self.cls_fusion(torch.cat([cls_A, cls_B], dim=-1))
        fused_tokens = self.token_fusion(torch.cat([tokens_A, tokens_B], dim=-1))

        feat_A = self.tokens_to_map(tokens_A)
        feat_B = self.tokens_to_map(tokens_B)
        feat_map = self.feature_fusion(torch.cat([feat_A, feat_B], dim=1))

        return {
            "fused_cls": fused_cls,
            "fused_tokens": fused_tokens,
            "feat_map": feat_map,
            "outs": outs,
        }

    # =====================================================
    # ===================== Loss Utils =====================
    # =====================================================

    def reconstruction_loss(self, pred, target, mask):
        """
        pred:   [B, L, patch_dim]
        target: [B, L, patch_dim]
        mask:   [B, L], 1 means masked
        """
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss

    @staticmethod
    def dice_loss(inputs, targets, eps=1e-6):
        """
        inputs:  [B, Q, H, W] logits
        targets: [B, Q, H, W] {0,1}
        """
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)

        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)

        loss = 1 - (numerator + eps) / (denominator + eps)
        return loss.mean()

    @staticmethod
    def sigmoid_ce_loss(inputs, targets):
        """
        inputs:  [B, Q, H, W] logits
        targets: [B, Q, H, W]
        """
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="mean")
        return loss

    # =====================================================
    # ======================= Forward ======================
    # =====================================================

    def forward(
        self,
        img_A,
        img_B,
        text_features=None,
        mask_ratio=0.75,
        mode="seg",
        return_reconstruction=False,
    ):
        """
        Args:
            img_A: [B, 3, H, W]
            img_B: [B, 1, H, W]
            text_features: [num_classes, text_dim] or None
            mode:
                - "pretrain"
                - "seg"

        Returns:
            mode == "pretrain":
                {
                    "predA", "predB", "maskA", "maskB",
                    "targetA", "targetB", "loss_rec_A", "loss_rec_B", "loss_rec"
                }

            mode == "seg":
                {
                    "pred_masks": [B, Q, H, W],
                    "pred_logits": [B, Q, C] or None,
                    "pred_objectness": [B, Q, 1],
                    "query_features": [B, Q, D],
                    "fused_tokens": [B, L, D],
                    "fused_cls": [B, D],
                }
        """

        if mode == "pretrain":
            outs = self.encode_images(img_A, img_B, mask_ratio=mask_ratio)

            predA = self.forward_decoder_single(
                outs["xA_enc"], outs["idsA"], self.decoder_pred_A
            )
            predB = self.forward_decoder_single(
                outs["xB_enc"], outs["idsB"], self.decoder_pred_B
            )

            targetA = self.patchify(outs["img_A_reduced"], self.dimen_A.out_channels)
            targetB = self.patchify(outs["img_B_reduced"], self.dimen_B.out_channels)

            loss_rec_A = self.reconstruction_loss(predA, targetA, outs["maskA"])
            loss_rec_B = self.reconstruction_loss(predB, targetB, outs["maskB"])

            return {
                "predA": predA,
                "predB": predB,
                "maskA": outs["maskA"],
                "maskB": outs["maskB"],
                "targetA": targetA,
                "targetB": targetB,
                "loss_rec_A": loss_rec_A,
                "loss_rec_B": loss_rec_B,
                "loss_rec": loss_rec_A + loss_rec_B,
            }

        elif mode == "seg":
            seg_feats = self.forward_segmentation_features(img_A, img_B)

            seg_out = self.seg_decoder(
                memory_tokens=seg_feats["fused_tokens"],
                feat_map=seg_feats["feat_map"],
                text_features=text_features,
                output_size=img_A.shape[-2:],
            )

            result = {
                "pred_masks": seg_out["pred_masks"],
                "pred_logits": seg_out["pred_logits"],
                "pred_objectness": seg_out["pred_objectness"],
                "query_features": seg_out["query_features"],
                "fused_tokens": seg_feats["fused_tokens"],
                "fused_cls": seg_feats["fused_cls"],
            }

            if return_reconstruction:
                outs = self.encode_images(img_A, img_B, mask_ratio=0.0)
                predA = self.forward_decoder_single(
                    outs["xA_enc"], outs["idsA"], self.decoder_pred_A
                )
                predB = self.forward_decoder_single(
                    outs["xB_enc"], outs["idsB"], self.decoder_pred_B
                )
                result["predA"] = predA
                result["predB"] = predB

            return result

        else:
            raise ValueError(f"Unsupported mode: {mode}")


# =========================================================
# ================== Builder Function =====================
# =========================================================

def mae_vit_underwater_ovinstseg_patch16(
    img_size=224,
    in_chans=3,
    in_chans_B=1,
    proj_dim=512,
    text_dim=512,
    num_queries=50,
    mask_dim=256,
):
    model = MaskedAutoencoderViT(
        img_size=img_size,
        patch_size=16,
        in_chans=in_chans,
        in_chans_B=in_chans_B,
        hid_chans=32,
        hid_chans_B=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=384,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        proj_dim=proj_dim,
        text_dim=text_dim,
        num_queries=num_queries,
        mask_dim=mask_dim,
    )
    return model


# =========================================================
# ===================== Quick Test ========================
# =========================================================

if __name__ == "__main__":
    B = 2
    num_classes = 7

    model = mae_vit_underwater_ovinstseg_patch16(
        img_size=224,
        in_chans=3,
        in_chans_B=1,
        proj_dim=512,
        text_dim=512,
        num_queries=20,
        mask_dim=256,
    )

    img_A = torch.randn(B, 3, 224, 224)
    img_B = torch.randn(B, 1, 224, 224)
    text_features = F.normalize(torch.randn(num_classes, 512), dim=-1)

    # pretrain
    out_pre = model(img_A, img_B, mode="pretrain", mask_ratio=0.75)
    print("=== pretrain ===")
    print("keys:", out_pre.keys())
    print("loss_rec:", out_pre["loss_rec"].item())

    # segmentation
    out_seg = model(
        img_A,
        img_B,
        text_features=text_features,
        mode="seg",
    )
    print("\n=== seg ===")
    print("keys:", out_seg.keys())
    print("pred_masks:", out_seg["pred_masks"].shape)         # [B, Q, H, W]
    print("pred_logits:", out_seg["pred_logits"].shape)       # [B, Q, C]
    print("pred_objectness:", out_seg["pred_objectness"].shape)  # [B, Q, 1]
