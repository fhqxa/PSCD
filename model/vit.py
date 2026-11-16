import torch
import torch.nn as nn
import torch.nn.functional as F

from models.prototype_bank import (
    EMAClassAccuracyBank,
    EMAConfusionBank,
    EMAPrototypeBank,
)
from models.resnet32 import NormedLinear


def load_deit_small_checkpoint(encoder, checkpoint_path):
    """Load the ImageNet-1K DeiT-S checkpoint into our ViT-S encoder."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source = checkpoint.get("model", checkpoint)
    if not isinstance(source, dict):
        raise TypeError("DeiT checkpoint must contain a state dictionary.")

    converted = {
        "patch_embed.weight": source["patch_embed.proj.weight"],
        "patch_embed.bias": source["patch_embed.proj.bias"],
        "cls_token": source["cls_token"],
        "pos_embed": source["pos_embed"],
        "blocks.norm.weight": source["norm.weight"],
        "blocks.norm.bias": source["norm.bias"],
    }
    for index in range(12):
        source_prefix = "blocks.{}.".format(index)
        target_prefix = "blocks.layers.{}.".format(index)
        block_mapping = {
            "norm1.weight": "norm1.weight",
            "norm1.bias": "norm1.bias",
            "attn.qkv.weight": "self_attn.in_proj_weight",
            "attn.qkv.bias": "self_attn.in_proj_bias",
            "attn.proj.weight": "self_attn.out_proj.weight",
            "attn.proj.bias": "self_attn.out_proj.bias",
            "norm2.weight": "norm2.weight",
            "norm2.bias": "norm2.bias",
            "mlp.fc1.weight": "linear1.weight",
            "mlp.fc1.bias": "linear1.bias",
            "mlp.fc2.weight": "linear2.weight",
            "mlp.fc2.bias": "linear2.bias",
        }
        for source_suffix, target_suffix in block_mapping.items():
            converted[target_prefix + target_suffix] = (
                source[source_prefix + source_suffix]
            )

    target = encoder.state_dict()
    missing = sorted(set(target) - set(converted))
    unexpected = sorted(set(converted) - set(target))
    mismatched = [
        (key, tuple(converted[key].shape), tuple(target[key].shape))
        for key in sorted(set(target) & set(converted))
        if converted[key].shape != target[key].shape
    ]
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "Incompatible DeiT-S checkpoint: missing={}, unexpected={}, "
            "shape_mismatch={}".format(missing, unexpected, mismatched)
        )

    encoder.load_state_dict(converted, strict=True)
    return len(converted)


class VisionTransformerSmall(nn.Module):
    """ViT-S/16 encoder with the DeiT-Small dimensions used by CAR."""

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        grid_size = image_size // patch_size
        num_patches = grid_size * grid_size

        self.patch_embed = nn.Conv2d(
            3,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            norm=nn.LayerNorm(embed_dim),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        for module in self.modules():
            if isinstance(module, nn.MultiheadAttention):
                nn.init.trunc_normal_(module.in_proj_weight, std=0.02)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                "ViT-S/16 expects {}x{} inputs, got {}x{}".format(
                    self.image_size,
                    self.image_size,
                    x.shape[-2],
                    x.shape[-1],
                )
            )
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        return self.blocks(x)[:, 0]


class BCLModelViTSmall(nn.Module):
    def __init__(
        self,
        num_classes=100,
        feat_dim=128,
        prototype_momentum=0.9,
        confusion_momentum=0.95,
        enable_confusion_bank=False,
        enable_class_accuracy_bank=False,
        class_accuracy_momentum=0.9,
        enable_balanced_head=False,
        use_norm=True,
    ):
        super().__init__()
        dim_in = 384
        self.encoder = VisionTransformerSmall()
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.BatchNorm1d(dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim),
        )
        classifier = NormedLinear if use_norm else nn.Linear
        self.fc = classifier(dim_in, num_classes)
        self.fc_balanced = (
            classifier(dim_in, num_classes) if enable_balanced_head else None
        )
        self.head_fc = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.BatchNorm1d(dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim),
        )
        self.prototype_bank = EMAPrototypeBank(
            num_classes, feat_dim, momentum=prototype_momentum
        )
        self.feature_adapter = None
        if enable_confusion_bank:
            self.confusion_bank = EMAConfusionBank(
                num_classes, momentum=confusion_momentum
            )
        if enable_class_accuracy_bank:
            self.class_accuracy_bank = EMAClassAccuracyBank(
                num_classes, momentum=class_accuracy_momentum
            )

    def forward(self, x, return_balanced=False, return_raw=False):
        feat = self.encoder(x)
        unfn_feat = self.head(feat)
        feat_mlp = F.normalize(unfn_feat, dim=1)
        logits = self.fc(feat)
        rectified_feat = None
        rectifier_diagnostics = None
        if isinstance(self.feature_adapter, nn.Module):
            rectified_feat, rectifier_diagnostics = self.feature_adapter(
                feat, logits
            )
            logits = self.fc(rectified_feat)

        classifier_weights = self.fc.weight
        if classifier_weights.shape[0] == feat.shape[1]:
            classifier_weights = classifier_weights.T
        unfn_centers = self.head_fc(classifier_weights)
        centers_logits = F.normalize(unfn_centers, dim=1)

        outputs = (
            feat_mlp,
            logits,
            centers_logits,
            unfn_centers,
            unfn_feat,
        )
        if return_balanced:
            if self.fc_balanced is None:
                raise RuntimeError("Balanced classifier head is not enabled.")
            balanced_features = (
                rectified_feat if rectified_feat is not None else feat
            )
            outputs = outputs + (self.fc_balanced(balanced_features),)
        if return_raw:
            outputs = outputs + (feat,)
            if rectified_feat is not None:
                outputs = outputs + (rectified_feat, rectifier_diagnostics)
        return outputs
