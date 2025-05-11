# MFTNet.py
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from layers.MFTNet_backbone import MFTNet_backbone

__all__ = ['MFTNet']

class Model(nn.Module):
    def __init__(self, configs, max_seq_len: Optional[int] = 1024, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 norm: str = 'BatchNorm', attn_dropout: float = 0.3, act: str = "gelu", key_padding_mask: bool = 'auto',
                 padding_var: Optional[int] = None, attn_mask: Optional[Tensor] = None, res_attention: bool = True,
                 pre_norm: bool = False, store_attn: bool = False, pe: str = 'zeros', learn_pe: bool = True,
                 pretrain_head: bool = False, head_type='flatten', verbose: bool = False, **kwargs):
        super().__init__()

        # Load parameters
        c_in = configs.enc_in
        self.configs = configs

        # Multi-scale patch embedding layers
        self.patch_embeddings = nn.ModuleList([
            nn.Linear(c_in * patch_len, configs.d_model) for patch_len in [4, 8, 16]  # Different patch lengths
        ])

        # Initialize MFTNet backbone
        self.model = MFTNet_backbone(
            ablation=configs.ablation,
            mlp_drop=configs.mlp_drop,
            use_nys=configs.use_nys,
            output=configs.output,
            mlp_hidden=configs.mlp_hidden,
            cf_dim=configs.cf_dim,
            cf_depth=configs.cf_depth,
            cf_heads=configs.cf_heads,
            cf_mlp=configs.cf_mlp,
            cf_head_dim=configs.cf_head_dim,
            cf_drop=configs.cf_drop,
            c_in=c_in,
            context_window=configs.seq_len,
            target_window=configs.pred_len,
            patch_len=configs.patch_len,
            stride=configs.stride,
            d_model=configs.d_model,
            head_dropout=configs.head_dropout,
            padding_patch=configs.padding_patch,
            individual=configs.individual,
            revin=configs.revin,
            affine=configs.affine,
            subtract_last=configs.subtract_last,
            **kwargs
        )

    def forward(self, x):  # x: [Batch, Input length, Channel]
        batch_size, seq_len, _ = x.size()
        multi_scale_features = []

        # Extract multi-scale patches
        for patch_len, patch_embedding in zip([4, 8, 16], self.patch_embeddings):
            num_patches = seq_len // patch_len
            x_patches = x.unfold(1, patch_len, patch_len).permute(0, 2, 1, 3).contiguous()
            x_patches = x_patches.view(batch_size, num_patches, -1)  # [Batch, Num_Patches, Patch_Length * Channels]

            # Patch embedding
            x_patches = patch_embedding(x_patches)  # [Batch, Num_Patches, d_model]
            multi_scale_features.append(x_patches)

        # Concatenate multi-scale features
        x = torch.cat(multi_scale_features, dim=1)  # [Batch, Total_Patches, d_model]

        # Forward through the model
        x = self.model(x)

        return x.permute(0, 2, 1)  # Restore to [Batch, Input length, Channel]

    def compute_loss(self, predictions, targets):
        pass