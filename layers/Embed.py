all = ['MFTNet']

#Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
#conda activate MFTNet
from layers.MFTNet_backbone import MFTNet_backbone


class Model(nn.Module):
    def __init__(self, configs, max_seq_len: Optional[int] = 1024, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 norm: str = 'BatchNorm', attn_dropout: float = 0.3, act: str = "gelu", key_padding_mask: bool = 'auto',
                 padding_var: Optional[int] = None, attn_mask: Optional[Tensor] = None, res_attention: bool = True,
                 pre_norm: bool = False, store_attn: bool = False, pe: str = 'zeros', learn_pe: bool = True,
                 pretrain_head: bool = False, head_type='flatten', verbose: bool = False, **kwargs):
        super().__init__()

        # load parameters
        c_in = configs.enc_in
        patch_len = configs.patch_len

        # Patch embedding layer
        self.patch_embedding = nn.Linear(c_in * patch_len, configs.d_model)

        # Initialize MFTNet backbone
        self.model = MFTNet_backbone(...)  # 保持原有的初始化逻辑

    def forward(self, x):  # x: [Batch, Input length, Channel]
        batch_size, seq_len, _ = x.size()
        __all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.MFTNet_backbone import MFTNet_backbone



class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0.3,
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True,
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):

        super().__init__()

        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        output = 0#configs.output
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout

        individual = configs.individual

        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        use_nys = configs.use_nys
        ablation = configs.ablation


        ###
        cf_dim = configs.cf_dim
        cf_depth = configs.cf_depth
        cf_heads = configs.cf_heads
        cf_mlp = configs.cf_mlp
        cf_head_dim = configs.cf_head_dim
        cf_drop = configs.cf_drop
        mlp_hidden = configs.mlp_hidden
        mlp_drop = configs.mlp_drop

        self.model = MFTNet_backbone(ablation=ablation,mlp_drop=mlp_drop, use_nys=use_nys,output=output,mlp_hidden=mlp_hidden,c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride,
                                max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                subtract_last=subtract_last, verbose=verbose,cf_dim=cf_dim,cf_depth =cf_depth,cf_heads=cf_heads,cf_mlp=cf_mlp,cf_head_dim=cf_head_dim,cf_drop=cf_drop, **kwargs)


    def forward(self, x):           # x: [Batch, Input length, Channel]
        x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
        x = self.model(x)
        x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        return x #,oz,t,attn

        # Patch splitting
        num_patches = seq_len // configs.patch_len
        x_patches = x.unfold(1, configs.patch_len, configs.patch_len).permute(0, 2, 1, 3).contiguous()
        x_patches = x_patches.view(batch_size, num_patches, -1)  # [Batch, Num_Patches, Patch_Length * Channels]

        # Patch embedding
        x_patches = self.patch_embedding(x_patches)  # [Batch, Num_Patches, d_model]

        # Forward through the model
        x = self.model(x_patches)

        return x.permute(0, 2, 1)  # 恢复为 [Batch, Input length, Channel]

    def compute_loss(self, predictions, targets):
        pass