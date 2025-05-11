__all__ = ['MFTNet_backbone']

import torch
from torch import nn
import torch.nn.functional as F
from layers.RevIN import RevIN
from layers.cross_Transformer import Trans_C


class AdaptiveTemporalAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_model // self.n_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, n_heads, seq_len, d_model // n_heads]

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d_model ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        out = self.out(out)
        return out


class MFTNet_backbone(nn.Module):
    def __init__(self, ablation: int, mlp_drop: float, use_nys: int, output: int, mlp_hidden: int, cf_dim: int,
                 cf_depth: int, cf_heads: int, cf_mlp: int, cf_head_dim: int, cf_drop: float, c_in: int,
                 context_window: int, target_window: int, patch_len: int, stride: int, d_model: int,
                 head_dropout=0, padding_patch=None, individual=False, revin=True, affine=True, subtract_last=False,
                 **kwargs):
        super().__init__()
        self.use_nys = use_nys
        self.ablation = ablation
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        self.output = output
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.targetwindow = target_window
        self.horizon = self.targetwindow
        patch_num = int((context_window - patch_len) / stride + 1)
        self.norm = nn.LayerNorm(patch_len)
        self.re_attn = True
        if self.use_nys == 0:
            self.fre_transformer = Trans_C(dim=cf_dim, depth=cf_depth, heads=cf_heads, mlp_dim=cf_mlp,
                                           dim_head=cf_head_dim, dropout=cf_drop, patch_dim=patch_len * 2,
                                           horizon=self.horizon * 2, d_model=d_model * 2)
        else:
            pass
        self.head_nf_f = d_model * 2 * patch_num
        self.n_vars = c_in
        self.individual = individual
        self.head_f1 = Flatten_Head(self.individual, self.n_vars, self.head_nf_f, target_window,
                                    head_dropout=head_dropout)
        self.head_f2 = Flatten_Head(self.individual, self.n_vars, self.head_nf_f, target_window,
                                    head_dropout=head_dropout)
        self.ircom = nn.Linear(self.targetwindow * 2, self.targetwindow)
        self.rfftlayer = nn.Linear(self.targetwindow * 2 - 2, self.targetwindow)
        self.final = nn.Linear(self.targetwindow * 2, self.targetwindow)
        self.get_r = nn.Linear(d_model * 2, d_model * 2)
        self.get_i = nn.Linear(d_model * 2, d_model * 2)
        self.output1 = nn.Linear(target_window, target_window)
        self.input = nn.Linear(c_in, patch_len * 2)
        self.outpt = nn.Linear(d_model * 2, c_in)
        self.abfinal = nn.Linear(patch_len * patch_num, target_window)

        # Add Adaptive Temporal Attention
        self.adaptive_temporal_attention = AdaptiveTemporalAttention(d_model, cf_heads, cf_drop)

    def forward(self, z):
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)
        z = torch.fft.fft(z)
        z1 = z.real
        z2 = z.imag
        z1 = z1.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z2 = z2.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z1 = z1.permute(0, 2, 1, 3)
        z2 = z2.permute(0, 2, 1, 3)
        batch_size = z1.shape[0]
        patch_num = z1.shape[1]
        c_in = z1.shape[2]
        patch_len = z1.shape[3]
        z1 = torch.reshape(z1, (batch_size * patch_num, c_in, z1.shape[-1]))
        z2 = torch.reshape(z2, (batch_size * patch_num, c_in, z2.shape[-1]))
        z = self.fre_transformer(torch.cat((z1, z2), -1))
        z1 = self.get_r(z)
        z2 = self.get_i(z)
        z1 = torch.reshape(z1, (batch_size, patch_num, c_in, z1.shape[-1]))
        z2 = torch.reshape(z2, (batch_size, patch_num, c_in, z2.shape[-1]))
        z1 = z1.permute(0, 2, 1, 3)
        z2 = z2.permute(0, 2, 1, 3)
        z1 = self.head_f1(z1)
        z2 = self.head_f2(z2)
        z = torch.fft.ifft(torch.complex(z1, z2))
        zr = z.real
        zi = z.imag
        z = self.ircom(torch.cat((zr, zi), -1))
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)
        return z


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars
        if self.individual:
            self.linears1 = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears1.append(nn.Linear(288, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear1 = nn.Linear(nf, nf)
            self.linear2 = nn.Linear(nf, nf)
            self.linear3 = nn.Linear(nf, nf)
            self.linear4 = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                print(f"Actual input shape before linear layer: {z.shape}")
                z = self.linears1[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = F.relu(self.linear1(x)) + x
            x = F.relu(self.linear2(x)) + x
            x = F.relu(self.linear3(x)) + x
            x = self.linear4(x)
        return x


class Flatten_Head_t(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear1 = nn.Linear(nf, nf)
        self.linear2 = nn.Linear(nf, nf)
        self.linear3 = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.linear1(x)) + x
        x = F.relu(self.linear2(x)) + x
        x = self.linear3(x)
        return x