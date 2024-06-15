import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, BoolTensor, dropout
from torch.autograd import Variable
from torch.nn.modules.container import Sequential
class MCattention(nn.Module):
    def __init__(self,cnn_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(cnn_dim)
        self.qkv = nn.Linear(cnn_dim,cnn_dim*3,bias=False)

        self.proj = nn.Linear(cnn_dim, cnn_dim)
        self.proj_drop = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(cnn_dim)
        self.attn_h_A = mcAttention_Atrous(0,cnn_dim,4)
        self.attn_v_A = mcAttention_Atrous(1,cnn_dim, 4)
        self.attn_h_L = mcAttention_local(0,cnn_dim,4)
        self.attn_v_L = mcAttention_local(1,cnn_dim, 4)
        self.drop = nn.Dropout(0.1)
        self.mlp = nn.Sequential(
            nn.Linear(cnn_dim,cnn_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cnn_dim,cnn_dim),
            nn.Dropout(0.1),

        )
        self.norm3 = nn.LayerNorm(cnn_dim)

    def forward(self, x):

        B,C,H,W = x.shape
        x_ = x.permute(0,2,3,1)
        x_ = x_.reshape(B,H*W,C)
        b, L, C = x_.shape
        assert L == H * W

        img = self.norm1(x_)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
        atten_x_h = self.attn_h_L(qkv,H,W)
        atten_x_v = self.attn_v_L(qkv, H, W)
        atten_x = self.proj(atten_x_v+atten_x_h)
        x_ = x_ + self.drop(atten_x)
        x_ = self.norm2(x_)

        qkv = self.qkv(x_).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
        atten_x_h = self.attn_h_A(qkv, H, W)
        atten_x_v = self.attn_v_A(qkv, H, W)
        atten_x = self.proj(atten_x_v + atten_x_h)
        x_ = x_ + self.drop(atten_x)
        x_ = self.norm2(x_)

        x_ = x_ + self.drop(self.mlp(x_))
        x_ = self.norm3(x_)
        return x_

class mcAttention_local(nn.Module):
    def __init__(self,idx,cnn_dim,num_heads):
        super().__init__()
        self.idx = idx
        self.dim = cnn_dim
        self.num_heads = num_heads
        head_dim = self.dim // num_heads
        self.scale = head_dim ** -0.5
        self.attn_drop = nn.Dropout(0.4)

        self.get_v = nn.Conv2d(cnn_dim, cnn_dim, kernel_size=3, stride=1, padding=1, groups=cnn_dim)

    def get_lepe(self, x, func,H_sp,W_sp):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)

        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp) ### B', C, H', W'

        lepe = func(x) ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, H_sp* W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self,qkv,H,W):
        q,k,v = qkv[0],qkv[1],qkv[2]

        if self.idx == 0:
            H_sp,W_sp = 1,W
        else:
            H_sp,W_sp = H,1
        B, L, C = q.shape
        assert L == H * W
        q = self.im2cswin(q,H_sp,W_sp)
        k = self.im2cswin(k,H_sp,W_sp)
        # v = self.im2cswin(v,H_sp,W_sp)
        v, lepe = self.get_lepe(v, self.get_v, H_sp, W_sp)
        q = q.view(q.size(0), q.size(1), q.size(2),1,q.size(-1))
        k,v = extract_seq_patches(k,5,1),extract_seq_patches(v,5,1)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)
        x = (attn @ v)+lepe.unsqueeze(-2)
        x = x.transpose(1, 2).reshape(-1, H_sp * W_sp, C)  # B head N N @ B head N C
        x = windows2img(x, H_sp, W_sp, H, W).view(B, -1, C)  # B H' W' C
        return x
    def im2cswin(self, x,H_sp,W_sp):
        B, N,C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, H_sp, W_sp)
        x = x.reshape(-1, H_sp* W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x



class mcAttention_Atrous(nn.Module):
    def __init__(self,idx,cnn_dim,num_heads):
        super().__init__()
        self.idx = idx
        self.dim = cnn_dim
        self.num_heads = num_heads
        head_dim = self.dim // num_heads
        self.scale = head_dim ** -0.5
        self.attn_drop = nn.Dropout(0.4)
        self.dilation=3

        self.get_v = nn.Conv2d(cnn_dim, cnn_dim, kernel_size=3, stride=1, padding=1, groups=cnn_dim)

    def get_lepe(self, x, func,H_sp,W_sp):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)

        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp) ### B', C, H', W'

        lepe = func(x) ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, H_sp* W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self,qkv,H,W):
        q,k,v = qkv[0],qkv[1],qkv[2]

        if self.idx == 0:
            H_sp,W_sp = 1,W
        else:
            H_sp,W_sp = H,1
        B, L, C = q.shape
        assert L == H * W
        q = self.im2cswin(q,H_sp,W_sp)
        k = self.im2cswin(k,H_sp,W_sp)
        # v = self.im2cswin(v, H_sp, W_sp)
        v, lepe = self.get_lepe(v, self.get_v,H_sp,W_sp)

        seq_len = int(np.sqrt(L))

        if seq_len % self.dilation == 0:
            padding_size = 0
        else:  # != 0
            padding_size = (seq_len // self.dilation + 1) * self.dilation - seq_len
        assert (padding_size + seq_len) % self.dilation == 0
        q = F.pad(q, (0, 0, padding_size, 0), "constant", 0)
        k = F.pad(k, (0, 0, padding_size, 0), "constant", 0)
        v = F.pad(v, (0, 0, padding_size, 0), "constant", 0)
        lepe = F.pad(lepe, (0, 0, padding_size, 0), "constant", 0)
        padded_seq_len = q.size(2)
        dim = q.size(-1)
        q=q.view(-1,self.num_heads,padded_seq_len // self.dilation, self.dilation, dim)\
            .permute(0,1,3,2,4)\
            .reshape(-1,self.num_heads,padded_seq_len // self.dilation,dim)

        k = k.view(-1, self.num_heads, padded_seq_len // self.dilation, self.dilation, dim) \
            .permute(0, 1, 3, 2, 4) \
            .reshape(-1, self.num_heads, padded_seq_len // self.dilation, dim)

        v = v.view(-1, self.num_heads, padded_seq_len // self.dilation, self.dilation, dim) \
            .permute(0, 1, 3, 2, 4) \
            .reshape(-1, self.num_heads, padded_seq_len // self.dilation, dim)

        lepe = lepe.view(-1, self.num_heads, padded_seq_len // self.dilation, self.dilation, dim) \
            .permute(0, 1, 3, 2, 4) \
            .reshape(-1, self.num_heads, padded_seq_len // self.dilation, dim)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)
        x = (attn @ v)+lepe
        x = x.transpose(1, 2).reshape(-1, H_sp * W_sp, C)  # B head N N @ B head N C
        x = x.view(-1,self.dilation,padded_seq_len//self.dilation,self.dim)
        x = x.permute(0,2,1,3)
        x = x.reshape(-1,padded_seq_len,self.dim)
        if padding_size > 0:  # != 0
            x= x[:, :-padding_size]
        x = windows2img(x, H_sp, W_sp, H, W).view(B, -1, C)  # B H' W' C

        return x


    def im2cswin(self, x,H_sp,W_sp):
        B, N,C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, H_sp, W_sp)
        x = x.reshape(-1, H_sp* W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x


def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img

def extract_seq_patches(x: Tensor, kernel_size: int, dilation: int = 1):
    # x.shape: (batch_size, seq_len, embed_dim)
    seq_len, embed_dim = x.size(2), x.size(-1)
    patch_size = kernel_size + (dilation - 1) * (kernel_size - 1)
    padding_right = (patch_size - 1) // 2
    padding_left = patch_size - padding_right - 1
    # x = x.transpose(1, -1)#x.shape: (batch_size, embed_dim, seq_len)
    # padding_layer = nn.ConstantPad1d(padding=(padding_left, padding_right), value=0.0)
    # # https://pytorch.org/docs/stable/generated/torch.nn.ConstantPad1d.html#torch.nn.ConstantPad1d
    # x = padding_layer(x)#也可以用F.pad()函数
    # x = x.transpose(1, -1)#x.shape: (batch_size, seq_len+padding_left+padding_right, embed_dim)
    x = F.pad(x, (0, 0, padding_left, padding_right), mode="constant",
              value=0.0)  # x.shape: (batch_size, seq_len+padding_left+padding_right, embed_dim)
    # x = [x[:, :,i: i + seq_len].detach().cpu().numpy() for i in range(0, patch_size, dilation)]
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # x = torch.tensor(x).to(device)  # x.shape: (kernel_size, batch_size, seq_len, embed_dim)
    x = [x[:, :, i: i + seq_len] for i in range(0, patch_size, dilation)]
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.stack(x)  # x.shape: (kernel_size, batch_size, seq_len, embed_dim)

    # x = x.transpose(0, 1)#x.shape: (batch_size, kernel_size, seq_len, embed_dim)
    # x = x.transpose(1, 2)#x.shape: (batch_size, seq_len, kernel_size, embed_dim)
    x = x.permute(1, 2, 3, 0,4)  # x.shape: (batch_size, seq_len, kernel_size, embed_dim)
    return x
