import contextlib
import logging
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_


def get_conv_act(act,dims_in,dims_out,kernel,strde,padding,bias):
    conv = []
    conv.append(nn.Conv2d(dims_in,dims_out,kernel_size=kernel,stride=stride,padding=padding,bias=bias))
    conv.append(nn.InstanceNorm2d())
    conv.append(act)
    return nn.Sequential(*conv)

# class Fuser(nn.Module):
#     def __init__(self,dims):
#         super(Fuser, self).__init__()
#         self.dims = dims
#         self.conv1 = nn.Conv2d(dims*2,dims*2,3,1,1,bias=False)
#         self.conv2 = nn.Conv2d(dims*2,dims*1,3,1,1,bias=False)

#         self.linear1 = nn.Conv2d(dims*2,dims*2,1,1,bias=False)
#         self.linear2 = nn.Conv2d(dims*2,dims*1,1,1,bias=False)
#     def forward(self,x,ref,mask):
#         b = x.shape[0]
#         x_tok = x[:,0:1,:].permute([0,2,1])
#         ref_tok = ref[:,0:1,:].permute([0,2,1])

#         x_conv = x[:,1:,:].permute([0,2,1])
#         ref_conv = ref[:,1:,:].permute([0,2,1])


#         convs = self.conv1(torch.cat([x_conv.reshape([b,1408,16,16]),ref_conv.reshape([b,1408,16,16])],1))
#         convs = self.conv2(convs).view([b,1408,256])
#         convs = convs.permute([0,2,1])

#         linears = self.linear1(torch.cat([x_tok.reshape([b,1408,1,1]),ref_tok.reshape([b,1408,1,1])],1))
#         linears = self.linear2(linears).view([b,1408,1])
#         linears = linears.permute([0,2,1])



#         return torch.cat([linears,convs],1)
class Fuser(nn.Module):
    def __init__(self,dims):
        super(Fuser, self).__init__()
        self.dims = dims
        self.conv1 = nn.Conv2d(dims+dims,dims+dims,3,1,1,bias=False)
        self.conv2 = nn.Conv2d(dims+dims,dims*1,3,1,1,bias=False)

        self.linear1 = nn.Conv2d(dims*2,dims*2,1,1,bias=False)
        self.linear2 = nn.Conv2d(dims*2,dims*1,1,1,bias=False)
    def forward(self,x,ref,mask):
        b = x.shape[0]
        x_tok = x[:,0:1,:].permute([0,2,1])
        ref_tok = ref[:,0:1,:].permute([0,2,1])

        x_conv = x[:,1:,:].permute([0,2,1])
        ref_conv = ref[:,1:,:].permute([0,2,1])

        # mask_ = mask.view([b,336,16,16])

        convs = self.conv1(torch.cat([x_conv.reshape([b,1408,16,16]),ref_conv.view([b,1408,16,16])],1))
        convs = self.conv2(convs).view([b,1408,256])
        convs = convs.permute([0,2,1])

        linears = self.linear1(torch.cat([x_tok.reshape([b,1408,1,1]),ref_tok.reshape([b,1408,1,1])],1))
        linears = self.linear2(linears).view([b,1408,1])
        # linears = linears.reshape([b,1408,1,1])
        linears = linears.permute([0,2,1])
        return torch.cat([linears,convs],1)


class MaskFuser(nn.Module):
    def __init__(self,dims):
        super(MaskFuser, self).__init__()
        self.dims = dims
        self.conv1 = nn.Conv2d(dims+64,dims+64,3,1,1,bias=False)
        self.conv2 = nn.Conv2d(dims+64,dims*1,3,1,1,bias=False)

        # self.linear1 = nn.Conv2d(dims*2,dims*2,1,1,bias=False)
        # self.linear2 = nn.Conv2d(dims*2,dims*1,1,1,bias=False)
    def forward(self,x,ref,mask):
        b = x.shape[0]
        x_tok = x[:,0:1,:]#.permute([0,2,1])
        ref_tok = ref[:,0:1,:].permute([0,2,1])

        x_conv = x[:,1:,:].permute([0,2,1])
        ref_conv = ref[:,1:,:].permute([0,2,1])

        # mask_ = mask.view([b,336,16,16])

        convs = self.conv1(torch.cat([x_conv.reshape([b,1408,16,16]),mask.view([b,1,16,16])],1))
        convs = self.conv2(convs).view([b,1408,256])
        convs = convs.permute([0,2,1])

        # linears = self.linear1(torch.cat([,ref_tok.reshape([b,1408,1,1])],1))
        # linears = self.linear2(linears).view([b,1408,1])
        # linears = x_tok.reshape([b,1408,1,1])
        # linears = linears.permute([0,2,1])



        return torch.cat([x_tok,convs],1)


class ROIFuser(nn.Module):
    def __init__(self,dims):
        super(ROIFuser, self).__init__()
        self.dims = dims
        self.conv1 = nn.Conv2d(dims*1,dims*1,3,1,1,bias=False)
        self.conv2 = nn.Conv2d(dims*1,dims*1,3,1,1,bias=False)

    def forward(self,x):


        convs = self.conv1(x)
        convs = self.conv2(convs)


        return convs
        
class MaskAdapter(nn.Module):
    def __init__(self,dims, input_dim=1):
        super(MaskAdapter, self).__init__()
        self.dims = dims
        self.conv1 = nn.Conv2d(dims+input_dim,dims+input_dim,3,1,1,bias=False)
        self.conv2 = nn.Conv2d(dims+input_dim,dims*1,3,1,1,bias=False)
        self.input_dim = input_dim

        # self.linear1 = nn.Conv2d(dims*2,dims*2,1,1,bias=False)
        # self.linear2 = nn.Conv2d(dims*2,dims*1,1,1,bias=False)
    def forward(self,x,mask):
        b = x.shape[0]
        x_tok = x[:,0:1,:]#.permute([0,2,1])
        # ref_tok = ref[:,0:1,:].permute([0,2,1])

        x_conv = x[:,1:,:].permute([0,2,1])
        # ref_conv = ref[:,1:,:].permute([0,2,1])

        # mask_ = mask.view([b,336,16,16])

        convs = self.conv1(torch.cat([x_conv.reshape([b,1408,16,16]),mask.view([b,self.input_dim,16,16])],1))
        convs = self.conv2(convs)
        convs = convs + x_conv.reshape([b,1408,16,16])
        convs = convs.view([b,1408,256])
        convs = convs.permute([0,2,1])

        # linears = self.linear1(torch.cat([,ref_tok.reshape([b,1408,1,1])],1))
        # linears = self.linear2(linears).view([b,1408,1])
        # linears = x_tok.reshape([b,1408,1,1])
        # linears = linears.permute([0,2,1])

        return torch.cat([x_tok,convs],1)

class AttentionAdaptor(nn.Module):
    def __init__(self,dims, input_dim=1):
        super(AttentionAdaptor, self).__init__()
        self.dims = dims
        self.conv1 = nn.Conv2d(dims,dims,3,1,1,bias=False)
        self.conv2 = nn.Conv2d(dims,dims*1,3,1,1,bias=False)
        self.input_dim = input_dim

        # self.linear1 = nn.Conv2d(dims*2,dims*2,1,1,bias=False)
        # self.linear2 = nn.Conv2d(dims*2,dims*1,1,1,bias=False)
    def forward(self,x):
        b = x.shape[0]
        x_tok = x[:,0:1,:]#.permute([0,2,1])
        # ref_tok = ref[:,0:1,:].permute([0,2,1])

        x_conv = x[:,1:,:].permute([0,2,1])
        # ref_conv = ref[:,1:,:].permute([0,2,1])
        convs = self.conv1(x_conv.reshape([b,1408,16,16]))
        convs = self.conv2(convs)
        convs = convs + x_conv.reshape([b,1408,16,16])
        convs = convs.view([b,1408,256])
        convs = convs.permute([0,2,1])



        return torch.cat([x_tok,convs],1)
class FeatAdaptor(nn.Module):
    def __init__(self,dims):
        super(FeatAdaptor, self).__init__()
        self.dims = dims
        self.conv1 = nn.Conv2d(dims,dims,3,1,1,bias=False)
        self.conv2 = nn.Conv2d(dims,dims,3,1,1,bias=False)

        # self.linear1 = nn.Conv2d(dims*2,dims*2,1,1,bias=False)
        # self.linear2 = nn.Conv2d(dims*2,dims*1,1,1,bias=False)
    def forward(self,x):
        b = x.shape[0]
        x_tok = x[:,0:1,:]#.permute([0,2,1])
        # ref_tok = ref[:,0:1,:].permute([0,2,1])

        x_conv = x[:,1:,:].permute([0,2,1])
        # ref_conv = ref[:,1:,:].permute([0,2,1])

        # mask_ = mask.view([b,336,16,16])

        convs = self.conv1(x_conv.reshape([b,1408,16,16]))
        convs = self.conv2(convs)
        convs = convs + x_conv.reshape([b,1408,16,16])
        convs = convs.view([b,1408,256])
        convs = convs.permute([0,2,1])

        # linears = self.linear1(torch.cat([,ref_tok.reshape([b,1408,1,1])],1))
        # linears = self.linear2(linears).view([b,1408,1])
        # linears = x_tok.reshape([b,1408,1,1])
        # linears = linears.permute([0,2,1])

        return torch.cat([x_tok,convs],1)


class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k):
        super(LinearLayer, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for i in k])

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                tokens[i] = self.fc[i](tokens[i][:, 1:, :])
            else:
                B, C, H, W = tokens[i].shape
                tokens[i] = self.fc[i](tokens[i].view(B, C, -1).permute(0, 2, 1).contiguous())
        return tokens

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class Transformer_blocks(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, dim_head = 64):
        super().__init__()

        # patch_dim = channels * patch_height * patch_width


        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

    def forward(self, img):
        device = img.device

        x = self.transformer(img)

        return x


# Adapted from https://github.com/CASIA-IVA-Lab/AnomalyGPT/blob/main/code/model/AnomalyGPT_models.py
class PromptLearner(nn.Module):
    def __init__(self, dim_in, dim_out, add_pos_embedding=False) -> None:
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.meta_net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in * 4, kernel_size=3, padding=1),
            # nn.BatchNorm2d(dim_in * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 112 * 112

            nn.Conv2d(dim_in * 4, dim_in * 16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(dim_in * 16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 56 * 56

            nn.Conv2d(dim_in * 16, dim_in * 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(dim_in * 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 28 * 28

            nn.Conv2d(dim_in * 64, dim_in * 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(dim_in * 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 14 * 14

            nn.Conv2d(dim_in * 256, dim_in * 1024, kernel_size=3, padding=1),
            # nn.BatchNorm2d(dim_in * 1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 7 * 7

            nn.Conv2d(dim_in * 1024, dim_out, kernel_size=5, padding=0),
            # nn.BatchNorm2d(dim_out),
            # nn.ReLU(inplace=True),
        )
        # self.base_prompts = nn.Parameter(torch.randn((9, dim_out)),requires_grad=True)
        self.add_pos_embedding = add_pos_embedding
        if self.add_pos_embedding:
            self.pos_embedding = nn.Parameter(torch.randn(1, 9, dim_out))
            trunc_normal_(self.pos_embedding, std=.02)
    
    def forward(self, input):
        bs = input.shape[0]
        ret = self.meta_net(input).reshape(bs, self.dim_out, -1).transpose(1, 2)  # (bs, 5120, 3, 3) -> (bs, 9, 5120)
        if self.add_pos_embedding:
            ret = ret + self.pos_embedding.expand(ret.shape[0], -1, -1)
        # ret = torch.cat([self.base_prompts.expand(bs, -1, -1), ret], dim=1)
        return ret

class veprompt(nn.Module):
    def __init__(self, dim_in=1) -> None:
        super().__init__()
        self.meta_net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in * 4, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(dim_in * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 112 * 112

            nn.Conv2d(dim_in * 4, dim_in * 16, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(dim_in * 16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 56 * 56

            nn.Conv2d(dim_in * 16, dim_in * 64, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(dim_in * 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 28 * 28

            nn.Conv2d(dim_in * 64, dim_in * 256, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(dim_in * 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 14 * 14

            nn.Conv2d(dim_in * 256, dim_in * 1024, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(dim_in * 1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 7 * 7

            nn.Conv2d(dim_in * 1024, 768, kernel_size=5, padding=0),
            # nn.BatchNorm2d(5120),
            # nn.ReLU(inplace=True),
        )
        self.base_prompts = nn.Parameter(torch.randn((9, 768)),requires_grad=True)

    def forward(self, input):
        B,C,H,W = input.shape
        img_prompts = self.meta_net(input)
        # print(input.shape, img_prompts.shape)
        img_prompts = img_prompts.reshape(B,768,9).transpose(-2,-1)
        output = torch.cat([self.base_prompts.expand(B,-1,-1), img_prompts], dim=1)
        return output
class visualqueryoneshot(nn.Module):
    def __init__(self, dim_in=1) -> None:
        super().__init__()
        self.meta_net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 112 * 112

            nn.Conv2d(dim_in * 4, dim_in * 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 56 * 56

            nn.Conv2d(dim_in * 16, dim_in * 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 28 * 28

            nn.Conv2d(dim_in * 64, dim_in * 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 14 * 14

            nn.Conv2d(dim_in * 256, dim_in * 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 7 * 7

            nn.Conv2d(dim_in * 1024, 768, kernel_size=3, padding=0),

        )
        self.base_prompts = nn.Parameter(torch.randn((7, 768)),requires_grad=True)

    def forward(self, input):
        B,C,H,W = input.shape
        img_prompts = self.meta_net(input)
        # print(img_prompts.shape)
        img_prompts = img_prompts.reshape(B,768,25).transpose(-2,-1)
        output = torch.cat([self.base_prompts.expand(B,-1,-1), img_prompts], dim=1)

        return output

class visualquery(nn.Module):
    def __init__(self, dim_in=1) -> None:
        super().__init__()
        self.meta_net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 112 * 112

            nn.Conv2d(dim_in * 4, dim_in * 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 56 * 56

            nn.Conv2d(dim_in * 16, dim_in * 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 28 * 28

            nn.Conv2d(dim_in * 64, dim_in * 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 14 * 14

            nn.Conv2d(dim_in * 256, dim_in * 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 7 * 7

            nn.Conv2d(dim_in * 1024, 768, kernel_size=5, padding=0),

        )
        self.base_prompts = nn.Parameter(torch.randn((9, 768)),requires_grad=True)

    def forward(self, input):
        B,C,H,W = input.shape
        img_prompts = self.meta_net(input)
        img_prompts = img_prompts.reshape(B,768,9).transpose(-2,-1)
        output = torch.cat([self.base_prompts.expand(B,-1,-1), img_prompts], dim=1)

        return output

class visualqueryv2(nn.Module):
    def __init__(self, dim_in=1) -> None:
        super().__init__()
        self.meta_net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 112 * 112

            nn.Conv2d(dim_in * 4, dim_in * 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 56 * 56

            nn.Conv2d(dim_in * 16, dim_in * 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 28 * 28

            nn.Conv2d(dim_in * 64, dim_in * 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 14 * 14

            nn.Conv2d(dim_in * 256, dim_in * 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 7 * 7

            nn.Conv2d(dim_in * 1024, 768, kernel_size=1, padding=0),

        )
        self.base_prompts = nn.Parameter(torch.randn((9, 768)),requires_grad=True)

    def forward(self, input):
        B,C,H,W = input.shape
        img_prompts = self.meta_net(input)
        img_prompts = img_prompts.reshape(B,768,49).transpose(-2,-1)
        output = torch.cat([self.base_prompts.expand(B,-1,-1), img_prompts], dim=1)

        return output



class PromptLearnerv2(nn.Module):
    def __init__(self, dim_in=1) -> None:
        super().__init__()
        self.meta_net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in * 4, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(dim_in * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 112 * 112

            nn.Conv2d(dim_in * 4, dim_in * 16, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(dim_in * 16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 56 * 56

            nn.Conv2d(dim_in * 16, dim_in * 64, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(dim_in * 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 28 * 28

            nn.Conv2d(dim_in * 64, dim_in * 256, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(dim_in * 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 14 * 14

            nn.Conv2d(dim_in * 256, dim_in * 1024, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(dim_in * 1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 7 * 7

            nn.Conv2d(dim_in * 1024, 5120, kernel_size=5, padding=0),
            # nn.BatchNorm2d(5120),
            # nn.ReLU(inplace=True),
        )
        self.base_prompts = nn.Parameter(torch.randn((9, 5120)),requires_grad=True)

    def forward(self, input):
        B,C,H,W = input.shape
        img_prompts = self.meta_net(input)
        # print(input.shape, img_prompts.shape)
        img_prompts = img_prompts.reshape(B,5120,-1).transpose(-2,-1)
        output = torch.cat([self.base_prompts.expand(B,-1,-1), img_prompts], dim=1)
        return output
class PromptLearnerv3(nn.Module):
    def __init__(self, dim_in=1) -> None:
        super().__init__()
        self.meta_net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in * 4, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(dim_in * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 112 * 112

            nn.Conv2d(dim_in * 4, dim_in * 16, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(dim_in * 16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 56 * 56

            nn.Conv2d(dim_in * 16, dim_in * 64, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(dim_in * 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 28 * 28

            nn.Conv2d(dim_in * 64, dim_in * 256, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(dim_in * 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 14 * 14

            nn.Conv2d(dim_in * 256, dim_in * 1024, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(dim_in * 1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 7 * 7

            nn.Conv2d(dim_in * 1024, 5120, kernel_size=5, padding=0),
            # nn.BatchNorm2d(5120),
            # nn.ReLU(inplace=True),
        )
        # self.base_prompts = nn.Parameter(torch.randn((9, 5120)),requires_grad=True)

    def forward(self, input):
        B,C,H,W = input.shape
        img_prompts = self.meta_net(input)
        # print(input.shape, img_prompts.shape)
        img_prompts = img_prompts.reshape(B,5120,9).transpose(-2,-1)
        # output = torch.cat([self.base_prompts.expand(B,-1,-1), img_prompts], dim=1)
        return img_prompts

class PromptLearneroneshot(nn.Module):
    def __init__(self, dim_in=1) -> None:
        super().__init__()
        self.meta_net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in * 4, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(dim_in * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 112 * 112

            nn.Conv2d(dim_in * 4, dim_in * 16, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(dim_in * 16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 56 * 56

            nn.Conv2d(dim_in * 16, dim_in * 64, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(dim_in * 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 28 * 28

            nn.Conv2d(dim_in * 64, dim_in * 256, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(dim_in * 256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 14 * 14

            nn.Conv2d(dim_in * 256, dim_in * 1024, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(dim_in * 1024),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 7 * 7

            nn.Conv2d(dim_in * 1024, 5120, kernel_size=5, padding=0),
            # nn.BatchNorm2d(5120),
            # nn.ReLU(inplace=True),
        )
        # self.base_prompts = nn.Parameter(torch.randn((9, 5120)),requires_grad=True)

    def forward(self, input):
        B,C,H,W = input.shape
        img_prompts = self.meta_net(input)
        # print(input.shape, img_prompts.shape)
        img_prompts = img_prompts.reshape(B,5120,9).transpose(-2,-1)
        return img_prompts
# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MaskTokenizer(nn.Module):
    def __init__(self, dim_in, dim_out, pos_type='append', activation=nn.GELU) -> None:
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.pos_type = pos_type
        self.meta_net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in * 4, kernel_size=2, stride=2, padding=1),
            LayerNorm2d(dim_in * 4),
            activation(),

            nn.Conv2d(dim_in * 4, dim_in * 16, kernel_size=2, stride=2, padding=1),
            LayerNorm2d(dim_in * 16),
            activation(),

            nn.Conv2d(dim_in * 16, dim_in * 64, kernel_size=2, stride=2, padding=1),
            LayerNorm2d(dim_in * 64),
            activation(),

            nn.Conv2d(dim_in * 64, dim_in * 256, kernel_size=2, stride=2, padding=1),
            LayerNorm2d(dim_in * 256),
            activation(),

            nn.Conv2d(dim_in * 256, dim_in * 1024, kernel_size=2, stride=2, padding=1),
            LayerNorm2d(dim_in * 1024),
            activation(),# nn.BatchNorm2d(dim_in * 1024),

            nn.Conv2d(dim_in * 1024, dim_out, kernel_size=5, padding=0),
            # nn.BatchNorm2d(dim_out),
            # nn.ReLU(inplace=True),
        )

        num_embeds = 9 if pos_type == 'append' else 16
        self.pos_embedding = nn.Parameter(torch.randn(1, num_embeds, dim_out))
        trunc_normal_(self.pos_embedding, std=.02)
    
    def forward(self, input):
        bs = input.shape[0]
        ret = self.meta_net(input).reshape(bs, self.dim_out, -1).transpose(1, 2)  # (bs, 5120, 3, 3) -> (bs, 9, 5120)
        if self.pos_type == 'append':
            ret = torch.cat([self.pos_embedding.expand(ret.shape[0], -1, -1), ret], dim=1)
        elif self.pos_type == 'add':
            # print(self.pos_embedding.shape, ret.shape)
            ret = self.pos_embedding.expand(ret.shape[0], -1, -1) + ret
        else:
            raise NotImplementedError(f"Not Implement position embedding type: {self.pos_type}")
        return ret
