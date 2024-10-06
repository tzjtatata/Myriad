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
# from timm.models.layers import trunc_normal_


class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ResidualBlock, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias = False)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias = False)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out


def get_conv_act(act,dims_in,dims_out,kernel,strde,padding,bias):
    conv = []
    conv.append(nn.Conv2d(dims_in,dims_out,kernel_size=kernel,stride=stride,padding=padding,bias=bias))
    conv.append(nn.InstanceNorm2d())
    conv.append(act)
    return nn.Sequential(*conv)


class AttentionAdaptor(nn.Module):
    def __init__(self,dims, input_dim=1):
        super(AttentionAdaptor, self).__init__()
        self.dims = dims
        self.conv1 = nn.Conv2d(dims,dims,3,1,1,bias=False)
        self.conv2 = nn.Conv2d(dims,dims*1,3,1,1,bias=False)
        self.input_dim = input_dim

        # self.cls_linear1 = nn.Conv2d(dims*1,dims*1,1,1,bias=False)
        # self.cls_linear2 = nn.Conv2d(dims*1,dims*1,1,1,bias=False)
    def forward(self,x):
        b = x.shape[0]
        x_tok = x[:,0:1,:]#.permute([0,2,1])
        x_conv = x[:,1:,:].permute([0,2,1])
        # ref_conv = ref[:,1:,:].permute([0,2,1])
        convs = self.conv1(x_conv.reshape([b,1408,16,16]))
        convs = self.conv2(convs)
        convs = convs + x_conv.reshape([b,1408,16,16])
        convs = convs.view([b,1408,256])
        convs = convs.permute([0,2,1])

        # cls_convs = self.cls_linear1(x_tok.reshape([b,1408,1,1]))
        # cls_convs = self.cls_linear2(cls_convs)
        # cls_convs = cls_convs + x_tok.reshape([b,1408,1,1])
        # cls_convs = cls_convs.view([b,1408,1])
        # cls_convs = cls_convs.permute([0,2,1])

        return torch.cat([x_tok,convs],1)


class LoraAdaptorV2(nn.Module):
    def __init__(self,dims, input_dim=1, out_dim=-1):
        super(LoraAdaptorV2, self).__init__()
        self.dims = dims
        self.out_dim = out_dim if out_dim != -1 else dims
        self.conv1 = nn.Linear(dims, input_dim,bias=False)
        self.conv2 = nn.Linear(input_dim, self.out_dim,bias=False)
        nn.init.normal_(self.conv1.weight, std=0.02)
        nn.init.normal_(self.conv2.weight, std=0.02)

    def forward(self,x):
        b = x.shape[0]

        convs = self.conv1(x)
        convs = self.conv2(convs)
        if convs.shape[-1] != x.shape[-1]:
            new_convs = torch.zeros_like(convs)
            new_convs[:, :, :x.shape[-1]] = x
            convs = convs + new_convs
        else:
            convs = convs + x

        return convs

class VEInstructorV2(nn.Module):
    def __init__(self, dim_in=1, version=0) -> None:
        super().__init__()
        meta_net = [
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
        ]
        if version == 0:
            meta_net.append(
                nn.Conv2d(dim_in * 1024, 768, kernel_size=1, padding=0)
            )
            self.dim = 49
        elif version == 1:
            meta_net += [
                nn.Conv2d(dim_in * 1024, dim_in * 1024, kernel_size=3, padding=0), 
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_in * 1024, 768, kernel_size=1, padding=0), 
            ]
            self.dim = 25
        elif version == 2:
            meta_net += [
                nn.Conv2d(dim_in * 1024, dim_in * 1024, kernel_size=3, padding=1), 
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2), # 3 * 3
                nn.Conv2d(dim_in * 1024, 768, kernel_size=1, padding=0), 
            ]
            self.dim = 9
        else:
            raise NotImplementError(f"Not Implement VEInstructorV2 with v{version}.")
        
        self.meta_net = nn.Sequential(*meta_net)

    def forward(self, input):
        B,C,H,W = input.shape
        img_prompts = self.meta_net(input)
        img_prompts = img_prompts.reshape(B,768,self.dim).transpose(-2,-1)
        return img_prompts


class VETokenizer(nn.Module):
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

            nn.Conv2d(dim_in * 1024, 4096, kernel_size=5, padding=0),
            # nn.BatchNorm2d(5120),
            # nn.ReLU(inplace=True),
        )
        self.base_prompts = nn.Parameter(torch.randn((9, 4096)),requires_grad=True)

    def forward(self, input):
        B,C,H,W = input.shape
        img_prompts = self.meta_net(input)
        # print(input.shape, img_prompts.shape)
        img_prompts = img_prompts.reshape(B,4096,9).transpose(-2,-1)
        output = torch.cat([self.base_prompts.expand(B,-1,-1), img_prompts], dim=1)
        return output

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

class SimpleViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        ) 

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        # x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)
class VETokenizerfea(nn.Module):
    def __init__(self, dim_in=1) -> None:
        super().__init__()
        self.meta_net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in * 4, kernel_size=3, stride=2, padding=0),
            # nn.InstanceNorm2d(dim_in * 4),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2), # 112 * 112

            nn.Conv2d(dim_in * 4, dim_in * 16, kernel_size=1, stride=1, padding=0),
            # nn.InstanceNorm2d(dim_in * 16),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2), # 56 * 56

            nn.Conv2d(dim_in * 16, dim_in * 64, kernel_size=1, stride=1, padding=0),
            # nn.InstanceNorm2d(dim_in * 64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2), # 28 * 28/

            nn.Conv2d(dim_in * 64, dim_in * 256, kernel_size=3,stride=2,  padding=0),
            # nn.InstanceNorm2d(dim_in * 256),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2), # 14 * 14

            nn.Conv2d(dim_in * 256, dim_in * 1024, kernel_size=1,stride=1,  padding=0),
            # nn.InstanceNorm2d(dim_in * 1024),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2), # 7 * 7

            nn.Conv2d(dim_in * 1024, 4096, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(5120),
            # nn.ReLU(inplace=True),
        )
        self.base_prompts = nn.Parameter(torch.randn((9, 4096)),requires_grad=True)

    def forward(self, input):
        B,C,H,W = input.shape
        # print(input.shape)
        img_prompts = self.meta_net(input)
        # print('tokenizer',input.shape, img_prompts.shape)
        img_prompts = img_prompts.reshape(B,4096,9).transpose(-2,-1)
        output = torch.cat([self.base_prompts.expand(B,-1,-1), img_prompts], dim=1)
        return output