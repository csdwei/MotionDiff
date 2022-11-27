#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
from timm.models.layers import DropPath
from functools import partial
from einops import rearrange
from models.rnn import RNN



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x





class PoseFormer(nn.Module):
    def __init__(self, config, num_joint=16, in_chans=3, num_frame=100, embed_dim=16, depth=4, num_heads=8, mlp_ratio=2.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        super(PoseFormer, self).__init__()

        # poseformer
        self.embed_dim = embed_dim
        self.num_joint = num_joint
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.joint_embedding_his = nn.Linear(in_chans, embed_dim)
        self.joint_embedding_pred = nn.Linear(in_chans, embed_dim)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joint, embed_dim))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim * self.num_joint))
        self.pos_drop = nn.Dropout(p=drop_rate)
        # rnn
        self.x_birnn = config.encoder_rnn
        self.rnn_type = config.rnn_type
        self.rnn_input_dim = num_joint * embed_dim
        self.rnn_output_dim = config.rnn_output_dim
        self.x_rnn = RNN(input_dim=self.rnn_input_dim, out_dim=self.rnn_output_dim, bi_dir=self.x_birnn, cell_type=self.rnn_type)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.Spatial_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim * num_joint, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim)
        self.Temporal_norm = norm_layer(embed_dim * self.num_joint)
        # self.weighted_mean = torch.nn.Conv1d(in_channels=num_joint, out_channels=1, kernel_size=1)


    def SpatialTrans(self, x, f):
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        for blk in self.Spatial_blocks:
            x = blk(x)

        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
        return x

    def r_encode(self, x):
        if self.x_birnn:
            h_x = self.x_rnn(x).mean(dim=0)
        else:
            h_x = self.x_rnn(x)[-1]
        return h_x


    def encode_his(self, x):
        b, f, p, c = x.shape       ##### b is batch size, f is number of frames, p is number of joints, c is dimension of each joint
        x = x.permute(0, 3, 1, 2)
        x = rearrange(x, 'b c f p  -> (b f) p  c', )
        x = self.joint_embedding_his(x)
        x = self.SpatialTrans(x, f)
        # ####### A easy way to implement weighted mean  (batch, joints, d) --> (batch, 1, d)
        # x = self.weighted_mean(x)
        # x = x.view(-1, self.embed_dim)
        ######## rnn to obtain the first predicted frame
        x = x.permute(1, 0, 2)
        x = self.r_encode(x)    # (b, d)
        return x


    def TemporalAttention(self, x):
        b = x.shape[0]
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.Temporal_norm(x)
        return x


    def forward(self, y):
        x = y.reshape(y.shape[0], y.shape[1], self.num_joint, -1)
        ##### b is batch size, f is number of frames, p is number of joints, c is dimension of each joint
        x = x.permute(0, 3, 1, 2)
        b, _, f, p = x.shape
        x = rearrange(x, 'b c f p  -> (b f) p  c', )
        x = self.joint_embedding_pred(x)
        x = self.SpatialTrans(x, f)
        x = self.TemporalAttention(x)
        return x




