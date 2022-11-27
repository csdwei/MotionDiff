#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
from einops.layers.torch import Rearrange
from models.PoseFormer import PoseFormer
import math
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = nn.Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret



class MotionDiff(nn.Module):
    def __init__(self, config, num_joint):
        super().__init__()
        self.dct_n = config.dct_n
        self.act = F.leaky_relu
        self.pose_embed_dim = config.pose_embed_dim
        self.rnn_output_dim = config.rnn_output_dim
        self.num_joint = num_joint
        self.concat1 = ConcatSquashLinear(self.pose_embed_dim * self.num_joint, self.rnn_output_dim, self.rnn_output_dim + 3)
        # concat
        self.concat2 = ConcatSquashLinear(self.rnn_output_dim, self.rnn_output_dim // 2, self.rnn_output_dim + 3)
        self.concat3 = ConcatSquashLinear(self.rnn_output_dim // 2, self.rnn_output_dim // 4, self.rnn_output_dim + 3)
        self.concat4 = ConcatSquashLinear(self.rnn_output_dim // 4, 3 * self.num_joint, self.rnn_output_dim + 3)
        # encoder
        self.poseformer = PoseFormer(config, num_joint=self.num_joint, in_chans=3, num_frame=config.pred_frames, embed_dim=config.pose_embed_dim,
                                       drop_rate=config.drop_rate_poseformer, drop_path_rate=config.drop_path_rate, norm_layer=None)
        # decoder
        self.pos_emb = PositionalEncoding(d_model=self.rnn_output_dim, dropout=0.1, max_len=200)
        self.layer = nn.TransformerEncoderLayer(d_model=self.rnn_output_dim, nhead=4, dim_feedforward=2*self.rnn_output_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=config.tf_layer)


    def forward(self, context, x, beta):
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)  # (B, 1, 1)
        context = context.view(batch_size, 1, -1)  # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)  # (B, 1, F+3)

        x = self.poseformer(x)
        x = self.concat1(ctx_emb, x)

        # Transformer Decoder
        final_emb = x.permute(1, 0, 2)
        final_emb = self.pos_emb(final_emb)
        x = self.transformer_encoder(final_emb).permute(1, 0, 2)

        # concat
        x = self.concat2(ctx_emb, x)
        x = self.act(x)
        x = self.concat3(ctx_emb, x)
        x = self.act(x)
        x = self.concat4(ctx_emb, x)

        return x




