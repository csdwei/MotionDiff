#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import math
import numpy as np



class VarianceSchedule(nn.Module):

    def __init__(self, num_steps, mode_beta='linear', beta_1=1e-4, beta_T=5e-2, cosine_s=8e-3):
        super().__init__()
        assert mode_beta in ('linear', 'cosine')
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode_beta

        if mode_beta == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode_beta == 'cosine':
            timesteps = (
            torch.arange(num_steps + 1) / num_steps + cosine_s
            )
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)


    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas



class Diffusion(nn.Module):
    def __init__(self, config, num_joint):
        super().__init__()
        self.num_steps = config.num_steps
        self.beta_1 = config.beta_1
        self.beta_T = config.beta_T
        self.num_joint = num_joint
        self.config = config
        self.var_sched = VarianceSchedule(num_steps=self.num_steps, mode_beta='linear', beta_1=self.beta_1,
                                          beta_T=self.beta_T, cosine_s=8e-3)


    def sample(self, net, encoded_x, flexibility=0.0, ret_traj=False):
        num_sample = encoded_x.shape[0]
        t_pred = self.config.pred_frames
        dim_each_frame = 3 * self.num_joint
        # start from standard Gaussian noise
        x_T = torch.randn([num_sample, t_pred, dim_each_frame]).cuda()
        traj = {self.var_sched.num_steps: x_T}

        for t in range(self.var_sched.num_steps, 0, -1):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]
            beta = self.var_sched.betas[[t] * num_sample]
            e_theta = net(encoded_x, x_t, beta)
            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t - 1] = x_next.detach()  # Stop gradient and save trajectory.
            # traj[t] = traj[t].cpu()  # Move previous output to CPU memory.
            # if not ret_traj:
            #     del traj[t]

        if ret_traj:
            return traj
        else:
            return traj[0]



    def forward(self, x_0, t=None):
        batch_size, _, _ = x_0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)

        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t].cuda()

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)        # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)    # (B, 1, 1)

        e_rand = torch.randn_like(x_0).cuda()
        x_T = c0 * x_0 + c1 * e_rand
        x_T = x_T.cuda()

        return x_T, e_rand, beta



















