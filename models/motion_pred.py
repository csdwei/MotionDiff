import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from utils.torch import *
from models.PoseFormer import PoseFormer
from models.MotionDiff import MotionDiff
from models.Diffusion import Diffusion
import models.mao_gcn as nnmodel


class DiffHM(nn.Module):
    def __init__(self, config, num_joint):
        super(DiffHM, self).__init__()
        # Variable
        self.config = config
        self.num_frame = config.obs_frames
        self.pose_embed_dim = config.pose_embed_dim
        self.drop_path_rate = config.drop_path_rate
        self.drop_rate_poseformer = config.drop_rate_poseformer
        self.num_joint = num_joint
        # Encoder
        self.poseformer = PoseFormer(config, num_joint=self.num_joint, in_chans=3, num_frame=self.num_frame, embed_dim=self.pose_embed_dim,
                                       drop_rate=self.drop_rate_poseformer, drop_path_rate=self.drop_path_rate, norm_layer=None)
        # Decoder
        self.y_diff = Diffusion(config, num_joint)
        self.y_mlp = MotionDiff(config, num_joint=self.num_joint)


    def diff(self, y):
        b, f, _, _ = y.shape
        y = y.reshape(b, f, -1)
        return self.y_diff(y)

    def encode(self, x, y):
        feat_x_encoded = self.poseformer.encode_his(x)
        diff_y, e_rand, beta = self.diff(y)
        # whether DCT ???

        return feat_x_encoded, diff_y, e_rand, beta

    def denoise(self, feat_x_encoded, diff_y, beta):
        return self.y_mlp(feat_x_encoded, diff_y, beta)

    def get_e_loss(self, e_rand, e_theta):
        loss = F.mse_loss(e_theta.view(-1, 3 * self.num_joint), e_rand.view(-1, 3 * self.num_joint), reduction='mean')
        return loss


    def get_loss(self, x, y):
        feat_x_encoded, diff_y, e_rand, beta = self.encode(x, y)
        e_theta = self.denoise(feat_x_encoded, diff_y, beta)
        loss = self.get_e_loss(e_rand, e_theta)
        return loss



    def generate(self, x):
        x = x.reshape(x.shape[0], x.shape[1], self.num_joint, -1)
        x = x.permute(1, 0, 2, 3).contiguous()
        encoded_x = self.poseformer.encode_his(x)
        predicted_x = self.y_diff.sample(self.y_mlp, encoded_x, flexibility=self.config.flexibility, ret_traj=self.config.ret_traj)
        predicted_x = predicted_x.permute(1, 0, 2).contiguous()
        return predicted_x



def get_diff_model(config, traj_dim):
    model_name = config.model_name
    num_joint = traj_dim // 3
    if model_name == "MotionDiff":
        return DiffHM(config, num_joint)
    else:
        print("The model doesn't exist: %s" % model_name)
        exit(0)



def get_refine_model(config, traj_dim):
    return nnmodel.GCN(input_feature=config.dct_n, hidden_feature=config.gcn_linear_size, p_dropout=config.gcn_dropout,
                        num_stage=config.gcn_layers, node_n=traj_dim, gamma=config.gamma)









