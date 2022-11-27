import os
import argparse
import torch
import pdb
import numpy as np
import os.path as osp
import logging
import time
from torch import nn, optim, utils
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import pickle
from progress.bar import Bar
from torch.autograd import Variable

from utils.logger import create_logger
from dataset.utils.dataset_h36m import DatasetH36M
from models.motion_pred import *
from models.common import *
from dataset.utils.dataset_generated_motions import CustomH36M
from torch.utils.data import DataLoader
from tqdm import tqdm
import dataset.utils as dutil



class DiffHM():
    def __init__(self, config):
        self.config = config
        torch.backends.cudnn.benchmark = True
        self.device = torch.device('cuda', index=config.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
        self.dtype = torch.float32
        self._build()


    def train_diff(self):
        model = self.model
        config = self.config
        t_his = config.obs_frames
        t_pred = config.pred_frames
        model.train()
        print('==========================')
        for epoch in range(config.iter_start_diff, config.num_diff_epoch):
            t_s = time.time()
            train_losses = 0
            total_num_sample = 0
            loss_names = ['MSE']
            generator = self.dataset_train.sampling_generator(num_samples=config.num_diff_data_sample, batch_size=config.batch_size)

            for traj_np in generator:
                traj_np = traj_np[..., 1:, :]
                traj = tensor(traj_np, device=self.device, dtype=self.dtype)
                X = traj[:, :t_his, :, :]
                Y = traj[:, t_his:, :, :]
                loss = model.get_loss(X, Y)
                self.optimizer.zero_grad()
                loss.backward()
                # gradient clipped
                if config.max_norm:
                    nn.utils.clip_grad_norm(self.model.parameters(), max_norm=1)
                self.optimizer.step()
                train_losses += loss
                total_num_sample += 1

            self.scheduler.step()
            dt = time.time() - t_s
            train_losses /= total_num_sample
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.info('====> Epoch: {} Time: {:.2f} MSE: {} lr: {:.5f}'.format(epoch+1, dt, train_losses, lr))
            self.log_writer_diff.add_scalar('DiffMotion_' + str(loss_names), train_losses, epoch)


            ############ Saving model ###############
            if config.save_model_interval > 0 and (epoch + 1) % config.save_model_interval == 0:
                with to_cpu(model):
                    cp_path = self.pretrained_model_dir_diff % (epoch + 1)
                    model_cp = {'model_dict': model.state_dict(), 'meta': {'std': self.dataset_train.std, 'mean': self.dataset_train.mean}}
                    pickle.dump(model_cp, open(cp_path, 'wb'))



    def generate_diff(self):
        device = self.device
        dtype = self.dtype
        config = self.config
        t_his = config.obs_frames
        torch.set_grad_enabled(False)
        logger_test = create_logger(os.path.join(self.model_dir_log, "log_eval.txt"))

        # get dataset
        dataset = self.dataset_train

        # get models
        algos = ['diff']
        models = {}
        for algo in algos:
            models[algo] = get_diff_model(config, self.dataset_train.traj_dim)
            cp_path = self.pretrained_model_dir_diff % config.eval_at_diff
            print('loading diffusion model from checkpoint: %s' % cp_path)
            diff_cp = pickle.load(open(cp_path, "rb"))
            models[algo].load_state_dict(diff_cp['model_dict'])
            models[algo].to(device)
            models[algo].eval()

        # normalize
        if config.normalize_data:
            dataset.normalize_data(diff_cp['meta']['mean'], diff_cp['meta']['std'])

        # generate 50 diversity training samples
        data_gen = dataset.sampling_generator(num_samples=config.num_generate_diff_data_sample, batch_size=config.generate_diff_batch_size)
        num_seeds = config.num_seeds

        data_diff = []
        data_gt = []
        count = 0

        for traj_np in data_gen:
            traj_gt = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1)
            traj_gt = torch.squeeze(tensor(traj_gt, dtype=self.dtype))
            data_gt.append(traj_gt)

            traj_np = tensor(traj_np, device=self.device, dtype=self.dtype)
            pred = get_prediction(config, models, traj_np, algo="diff", sample_num=config.nk, device=device, dtype=dtype, num_seeds=num_seeds, concat_hist=True)
            data_diff.append(torch.squeeze(torch.tensor(pred)))

            count += 1
            if count % 500 == 0:
                print(count)


        data_diff = torch.stack(data_diff)
        data_gt = torch.stack(data_gt)
        generated_diff = osp.join(self.generated_motions, "generate_diversity_diff.pth")
        generated_gt = osp.join(self.generated_motions, "generate_diversity_gt.pth")
        torch.save(data_diff.to(torch.device('cpu')), generated_diff)
        torch.save(data_gt.to(torch.device('cpu')), generated_gt)



    def train_refine(self):
        device = self.device
        dtype = self.dtype
        config = self.config
        logger_refine = create_logger(os.path.join(self.model_dir_log, "log_refine.txt"))

        """data"""
        t_his = config.obs_frames
        t_pred = config.pred_frames
        nk = config.nk

        # generated dataset and dataloader
        dataset = self.dataset_custom
        dataloader = self.dataloader_custom

        """model"""
        optimizer = self.optimizer
        refine = self.refine
        lr_now = config.refine_lr

        for epoch in range(config.iter_start_refine, config.num_refine_epoch):

            if (epoch + 1) % config.gcn_lr_decay == 0:
                lr_now = lr_decay(optimizer, lr_now, config.gcn_lr_gamma)

            t_l = AccumLoss()
            refine.train()

            i = 0
            st = time.time()
            # bar = Bar('>>>', fill='>', max=len(dataloader))
            train_losses = 0
            total_num_sample = 0
            loss_names = ['TOTAL', 'RECON', 'R1', 'JL']

            for (inputs, targets, all_seqs) in tqdm(dataloader):

                b, f, c = targets.shape
                # batch_size = inputs.shape[0]
                bt = time.time()
                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda()).float()
                    targets = Variable(targets.cuda()).float()
                    all_seqs = Variable(all_seqs.cuda()).float()

                X = inputs.reshape(-1, config.dct_n, c)
                X = X.permute(0, 2, 1).contiguous()
                outputs = refine(X)

                # IDCT
                _, idct_m = dutil.dataset_generated_motions.get_dct_matrix(t_his + t_pred)
                idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
                outputs_t = outputs.view(-1, config.dct_n).transpose(0, 1)
                outputs_g = torch.matmul(idct_m[:, 0:config.dct_n], outputs_t).transpose(0, 1).contiguous().view(-1, c, t_his + t_pred).transpose(1,2)

                Y = targets.permute(1, 0, 2).contiguous()
                Y_g = outputs_g.permute(1, 0, 2).contiguous()
                loss, losses = loss_function(config, Y_g, Y, device, dtype, all_seqs)
                optimizer.zero_grad()
                loss.backward()
                if config.gcn_max_norm:
                    nn.utils.clip_grad_norm(refine.parameters(), max_norm=1)
                optimizer.step()

                train_losses += losses
                total_num_sample += 1

            dt = time.time() - st
            train_losses /= total_num_sample
            losses_str = ' '.join(['{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, train_losses)])
            logger_refine.info('====> Epoch: {} Time: {:.2f} {} lr: {:.5f}'.format(epoch, dt, losses_str, lr_now))
            for name, loss in zip(loss_names, train_losses):
                self.log_writer_refine.add_scalar('refine_' + name, loss, epoch)

            if config.save_model_interval > 0 and (epoch + 1) % config.save_model_interval == 0:
                with to_cpu(refine):
                    cp_path = self.pretrained_model_dir_refine % (epoch + 1)
                    model_cp = {'model_dict': refine.state_dict()}
                    pickle.dump(model_cp, open(cp_path, 'wb'))



    def eval(self):
        device = self.device
        dtype = self.dtype
        config = self.config
        torch.set_grad_enabled(False)
        logger_test = create_logger(os.path.join(self.model_dir_log, "log_eval.txt"))

        algos = []
        all_algos = ['refine', 'diff']
        for algo in all_algos:
            iter_algo = 'iter_%s' % algo
            num_algo = 'eval_at_%s' % algo
            setattr(config, iter_algo, getattr(config, num_algo))
            algos.append(algo)
        vis_algos = algos.copy()

        # get dataset
        dataset = self.dataset_test

        # get models
        model_generator = {
            'refine': get_refine_model,
            'diff': get_diff_model
        }
        models = {}
        for algo in all_algos:
            models[algo] = model_generator[algo](config, dataset.traj_dim)
            if algo == 'diff':
                model_path = self.pretrained_model_dir_diff % getattr(config, f'iter_{algo}')
            elif algo == 'refine':
                model_path = self.pretrained_model_dir_refine % getattr(config, f'iter_{algo}')
            print(f'loading {algo} model from checkpoint: {model_path}')
            model_cp = pickle.load(open(model_path, "rb"))
            models[algo].load_state_dict(model_cp['model_dict'])
            models[algo].to(device)
            models[algo].eval()

        # visualization or compute statistics
        if config.mode_test == 'vis':
            visualize(config, models, dataset, self.device, self.dtype, algos, self.dir_out)
        elif config.mode_test == 'stats':
            compute_stats(config, models, dataset, self.device, self.dtype, vis_algos, logger_test, self.dir_out)





    def _build(self):
        self._build_dir()

        if self.config.mode == "train_diff":
            self._build_train_loader()
        elif self.config.mode == "generate_diff":
            self._build_train_loader()
        elif self.config.mode == "train_refine":
            self._build_custom_loader()
        elif self.config.mode == "test":
            self._build_val_loader()

        self._build_model()
        self._build_optimizer()

        print("> Everything built. Have fun :)")


    def _build_dir(self):
        self.model_dir = osp.join("./results", self.config.dataset)
        os.makedirs(self.model_dir, exist_ok=True)
        self.log_writer_diff = SummaryWriter(osp.join(self.model_dir, "tb")) if self.config.mode == "train_diff" else None
        self.log_writer_refine = SummaryWriter(osp.join(self.model_dir, "tb")) if self.config.mode == "train_refine" else None
        self.model_dir_log = osp.join(self.model_dir, "log")
        os.makedirs(self.model_dir_log, exist_ok=True)
        self.dir_out = osp.join(self.model_dir, "out")
        os.makedirs(self.dir_out, exist_ok=True)
        self.logger = create_logger(osp.join(self.model_dir_log, "log_diff.txt"))
        tmp = osp.join(self.model_dir, "models")
        os.makedirs(tmp, exist_ok=True)
        self.pretrained_model_dir_diff = osp.join(tmp, "diffMotion_%04d.p")
        self.pretrained_model_dir_refine = osp.join(tmp, "refine_%04d.p")
        self.generated_motions = osp.join(self.model_dir, "generated_diff")

        print("> Directory built!")


    def _build_optimizer(self):
        if self.config.mode == "train_diff":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
            self.scheduler = get_scheduler(self.optimizer, policy='lambda', nepoch_fix=self.config.num_diff_epoch_fix,
                                           nepoch=self.config.num_diff_epoch)
        elif self.config.mode == "train_refine":
            self.optimizer = optim.Adam(self.refine.parameters(), lr=self.config.refine_lr)

        print("> Optimizer built!")


    def _build_model(self):
        """ Define Model """
        config = self.config
        if self.config.mode == "train_diff":
            model = get_diff_model(config, self.dataset_train.traj_dim)
            self.model = model.to(self.device)
            print("> Model built!")

        elif self.config.mode == "train_refine":
            refine = get_refine_model(config, 48)
            self.refine = refine.to(self.device)
            print("> Model built!")


        # loading model from checkpoint
        if config.iter_start_diff > 0:
            if self.config.mode == "train_diff":
                cp_path = self.pretrained_model_dir_diff % config.iter_start_diff
                print('loading diff model from checkpoint: %s' % cp_path)
                model_cp = pickle.load(open(cp_path, "rb"))
                model.load_state_dict(model_cp['model_dict'])
        if config.iter_start_refine > 0:
            if self.config.mode == "train_refine":
                cp_path = self.pretrained_model_dir_refine % config.iter_start_refine
                print('loading refine model from checkpoint: %s' % cp_path)
                model_cp = pickle.load(open(cp_path, "rb"))
                model.load_state_dict(model_cp['model_dict'])



    def _build_train_loader(self):
        config = self.config
        print(">>> loading data...")

        t_his = config.obs_frames
        t_pred = config.pred_frames

        dataset_cls = DatasetH36M
        dataset = dataset_cls('train', t_his, t_pred, actions='all', use_vel=config.use_vel)
        if config.normalize_data:
            dataset.normalize_data()

        self.dataset_train = dataset


    def _build_custom_loader(self):
        config = self.config
        print(">>> loading data...")

        t_his = config.obs_frames
        t_pred = config.pred_frames
        train_dataset = CustomH36M(config, path_to_data=self.generated_motions, input_n=t_his, output_n=t_pred, dct_used=config.dct_n)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config.refine_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True)
        self.dataset_custom = train_dataset
        self.dataloader_custom = train_loader


    def _build_val_loader(self):
        config = self.config
        t_his = config.obs_frames
        t_pred = config.pred_frames

        dataset_cls = DatasetH36M
        dataset = dataset_cls('test', t_his, t_pred, actions='all', use_vel=config.use_vel)
        if config.normalize_data:
            dataset.normalize_data()

        self.dataset_test = dataset

        print("> Dataset built!")

















