import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from visualization.visualization import *
from utils import *
from utils.logger import AverageMeter
import csv
import pickle
import os.path as osp
from torch.nn import functional as F
import dataset.utils as dutil
from torch.autograd import Variable

###########################################################################################################
###########################################################################################################
################################    Visualization    ######################################################
###########################################################################################################
###########################################################################################################

def denomarlize(dataset, *data):
    out = []
    for x in data:
        x = x * dataset.std + dataset.mean
        out.append(x)
    return out


def get_prediction(config, models, data, algo, sample_num, device, dtype, num_seeds=1, concat_hist=True):
    t_his = config.obs_frames
    t_pred = config.pred_frames
    traj_np = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
    traj = torch.tensor(traj_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous()
    X = traj[:t_his]

    if algo == 'diff':
        X = X.repeat((1, sample_num, 1))
        Y = models[algo].generate(X)
    elif algo == 'refine':
        X = X.repeat((1, sample_num, 1))
        Y = models['diff'].generate(X)
        Y = torch.cat((X, Y), dim=0)
        # DCT
        c = Y.shape[-1]  # 48
        Y = Y.permute(1, 2, 0)
        Y = Y.reshape(-1, t_his + t_pred)
        Y = Y.transpose(0, 1)
        dct_m_in, _ = dutil.dataset_generated_motions.get_dct_matrix(t_his + t_pred)
        dct_m_in = Variable(torch.from_numpy(dct_m_in)).float().cuda()
        input_dct_seq = torch.matmul(dct_m_in[0 : config.dct_n, :], Y)
        input_dct_seq = input_dct_seq.transpose(0, 1).reshape([config.nk, c, config.dct_n])
        outputs = models[algo](input_dct_seq)
        # IDCT
        _, idct_m = dutil.dataset_generated_motions.get_dct_matrix(t_his + t_pred)
        idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
        outputs_t = outputs.view(-1, config.dct_n).transpose(0, 1)
        Y = torch.matmul(idct_m[:, 0:config.dct_n], outputs_t).transpose(0, 1).contiguous().view(-1, outputs.shape[1], t_his + t_pred).transpose(1, 2)
        Y = Y.permute(1, 0, 2).contiguous()
        Y = Y[t_his:]

    if concat_hist:
        Y = torch.cat((X, Y), dim=0)
    Y = Y.permute(1, 0, 2).contiguous().cpu().numpy()
    if Y.shape[0] > 1:
        Y = Y.reshape(-1, sample_num, Y.shape[-2], Y.shape[-1])
    else:
        Y = Y[None, ...]
    return Y



def visualize(config, model, dataset, device, dtype, algos, out_path):

    def post_process(config, pred, data):
        pred = pred.reshape(pred.shape[0], pred.shape[1], -1, 3)
        if config.normalize_data:
            pred = denomarlize(dataset, pred)
        pred = np.concatenate((np.tile(data[..., :1, :], (pred.shape[0], 1, 1, 1)), pred), axis=2)
        pred[..., :1, :] = 0
        return pred

    def pose_generator(config, model, dataset, device, dtype):

        while True:
            data = dataset.sample()

            # gt
            gt = data[0].copy()
            gt[:, :1, :] = 0
            poses = {'context': gt, 'gt': gt}
            # vae
            for algo in vis_algos:
                pred = get_prediction(config, model, data, algo, config.nk, device, dtype)[0]
                pred = post_process(config, pred, data)
                for i in range(pred.shape[0]):
                    poses[f'{algo}_{i}'] = pred[i]

            yield poses

    vis_algos = algos
    t_his = config.obs_frames
    # t_pred = config.pred_frames
    pose_gen = pose_generator(config, model, dataset, device, dtype)
    out = osp.join(out_path, 'video.mo4')
    render_animation(dataset.skeleton, pose_gen, vis_algos, t_his, ncol=12, output=out)




###########################################################################################################
###########################################################################################################
###################################    Statistics    ######################################################
###########################################################################################################
###########################################################################################################

def get_gt(data, t_his):
    gt = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
    return gt[:, t_his:, :]


def get_multimodal_gt(config, dataset_test, logger_test):
    all_data = []
    t_his = config.obs_frames
    t_pred = config.pred_frames
    data_gen = dataset_test.iter_generator(t_his)
    for data in data_gen:
        data = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
        all_data.append(data)
    all_data = np.concatenate(all_data, axis=0)
    all_start_pose = all_data[:, t_his - 1, :]
    pd = squareform(pdist(all_start_pose))
    traj_gt_arr = []
    num_mult = []
    for i in range(pd.shape[0]):
        ind = np.nonzero(pd[i] < config.multimodal_threshold)
        traj_gt_arr.append(all_data[ind][:, t_his:, :])
        num_mult.append(len(ind[0]))

    # num_mult = np.array(num_mult)
    # logger_test.info('')
    # logger_test.info('')
    # logger_test.info('=' * 80)
    # logger_test.info(f'#1 future: {len(np.where(num_mult == 1)[0])}/{pd.shape[0]}')
    # logger_test.info(f'#<10 future: {len(np.where(num_mult < 10)[0])}/{pd.shape[0]}')
    return traj_gt_arr


"""metrics"""

def compute_diversity(pred, *args):
    if pred.shape[0] == 1:
        return 0.0
    dist = pdist(pred.reshape(pred.shape[0], -1))
    a, idx1 = torch.sort(torch.tensor(dist), descending=True)
    diversity = a[:50].mean().item()
    # diversity = dist.mean().item()
    return diversity


def compute_ade(pred, gt, *args):
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=2).mean(axis=1)
    return dist.min()


def compute_fde(pred, gt, *args):
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=2)[:, -1]
    return dist.min()


def compute_mmade(pred, gt, gt_multi):
    gt_dist = []
    for gt_multi_i in gt_multi:
        dist = compute_ade(pred, gt_multi_i)
        gt_dist.append(dist)
    gt_dist = np.array(gt_dist).mean()
    return gt_dist


def compute_mmfde(pred, gt, gt_multi):
    gt_dist = []
    for gt_multi_i in gt_multi:
        dist = compute_fde(pred, gt_multi_i)
        gt_dist.append(dist)
    gt_dist = np.array(gt_dist).mean()
    return gt_dist



def compute_stats(config, model, dataset, device, dtype, algos, logger_test, out_path):
    stats_algos = algos
    t_his = config.obs_frames
    # t_pred = config.pred_frames
    num_seeds = config.num_seeds

    stats_func = {'Diversity': compute_diversity, 'ADE': compute_ade,
                  'FDE': compute_fde, 'MMADE': compute_mmade, 'MMFDE': compute_mmfde}
    stats_names = list(stats_func.keys())
    stats_meter = {x: {y: AverageMeter() for y in stats_algos} for x in stats_names}
    # generate multi-modal ground truth (only in test stage)
    traj_gt_arr = get_multimodal_gt(config, dataset, logger_test)

    data_gen = dataset.iter_generator(step=t_his)
    num_samples = 0
    for i, data in enumerate(data_gen):
        num_samples += 1
        gt = get_gt(data, t_his)
        gt_multi = traj_gt_arr[i]
        for algo in stats_algos:
            pred = get_prediction(config, model, data, algo, sample_num=config.nk, device=device, dtype=dtype, num_seeds=num_seeds, concat_hist=False)
            for stats in stats_names:
                val = 0
                for pred_i in pred:
                    val += stats_func[stats](pred_i, gt, gt_multi) / num_seeds
                stats_meter[stats][algo].update(val)
        print('-' * 80)
        for stats in stats_names:
            str_stats = f'{num_samples:04d} {stats}: ' + ' '.join([f'{x}: {y.val:.4f}({y.avg:.4f})' for x, y in stats_meter[stats].items()])
            print(str_stats)

    logger_test.info('=' * 80)
    for stats in stats_names:
        str_stats = f'Total {stats}: ' + ' '.join([f'{x}: {y.avg:.4f}' for x, y in stats_meter[stats].items()])
        logger_test.info(str_stats)
    logger_test.info('=' * 80)

    with open('%s/stats_%s.csv' % (out_path, config.nk), 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['Metric'] + algos)
        writer.writeheader()
        for stats, meter in stats_meter.items():
            new_meter = {x: y.avg for x, y in meter.items()}
            new_meter['Metric'] = stats
            writer.writerow(new_meter)




#########################################################################################
##########################################################################################
def lr_decay(optimizer, lr_now, gamma):
    lr = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def loss_function(config, Y_g, Y, device, dtype, X):
    t_his = config.obs_frames
    #  loss
    JL = joint_loss(config, Y_g) if config.lambda_j > 0 else 0.0
    RECON = recon_loss(config, Y_g, Y) if config.lambda_recon > 0 else 0.0
    X = X.reshape(-1, X.shape[2], X.shape[3]).permute(1, 0, 2)
    loss_r = RECON * config.lambda_recon + JL * config.lambda_j
    return loss_r, np.array([loss_r.item()])


def joint_loss(config, Y_g):
    loss = 0.0
    Y_g = Y_g.permute(1, 0, 2).contiguous()
    Y_g = Y_g.view(Y_g.shape[0] // config.nk, config.nk, -1)
    for Y in Y_g:
        dist = F.pdist(Y, 2) ** 2
        loss += (-dist / config.d_scale).exp().mean()
    loss /= Y_g.shape[0]
    return loss


def recon_loss(config, Y_g, Y):
    Y_g = Y_g.view(Y_g.shape[0], -1, config.nk, Y_g.shape[2])
    diff = Y_g - Y.unsqueeze(2)
    dist = diff.pow(2).sum(dim=-1).sum(dim=0)
    loss_recon = dist.min(dim=1)[0].mean()
    return loss_recon / 100000.0




