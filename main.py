from DiffHM import DiffHM
import argparse
import os
import yaml
# from pprint import pprint
from easydict import EasyDict
import numpy as np
import torch



def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of MotionDiff')
    parser.add_argument('--config', default='configs/baseline.yaml')
    return parser.parse_args()


def main():

    # parse arguments and load config
    args = parse_args()

    with open(args.config) as f:
       config = yaml.safe_load(f)

    for k, v in vars(args).items():
       config[k] = v

    config = EasyDict(config)

    """setup"""
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)


    agent = DiffHM(config)

    if config["mode"] == 'train_diff':
        agent.train_diff()
    elif config["mode"] == 'train_refine':
        agent.train_refine()
    elif config["mode"] == 'generate_diff':
        agent.generate_diff()
    else:
        agent.eval()





if __name__ == '__main__':
    main()
