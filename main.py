import argparse
import traceback
import time
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
from runners import *


def parse_args_and_config(args):
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    
    parser.add_argument('--runner', type=str, default='Naive3p_Runner', help='Runner for corresponding probelm setting.')
    parser.add_argument('--config', type=str, default='naive_3p.yml',  help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--logs', type=str, default='log', help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, default='naive_3_period', help='A string for documentation purpose')
    parser.add_argument('--test', action='store_true', help='Test model.')
    parser.add_argument('--num_test_batch', type=int, default=4, help='Number of batches of path to generate.')
    parser.add_argument('--resume_training',  type=int, default=0, help='Resume training from certain epoch')
    parser.add_argument('-o', '--plot_folder', type=str, default='plots', help="The directory of plot outputs")
    
    args = parser.parse_args(args)
    args.log = os.path.join('logs', args.doc)
    
    
    if not args.test:
        with open(os.path.join('configs', args.config), 'r') as f:
            config = yaml.safe_load(f)
        new_config = dict2namespace(config)
        
        with open(os.path.join(args.log, 'config.yml'), 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False)
    else:
        with open(os.path.join(args.log, 'config.yml'), 'r') as f:
            config = yaml.unsafe_load(f)
        new_config = config


    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace



def main():
    args, config = parse_args_and_config(['--runner', 'Fb3p_Runner', '--config', 'finite_banking_3p.yml', '--doc', 'finite_banking_3_period'])
                                        #'--test', '--num_test_batch', '4','--resume_training', '200'])
    runner = eval(args.runner)(args, config)
    if not args.test:
        runner.train()
    else: 
        runner.test()
    return 0


if __name__ == '__main__':
    sys.exit(main())