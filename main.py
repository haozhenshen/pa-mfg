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
    
    parser.add_argument('--runner', type=str, default='Runner', help='Runner for corresponding probelm setting.')
    parser.add_argument('--config', type=str, default='finite_banking_3p.yml',  help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--logs', type=str, default='log', help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, default='finite_banking_3_period', help='A string for documentation purpose')

    parser.add_argument('--test', action='store_true', help='Whether to test the model')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    parser.add_argument('-o', '--plot_folder', type=str, default='plots', help="The directory of plot outputs")
    
    args = parser.parse_args(args)
    args.log = os.path.join('logs', args.doc)
    
    
    if not args.test:
        with open(os.path.join('configs', args.config), 'r') as f:
            config = yaml.safe_load(f)
            print(type(config))
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
    torch.cuda.empty_cache()
    args, config = parse_args_and_config(["--runner", "Runner", "--config", "finite_banking_3p.yml", "--plot_folder", "'plots'"])
    print(type(config.training.plot_freq))
    runner = eval(args.runner)(args, config)
    if not args.test:
          runner.train()
    # else:
    #   runner.test()
    # return 0


if __name__ == '__main__':
    #sys.exit(main())
    main()