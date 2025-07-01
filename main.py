import yaml
import torch
import random
import argparse
import numpy as np

from train import Trainer

if __name__ == "__main__":

    # Set random seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    
    parser = argparse.ArgumentParser( description="PyTorch implementation of Temporal Segment Networks" )
    parser.add_argument('--clip-gradient', '--gd', default=None, type=float, metavar='W',
                    help='gradient norm clipping (default: disabled)')
    parser.add_argument('--save-name', type=str, default='QAQ', help='save name')
    parser.add_argument('--config', type=str, default='./configs/Train_config.yaml', help='config file')
    args = parser.parse_args()
    args = vars(args)
    # print(args)

    with open(args["config"], "r", encoding="utf-8") as f:
        args.update(yaml.safe_load(f))
    args = argparse.Namespace(**args)

    trainer = Trainer(args)
    trainer.run()