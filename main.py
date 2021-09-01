import numpy as np
np.random.seed(0)
import torch
import time
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import argparse

def parse_args():
    # input arguments
    parser = argparse.ArgumentParser(description='BiHIN')
    parser.add_argument('--gpu_num', nargs='?', default='0')
    parser.add_argument('--embedder', nargs='?', default='BiHIN')
    parser.add_argument('--dataset', nargs='?', default='acm')
    parser.add_argument('--hopK', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--nb_epochs', type=int, default=10000)

    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--hid_units', type=int, default=128)
    parser.add_argument('--out_ft', type=int, default=64)
    parser.add_argument('--lr', type = float, default = 0.0005)
    parser.add_argument('--l2_coef', type=float, default=0.0001)
    parser.add_argument('--drop_prob', type=float, default=0.5)
    parser.add_argument('--reg_coef', type=float, default=0.001)
    parser.add_argument('--sup_coef', type=float, default=0.0)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--activation', nargs='?', default='relu')

    parser.add_argument('--isBias', action='store_true', default=False)
    parser.add_argument('--isAtt', action='store_true', default=False)

    return parser.parse_known_args()

def printConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))
    print(args_names)
    print(args_vals)

def main():
    args, unknown = parse_args()

    if args.embedder == 'BiHIN':
        # from models import DMGI
        embedder = BiHIN(args, hin)

    # elif args.embedder == 'DGI':
    #     from models import DGI
    #     embedder = DGI(args)

    start = time.time()
    embedder.training()
    print('time (s):%.2f'%(time.time()-start))


if __name__ == '__main__':
    main()
