import argparse
from random import choices

__all__ = ['parse_arguments_seg']


def parse_arguments_seg():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', '-d', type=str, default='brats2021', choices=['brats2021'])
    parser.add_argument('--model', '-m', type=str, default='cyclegan', choices=['cyclegan', 'munit', 'unit'])
    parser.add_argument('--source-domain', '-s', default='t1', choices=['t1', 't2', 'flair'])
    parser.add_argument('--target-domain', '-t', default='t2', choices=['t1', 't2', 'flair'])

    parser.add_argument('--noise-type', type=str, default='gaussian', choices=['normal', 'gaussian', 'slight', 'severe'])
    parser.add_argument('--gpu-id', '-g', type=str, default=None)
    parser.add_argument('--debug', action='store_true', default=None)
    parser.add_argument('--msl-stats', action='store_true', help='mask stastical learning')
    parser.add_argument('--msl-assigned', action='store_true', help='mask assigned flag')
    parser.add_argument('--msl-assigned-value', type=float, help='msl assigned value')
    parser.add_argument('--msl-path', type=str, default=None, help='mask side length storage path')
    parser.add_argument('--delta-diff', type=float, default=None, help='mask side length difference vairation thereshold value')
    parser.add_argument('--num-epochs', type=int, default=None)
    parser.add_argument('--lambda-recon', type=float, default=1.0, help='weight for reconstruction loss')
    parser.add_argument('--lambda-contrastive', type=float, default=1.0, help='weight for contrastive loss')
    parser.add_argument('--lambda-hf', type=float, default=1.0, help='weight for high frequency part')
    parser.add_argument('--lambda-lf', type=float, default=1.0, help='weight for low frequency part')
    parser.add_argument('--pair-num', '-pn', type=int, default=10000)
    parser.add_argument('--test-model', type=str, default='cyclegan', choices=['cyclegan','munit','unit'])
    #parser.add_argument('--mode', type=str, default='train', choices=['train', 'pred', 'trainpred'])
    parser.add_argument('--diff-method', type=str, default=None, choices=['l1', 'l2', 'cos'])
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--step-size', type=int, default=None, help='learning rate will be adjust for epoch numbers')
    parser.add_argument('--gamma', type=float, default=None, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--beta1', type=float, default=None, help='Adam Optimizer parameter')
    parser.add_argument('--beta2', type=float, default=None, help='Adam Optimizer parameter')
    parser.add_argument('--fid', action='store_true', default=True)
    #segmentation
    parser.add_argument('--segment', action='store_true', default=None, help='segmentation method evaluation or not')
    parser.add_argument('--seg-method', type=str, choices=['2d', '3d'])
    parser.add_argument('--seg-model', type=str, choices=['ex_unet', 'unet'])
    parser.add_argument('--max-filters-2d', type=int, default=999)

    args = parser.parse_args()
    return args