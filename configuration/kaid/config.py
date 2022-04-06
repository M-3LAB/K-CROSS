import argparse
from random import choices

__all__ = ['parse_arguments_kaid']


def parse_arguments_kaid():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', '-d', type=str, default='brats2021', choices=['ixi', 'brats2021'])
    parser.add_argument('--source-domain', '-s', type=str, default='t1', choices=['t1', 't2', 'pd', 'flair'])
    parser.add_argument('--target-domain', '-t', type=str, default='t2', choices=['t1', 't2', 'pd', 'flair'])
    parser.add_argument('--data-num', '-dn', type=int, default=None)
    parser.add_argument('--data-path', '-dp', type=str, default=None)
    parser.add_argument('--valid-path', '-vp', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=None)

    parser.add_argument('--gpu-id', '-g', type=str, default=None)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--normalized-method', type=str, default=None, choices=['forward', 'backward', 'ortho'])
    parser.add_argument('--num-epochs', type=int, default=30)
    parser.add_argument('--diff-method', type=str, default=None, choices=['l1', 'l2', 'cos', 'freq'])
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--fid', action='store_true', default=True)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--validate', action='store_true', default=False)
    #Gaussian Noise
    parser.add_argument('--noisy-loss', action='store_true', default=None)
    parser.add_argument('--mu', type=float, default=None)
    parser.add_argument('--sigma', type=float, default=None)

    # method
    parser.add_argument('--method', type=str, choices=['combined', 'complex', 'normal'])
    # inference 
    parser.add_argument('--nirps-path', type=str, default='./nirps_dataset', help='nirps data path')
    parser.add_argument('--test-model', type=str, default=None, choices=['cyclegan','munit','unit'])
    parser.add_argument('--infer-range', type=str, default=None, choices=['all', 'ixi', 'brats2021'])

    args = parser.parse_args()
    return args

